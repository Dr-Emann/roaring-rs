//! Ported from CRoaring and arXiv:1709.07821
//! Lemire et al, Roaring Bitmaps: Implementation of an Optimized Software Library
//!
//! Prior work: Schlegel et al., Fast Sorted-Set Intersection using SIMD Instructions
//!
//! Rust port notes:
//! The algorithms are generic over the vector width, and use the native vector width of the SIMD
//! level (`Simd::u16s`) first: 8 lanes for 128-bit SIMD, 16 lanes for AVX2 and 32 lanes for
//! AVX-512. Where an array is too short for the native width, they step down to 128-bit vectors
//! (`u16x8`), and only then to scalar code. Every vector is made of 128-bit blocks of 8 lanes,
//! which some operations work on independently.
//!
//! Like CRoaring, the intersection and difference use the x86 PCMPISTRM instruction (falling back
//! to PCMPESTRM while a zero may be present) on each pair of 128-bit blocks where SSE4.2 is
//! available, up to 256-bit vectors. Wider vectors and other SIMD levels use a portable all-pairs
//! comparison (see `matrix_cmp_u16`) instead.
//!
//! The union and symmetric difference merge vectors with a bitonic merge network, which takes
//! `log2(N)` steps for `N` lanes.
//!
//! The public functions select a SIMD level at runtime with `fearless_simd::dispatch!`, and call
//! implementations generic over the SIMD level. Those are annotated with `#[simd]`, so that they
//! are compiled with the matching target features enabled.
//!
//! The small helpers they call are `#[inline(always)]` rather than `#[simd]`: they get the target
//! features by being inlined, and `#[simd]` does not force inlining, which can leave them out of
//! line in the hot loops.

#![cfg(feature = "simd")]

use super::scalar;
use crate::bitmap::store::array_store::visitor::BinaryOperationVisitor;
use fearless_simd::prelude::*;
use fearless_simd::{dispatch, u16x8, Level};
use fearless_simd_macros::simd;

/// A vector of `u16`s made of 128-bit blocks, which the algorithms are generic over
///
/// This is implemented by the native-width vector of each SIMD level (`Simd::u16s`), and by
/// `u16x8`.
pub trait U16Vector<S: Simd>: SimdBase<S, Element = u16, Block = u16x8<S>> {}
impl<S: Simd, V: SimdBase<S, Element = u16, Block = u16x8<S>>> U16Vector<S> for V {}

/// A native-width vector of `u16`s
type U16s<S> = <S as Simd>::u16s;

/// The number of lanes in a block of a vector (a `u16x8`)
const BLOCK_LANES: usize = 8;
/// The largest number of lanes in a vector (a `u16x32` for AVX-512), used to size buffers
const MAX_LANES: usize = 32;

/// The number of lanes in a native-width vector
#[inline(always)]
fn native_lanes<S: Simd>() -> usize {
    <U16s<S> as SimdBase<S>>::LEN
}

/// A bitmask with a bit set for each of `lanes` lanes
#[inline(always)]
fn lanes_bitmask(lanes: usize) -> u64 {
    u64::MAX >> (64 - lanes)
}

/// The SIMD level to use.
///
/// With `std`, this is detected at runtime on x86/x86-64 (and cached by `fearless_simd`).
/// Otherwise, it is determined by the target features enabled at compile time.
#[inline]
fn level() -> Level {
    // `Level::new()` is inlined, unlike `Level::try_detect()`
    #[cfg(feature = "std")]
    return Level::new();
    #[cfg(not(feature = "std"))]
    return Level::baseline();
}

// a one-pass union algorithm
pub fn or(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    dispatch!(level(), simd => or_impl(simd, lhs, rhs, visitor))
}

#[simd]
fn or_impl<S: Simd>(simd: S, lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // returns `new`, with a mask of its values which were not already written,
    // assuming that the previously written vector was `old`
    #[inline(always)]
    fn handle_vector<S: Simd, V: U16Vector<S>>(old: V, new: V) -> (V, u64) {
        let tmp: V = shift_in_1(old, new);
        let mask = !tmp.simd_eq(new).to_bitmask() & lanes_bitmask(V::LEN);
        (new, mask)
    }

    // The merge state is a whole vector, so the vector width is chosen for the whole merge: the
    // native width if both arrays have at least that many values, otherwise 128 bits. With fewer
    // values than a 128-bit vector, the scalar algorithm is faster
    let n = native_lanes::<S>();
    if (lhs.len() >= n) && (rhs.len() >= n) {
        merge_vectors::<S, U16s<S>>(simd, lhs, rhs, visitor, handle_vector);
    } else if (lhs.len() >= BLOCK_LANES) && (rhs.len() >= BLOCK_LANES) {
        merge_vectors::<S, u16x8<S>>(simd, lhs, rhs, visitor, handle_vector);
    } else {
        scalar::or(lhs, rhs, visitor);
        return;
    }

    // `u16::MAX` is skipped, as it is used as padding
    if lhs.last() == Some(&u16::MAX) || rhs.last() == Some(&u16::MAX) {
        visitor.visit_scalar(u16::MAX);
    }
}

pub fn and(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    dispatch!(level(), simd => and_impl(simd, lhs, rhs, visitor))
}

#[simd]
fn and_impl<S: Simd>(simd: S, lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // Intersect with native-width vectors, then the rest with 128-bit vectors, then the rest with
    // scalar code
    let (i, j) = and_vectors::<S, U16s<S>>(simd, lhs, rhs, visitor);
    let (lhs, rhs) = (&lhs[i..], &rhs[j..]);
    let (i, j) = if native_lanes::<S>() > BLOCK_LANES {
        and_vectors::<S, u16x8<S>>(simd, lhs, rhs, visitor)
    } else {
        (0, 0)
    };

    // intersect the tail using scalar intersection
    scalar::and(&lhs[i..], &rhs[j..], visitor);
}

/// Intersects whole vectors of `lhs` and `rhs`, and returns the number of values of each which
/// were handled
#[inline(always)]
fn and_vectors<S: Simd, V: U16Vector<S>>(
    simd: S,
    lhs: &[u16],
    rhs: &[u16],
    visitor: &mut impl BinaryOperationVisitor,
) -> (usize, usize) {
    let n = V::LEN;
    let st_a = (lhs.len() / n) * n;
    let st_b = (rhs.len() / n) * n;

    let mut i: usize = 0;
    let mut j: usize = 0;
    if (i < st_a) && (j < st_b) {
        let mut v_a: V = load(simd, &lhs[i..]);
        let mut v_b: V = load(simd, &rhs[j..]);
        loop {
            let may_contain_zero = lhs[i] == 0 || rhs[j] == 0;
            let mask = matrix_cmp_bitmask(v_a, v_b, &lhs[i..], &rhs[j..], may_contain_zero);
            visitor.visit_vector(v_a, mask);

            let a_max: u16 = lhs[i + n - 1];
            let b_max: u16 = rhs[j + n - 1];
            if a_max <= b_max {
                i += n;
                if i == st_a {
                    break;
                }
                v_a = load(simd, &lhs[i..]);
            }
            if b_max <= a_max {
                j += n;
                if j == st_b {
                    break;
                }
                v_b = load(simd, &rhs[j..]);
            }
        }
    }
    (i, j)
}

// a one-pass xor algorithm
pub fn xor(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    dispatch!(level(), simd => xor_impl(simd, lhs, rhs, visitor))
}

#[simd]
fn xor_impl<S: Simd>(simd: S, lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // returns the vector to write, and a mask of the values to write from it,
    // omitting repeated values assuming that previously written vector was "old"
    #[inline(always)]
    fn handle_vector<S: Simd, V: U16Vector<S>>(old: V, new: V) -> (V, u64) {
        let tmp1: V = shift_in_2(old, new);
        let tmp2: V = shift_in_1(old, new);
        let eq_l = tmp2.simd_eq(tmp1);
        let eq_r = tmp2.simd_eq(new);
        let eq_l_or_r = eq_l | eq_r;
        let mask: u64 = eq_l_or_r.to_bitmask();
        (tmp2, !mask & lanes_bitmask(V::LEN))
    }

    // Each value is written once the value after it is known, so write the last value of the
    // last vector, which is followed by padding
    #[inline(always)]
    fn merge_and_finish<S: Simd, V: U16Vector<S>>(
        simd: S,
        lhs: &[u16],
        rhs: &[u16],
        visitor: &mut impl BinaryOperationVisitor,
    ) {
        let v_max: V = merge_vectors(simd, lhs, rhs, visitor, handle_vector);
        let (v, m) = handle_vector(v_max, V::splat(simd, u16::MAX));
        visit_without_max(visitor, v, m);
    }

    // The vector width is chosen for the whole merge, as for `or`
    let n = native_lanes::<S>();
    if (lhs.len() >= n) && (rhs.len() >= n) {
        merge_and_finish::<S, U16s<S>>(simd, lhs, rhs, visitor);
    } else if (lhs.len() >= BLOCK_LANES) && (rhs.len() >= BLOCK_LANES) {
        merge_and_finish::<S, u16x8<S>>(simd, lhs, rhs, visitor);
    } else {
        scalar::xor(lhs, rhs, visitor);
        return;
    }

    // `u16::MAX` is skipped, as it is used as padding
    if (lhs.last() == Some(&u16::MAX)) != (rhs.last() == Some(&u16::MAX)) {
        visitor.visit_scalar(u16::MAX);
    }
}

/// The vectors of a sorted array, where the last vector is padded with `u16::MAX` if the length
/// of the array isn't a multiple of the number of lanes
struct PaddedVectors<'a> {
    values: &'a [u16],
    /// The number of lanes in a vector
    lanes: usize,
    /// The number of vectors which aren't padded
    full: usize,
    /// The number of vectors, including the padded one
    len: usize,
    padded: [u16; MAX_LANES],
}

impl<'a> PaddedVectors<'a> {
    #[inline(always)]
    fn new(values: &'a [u16], lanes: usize) -> Self {
        let full = values.len() / lanes;
        let rest = &values[full * lanes..];
        let mut padded = [u16::MAX; MAX_LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let len = full + usize::from(!rest.is_empty());
        PaddedVectors { values, lanes, full, len, padded }
    }

    /// The first value of vector `i`
    #[inline(always)]
    fn first(&self, i: usize) -> u16 {
        self.values[i * self.lanes]
    }

    /// Loads vector `i`
    #[inline(always)]
    fn load<S: Simd, V: U16Vector<S>>(&self, simd: S, i: usize) -> V {
        if i < self.full {
            load(simd, &self.values[i * self.lanes..])
        } else {
            load(simd, &self.padded)
        }
    }
}

/// Merges the vectors of the sorted arrays `lhs` and `rhs`, which must not be empty, visiting each
/// merged vector with the mask from `handle_vector(previous vector, vector)`, except for lanes
/// which are `u16::MAX`
///
/// Returns the last vector, which is the larger half of the last merge, and has been visited
/// with the vector before it.
#[inline(always)]
fn merge_vectors<S: Simd, V: U16Vector<S>>(
    simd: S,
    lhs: &[u16],
    rhs: &[u16],
    visitor: &mut impl BinaryOperationVisitor,
    handle_vector: impl Fn(V, V) -> (V, u64),
) -> V {
    // The last vector of each array is padded with `u16::MAX`, so that the vectorized merge
    // handles every value: `u16::MAX` is skipped when visiting merged vectors
    let a = PaddedVectors::new(lhs, V::LEN);
    let b = PaddedVectors::new(rhs, V::LEN);

    let [mut v_min, mut v_max]: [V; 2] = simd_merge_u16(a.load(simd, 0), b.load(simd, 0));
    let (v, m) = handle_vector(V::splat(simd, u16::MAX), v_min);
    visit_without_max(visitor, v, m);
    let mut v_prev = v_min;

    let mut i = 1;
    let mut j = 1;
    loop {
        // Merge the vector which starts with the smaller value next
        let v: V = if i < a.len && (j == b.len || a.first(i) <= b.first(j)) {
            i += 1;
            a.load(simd, i - 1)
        } else if j < b.len {
            j += 1;
            b.load(simd, j - 1)
        } else {
            break;
        };
        [v_min, v_max] = simd_merge_u16(v, v_max);
        let (v, m) = handle_vector(v_prev, v_min);
        visit_without_max(visitor, v, m);
        v_prev = v_min;
    }

    let (v, m) = handle_vector(v_prev, v_max);
    visit_without_max(visitor, v, m);
    v_max
}

/// Visits the lanes of `v` selected by `mask`, except for lanes which are `u16::MAX`
#[inline(always)]
fn visit_without_max<S: Simd, V: U16Vector<S>>(
    visitor: &mut impl BinaryOperationVisitor,
    v: V,
    mask: u64,
) {
    let mask = mask & !v.simd_eq(u16::MAX).to_bitmask();
    visitor.visit_vector(v, mask);
}

pub fn sub(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    dispatch!(level(), simd => sub_impl(simd, lhs, rhs, visitor))
}

#[simd]
fn sub_impl<S: Simd>(simd: S, lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // we handle the degenerate cases
    if lhs.is_empty() {
        return;
    } else if rhs.is_empty() {
        visitor.visit_slice(lhs);
        return;
    }

    // Subtract with native-width vectors, then the rest with 128-bit vectors, then the rest with
    // scalar code
    let (i, j) = sub_vectors::<S, U16s<S>>(simd, lhs, rhs, visitor);
    let (lhs, rhs) = (&lhs[i..], &rhs[j..]);
    let (i, j) = if native_lanes::<S>() > BLOCK_LANES {
        sub_vectors::<S, u16x8<S>>(simd, lhs, rhs, visitor)
    } else {
        (0, 0)
    };

    // do the tail using scalar code
    scalar::sub(&lhs[i..], &rhs[j..], visitor);
}

/// Subtracts `rhs` from whole vectors of `lhs`, and returns the number of values of each which
/// were handled
#[inline(always)]
fn sub_vectors<S: Simd, V: U16Vector<S>>(
    simd: S,
    lhs: &[u16],
    rhs: &[u16],
    visitor: &mut impl BinaryOperationVisitor,
) -> (usize, usize) {
    let n = V::LEN;
    let st_a = (lhs.len() / n) * n;
    let st_b = (rhs.len() / n) * n;

    let mut i = 0;
    let mut j = 0;
    if (i < st_a) && (j < st_b) {
        let mut v_a: V = load(simd, &lhs[i..]);
        let mut v_b: V = load(simd, &rhs[j..]);
        // we have a running mask which indicates which values from a have been
        // spotted in b, these don't get written out.
        let mut runningmask_a_found_in_b: u64 = 0;
        loop {
            // a_found_in_b will contain a mask indicate for each entry in A
            // whether it is seen in B
            let may_contain_zero = lhs[i] == 0 || rhs[j] == 0;
            let a_found_in_b: u64 =
                matrix_cmp_bitmask(v_a, v_b, &lhs[i..], &rhs[j..], may_contain_zero);
            runningmask_a_found_in_b |= a_found_in_b;
            // we always compare the last values of A and B
            let a_max: u16 = lhs[i + n - 1];
            let b_max: u16 = rhs[j + n - 1];
            if a_max <= b_max {
                // Ok. In this code path, we are ready to write our v_a
                // because there is no need to read more from B, they will
                // all be large values.
                let bitmask_belongs_to_difference = !runningmask_a_found_in_b & lanes_bitmask(n);
                visitor.visit_vector(v_a, bitmask_belongs_to_difference);
                i += n;
                if i == st_a {
                    break;
                }
                runningmask_a_found_in_b = 0;
                v_a = load(simd, &lhs[i..]);
            }
            if b_max <= a_max {
                // in this code path, the current v_b has become useless
                j += n;
                if j == st_b {
                    break;
                }
                v_b = load(simd, &rhs[j..]);
            }
        }

        debug_assert!(i == st_a || j == st_b);

        // End of main vectorized loop
        // At this point either i_a == st_a, which is the end of the vectorized processing,
        // or i_b == st_b and we are not done processing the vector...
        // so we need to finish it off.
        if i < st_a {
            let remaining_rhs = &rhs[j..];
            if !remaining_rhs.is_empty() {
                // buffer to do a masked load
                let mut buffer: [u16; MAX_LANES] = [0; MAX_LANES];
                let buffer = &mut buffer[..n];
                buffer[..remaining_rhs.len()].copy_from_slice(remaining_rhs);
                // Ensure the buffer is filled with a value we should remove: we do not want to
                // end up trying to remove zero values which aren't actually in rhs
                buffer[remaining_rhs.len()..].fill(remaining_rhs[0]);
                v_b = load(simd, buffer);
                let may_contain_zero = lhs[i] == 0 || remaining_rhs[0] == 0;
                let a_found_in_b: u64 =
                    matrix_cmp_bitmask(v_a, v_b, &lhs[i..], buffer, may_contain_zero);
                runningmask_a_found_in_b |= a_found_in_b;
                // Read from `lhs` (which `v_a` was loaded from): reading a lane of `v_a` makes LLVM
                // split `v_a` into pieces in the loop above
                let max_va = lhs[i + n - 1];
                let used_rhs = remaining_rhs.partition_point(|&b| b <= max_va);
                j += used_rhs;
            }
            let bitmask_belongs_to_difference: u64 = !runningmask_a_found_in_b & lanes_bitmask(n);
            visitor.visit_vector(v_a, bitmask_belongs_to_difference);
            i += n;
        }
    }
    (i, j)
}

/// load the first `V::LEN` values of `src`
///
/// ### Panics
///   - If `src` is shorter than `V::LEN`
#[inline(always)]
fn load<S: Simd, V: U16Vector<S>>(simd: S, src: &[u16]) -> V {
    V::from_slice(simd, &src[..V::LEN])
}

/// `[old[N - 1], new[0], ..., new[N - 2]]`, where `N` is the number of lanes
#[inline(always)]
fn shift_in_1<S: Simd, V: U16Vector<S>>(old: V, new: V) -> V {
    // `slide` takes the shift (`N - 1`) as a const generic, so match on the
    // number of lanes, which is known at compile time
    match V::LEN {
        8 => old.slide::<7>(new),
        16 => old.slide::<15>(new),
        32 => old.slide::<31>(new),
        _ => unreachable!(),
    }
}

/// `[old[N - 2], old[N - 1], new[0], ..., new[N - 3]]`, where `N` is the number of lanes
#[inline(always)]
fn shift_in_2<S: Simd, V: U16Vector<S>>(old: V, new: V) -> V {
    match V::LEN {
        8 => old.slide::<6>(new),
        16 => old.slide::<14>(new),
        32 => old.slide::<30>(new),
        _ => unreachable!(),
    }
}

/// Compare all lanes in `a` to all lanes in `b`
///
/// Returns result mask will be set if any lane at `a[i]` is in any lane of `b`
///
/// ### Example
/// ```ignore
/// let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 32, 33, 34, 35]);
/// let b = u16x8::from_slice(simd, &[2, 4, 6, 8, 10, 12, 14, 16]);
/// let result = matrix_cmp_u16(a, b);
/// assert_eq!(result.to_bitmask(), 0b0000_1010);
/// ```
#[inline(always)]
fn matrix_cmp_u16<S: Simd, V: U16Vector<S>>(a: V, b: V) -> V::Mask {
    // Compare `a` to every rotation of each 128-bit block of `b`
    #[inline(always)]
    fn cmp_rotations_within_blocks<S: Simd, V: U16Vector<S>>(a: V, b: V) -> V::Mask {
        a.simd_eq(b)
            | a.simd_eq(b.slide_within_blocks::<1>(b))
            | a.simd_eq(b.slide_within_blocks::<2>(b))
            | a.simd_eq(b.slide_within_blocks::<3>(b))
            | a.simd_eq(b.slide_within_blocks::<4>(b))
            | a.simd_eq(b.slide_within_blocks::<5>(b))
            | a.simd_eq(b.slide_within_blocks::<6>(b))
            | a.simd_eq(b.slide_within_blocks::<7>(b))
    }

    // ...then rotate the blocks of `b`, so every block of `a` meets every block of `b`
    let mut found = cmp_rotations_within_blocks(a, b);
    let mut b = b;
    for _ in 1..V::LEN / BLOCK_LANES {
        b = b.rotate_elements_left::<BLOCK_LANES>();
        found |= cmp_rotations_within_blocks(a, b);
    }
    found
}

/// Like [`matrix_cmp_u16`], but returns the mask as a bitmask, with bit `i` set if `a[i]` is in `b`
///
/// `a_src` and `b_src` must start with the values `a` and `b` were loaded from.
///
/// `may_contain_zero` must be true if either vector may contain a zero lane: because the vectors
/// come from sorted arrays, that can only be the first lane of the first vector of either array.
///
/// Uses the SSE4.2 PCMPISTRM instruction on each pair of 128-bit blocks where it is available,
/// for vectors of up to 256 bits, falling back to the slower PCMPESTRM when there may be a zero
/// (which PCMPISTRM treats as the end of the string).
#[inline(always)]
fn matrix_cmp_bitmask<S: Simd, V: U16Vector<S>>(
    a: V,
    b: V,
    a_src: &[u16],
    b_src: &[u16],
    may_contain_zero: bool,
) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if V::LEN <= 2 * BLOCK_LANES {
        if let Some(sse4_2) = a.token().level().as_sse4_2() {
            return x86::matrix_cmp_bitmask(sse4_2, a_src, b_src, V::LEN, may_contain_zero);
        }
    }
    let _ = (a_src, b_src, may_contain_zero);
    matrix_cmp_u16(a, b).to_bitmask()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::BLOCK_LANES;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use fearless_simd::{prelude::*, u16x8};

    fearless_simd::kernel!(
        /// Returns a bitmask with bit `i` set if `a[i]` is equal to any of the first `lanes`
        /// values of `b`, for the first `lanes` values of `a`, comparing each pair of 8 lane
        /// blocks
        #[inline]
        pub(super) fn matrix_cmp_bitmask(
            sse4_2: Sse4_2,
            a: &[u16],
            b: &[u16],
            lanes: usize,
            may_contain_zero: bool,
        ) -> u64 {
            const MODE: i32 = _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK;
            let mut found = 0;
            for (i, a) in a[..lanes].as_chunks::<BLOCK_LANES>().0.iter().enumerate() {
                let a: __m128i = u16x8::load_array_ref(sse4_2, a).into();
                let mut block_found = _mm_setzero_si128();
                for b in b[..lanes].as_chunks::<BLOCK_LANES>().0 {
                    let b: __m128i = u16x8::load_array_ref(sse4_2, b).into();
                    let in_b = if may_contain_zero {
                        _mm_cmpestrm::<MODE>(b, 8, a, 8)
                    } else {
                        _mm_cmpistrm::<MODE>(b, a)
                    };
                    block_found = _mm_or_si128(block_found, in_b);
                }
                found |= (_mm_cvtsi128_si32(block_found) as u8 as u64) << (i * BLOCK_LANES);
            }
            found
        }
    );
}

/// Assuming that a and b are sorted, returns an array of the sorted output.
///
/// Uses a bitonic merge network: reversing `b` makes `a` followed by `b` a bitonic sequence, so
/// the lane-wise min and max of `a` and reversed `b` are two bitonic sequences, with every value
/// of the min <= every value of the max. Each is then sorted in `log2(N)` steps.
#[inline(always)]
fn simd_merge_u16<S: Simd, V: U16Vector<S>>(a: V, b: V) -> [V; 2] {
    let b = b.reverse();
    [sort_bitonic(a.min(b)), sort_bitonic(a.max(b))]
}

/// Sorts a bitonic sequence (one which increases then decreases, or vice versa)
#[inline(always)]
fn sort_bitonic<S: Simd, V: U16Vector<S>>(mut v: V) -> V {
    let mut distance = V::LEN / 2;
    while distance > 0 {
        // Compare each lane to the lane `distance` away, in groups of `2 * distance` lanes,
        // keeping the min in the lower lane and the max in the upper lane
        let partner = swap_lanes(v, distance);
        let lower = V::Mask::from_bitmask(v.token(), lower_lanes_bitmask(distance));
        v = lower.select(v.min(partner), v.max(partner));
        distance /= 2;
    }
    v
}

/// A bitmask of the lanes `i` where `i & distance == 0`, for a power of two `distance`
#[inline(always)]
fn lower_lanes_bitmask(distance: usize) -> u64 {
    match distance {
        1 => 0x5555_5555_5555_5555,
        2 => 0x3333_3333_3333_3333,
        4 => 0x0F0F_0F0F_0F0F_0F0F,
        8 => 0x00FF_00FF_00FF_00FF,
        16 => 0x0000_FFFF_0000_FFFF,
        _ => unreachable!(),
    }
}

/// Swaps each lane `i` of `v` with lane `i ^ distance`, for a power of two `distance`
#[inline(always)]
fn swap_lanes<S: Simd, V: U16Vector<S>>(v: V, distance: usize) -> V {
    let simd = v.token();
    if distance < BLOCK_LANES {
        // Within each block: swizzle the two bytes of each lane, with indices relative to the block
        let indices = V::ByteVector::from_fn(simd, |byte| {
            let lane = (byte / 2) % BLOCK_LANES;
            ((lane ^ distance) * 2 + byte % 2) as u8
        });
        v.swizzle_dyn_within_blocks(indices)
    } else if 2 * distance == V::LEN {
        // Swap the two halves
        match distance {
            8 => v.rotate_elements_left::<8>(),
            16 => v.rotate_elements_left::<16>(),
            _ => unreachable!(),
        }
    } else {
        let indices = V::ByteVector::from_fn(simd, |byte| {
            let lane = byte / 2;
            ((lane ^ distance) * 2 + byte % 2) as u8
        });
        v.swizzle_dyn(indices)
    }
}

/// Calls `f` for each 128-bit block of `v` in order, with the values of the block selected by the
/// corresponding byte of `mask` moved to the front, and the number of selected values
#[inline(always)]
pub fn for_each_compressed_block<S: Simd, V: U16Vector<S>>(
    v: V,
    mask: u64,
    mut f: impl FnMut(u16x8<S>, usize),
) {
    let mut values = [0u16; MAX_LANES];
    let values = &mut values[..V::LEN];
    v.store_slice(values);
    for (i, block) in values.as_chunks::<BLOCK_LANES>().0.iter().enumerate() {
        let block_mask = (mask >> (i * BLOCK_LANES)) as u8;
        let block = u16x8::load_array_ref(v.token(), block);
        f(swizzle_to_front(block, block_mask), block_mask.count_ones() as usize);
    }
}

/// Move the values in `val` with the corresponding index in `bitmask`
/// set to the front of the return vector, preserving their order.
///
/// The values in the return vector after index bitmask.count_ones() is unspecified.
// Dynamic swizzles operate on bytes, so the swizzle table moves the `u16` lanes two bytes at a time.
//
// e.g. if `bitmask` is `0b0101`, then swizzle the first two bytes (the first u16 lane) to the
// first two positions, and the 5th and 6th bytes (the third u16 lane) to the next two positions.
#[inline(always)]
pub fn swizzle_to_front<S: Simd>(val: u16x8<S>, bitmask: u8) -> u16x8<S> {
    static SWIZZLE_TABLE: [[u8; 16]; 256] = {
        let mut table = [[0; 16]; 256];
        let mut n = 0usize;
        while n < table.len() {
            let mut x = n;
            let mut i = 0;
            while x > 0 {
                let lsb = x.trailing_zeros() as u8;
                x ^= 1 << lsb;
                table[n][i] = lsb * 2; // first byte
                table[n][i + 1] = lsb * 2 + 1; // second byte
                i += 2;
            }
            n += 1;
        }
        table
    };

    // Our swizzle table retains the order of the bytes in the 16 bit lanes,
    // so it works with either native byte order.
    val.swizzle_dyn(SWIZZLE_TABLE[bitmask as usize])
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bitmap::store::array_store::visitor::VecWriter;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    /// Checks the vectorized op produces the same result as the scalar op
    fn check_op(
        vector: impl Fn(&[u16], &[u16], &mut VecWriter),
        scalar: impl Fn(&[u16], &[u16], &mut VecWriter),
        lhs: &[u16],
        rhs: &[u16],
    ) {
        let mut expected = VecWriter::new(lhs.len() + rhs.len());
        scalar(lhs, rhs, &mut expected);
        let mut actual = VecWriter::new(lhs.len() + rhs.len());
        vector(lhs, rhs, &mut actual);
        assert_eq!(actual.into_inner(), expected.into_inner());
    }

    /// Checks all vectorized ops produce the same results as the scalar ops
    fn check_all(lhs: &[u16], rhs: &[u16]) {
        check_op(or, scalar::or, lhs, rhs);
        check_op(and, scalar::and, lhs, rhs);
        check_op(xor, scalar::xor, lhs, rhs);
        check_op(sub, scalar::sub, lhs, rhs);
    }

    proptest! {
        #[test]
        fn vector_ops_match_scalar_dense(
            lhs in prop::collection::btree_set(0u16..256, 0..200),
            rhs in prop::collection::btree_set(0u16..256, 0..200),
        ) {
            check_all(&Vec::from_iter(lhs), &Vec::from_iter(rhs));
        }

        #[test]
        fn vector_ops_match_scalar_sparse(
            lhs in prop::collection::btree_set(any::<u16>(), 0..200),
            rhs in prop::collection::btree_set(any::<u16>(), 0..200),
        ) {
            check_all(&Vec::from_iter(lhs), &Vec::from_iter(rhs));
        }
    }

    #[test]
    fn swizzle_to_front_keeps_masked_lanes_in_order() {
        dispatch!(level(), simd => {
            let values: [u16; 8] = [10, 11, 12, 13, 14, 15, 16, 17];
            let v = u16x8::from_slice(simd, &values);
            for mask in 0..=255u8 {
                let expected: Vec<u16> =
                    (0..8).filter(|i| mask & (1 << i) != 0).map(|i| values[i]).collect();
                let result = swizzle_to_front(v, mask);
                assert_eq!(&result.as_slice()[..expected.len()], &expected[..]);
            }
        });
    }

    #[test]
    fn vector_ops_match_scalar_at_lane_boundaries() {
        // Lengths around multiples of the vector widths (8, 16 and 32 lanes) exercise the scalar
        // fallback, the vector loops, stepping down to 128-bit vectors, and the partial vector
        // tails; the offsets give disjoint, interleaved, and equal inputs.
        let lens = [0, 1, 7, 8, 9, 15, 16, 17, 24, 25, 31, 32, 33, 48, 63, 64, 65, 96, 97];
        for len_l in lens {
            for len_r in lens {
                for offset in [0, 1, 4, 8, 16, 32, 128] {
                    let lhs: Vec<u16> = (0..len_l).map(|x| 2 * x).collect();
                    let rhs: Vec<u16> = (0..len_r).map(|x| 2 * x + offset).collect();
                    check_all(&lhs, &rhs);
                }
            }
        }
    }

    #[test]
    fn vector_ops_match_scalar_with_max_value() {
        // `u16::MAX` is used as padding, so check inputs where neither, either or both arrays
        // contain it, with lengths around multiples of the vector widths
        let lens = [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];
        for len_l in lens {
            for len_r in lens {
                for (gap_l, gap_r) in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 3)] {
                    let lhs: Vec<u16> =
                        (0..len_l).rev().map(|x| u16::MAX - gap_l - 2 * x).collect();
                    let rhs: Vec<u16> =
                        (0..len_r).rev().map(|x| u16::MAX - gap_r - 2 * x).collect();
                    check_all(&lhs, &rhs);
                }
            }
        }
    }

    fn check_matrix_cmp<S: Simd, V: U16Vector<S>>(
        simd: S,
        a: &[u16],
        b: &[u16],
        may_contain_zero: bool,
    ) {
        let expected = a
            .iter()
            .enumerate()
            .filter(|(_, x)| b.contains(x))
            .fold(0u64, |mask, (i, _)| mask | (1 << i));
        let (va, vb): (V, V) = (load(simd, a), load(simd, b));
        assert_eq!(matrix_cmp_u16(va, vb).to_bitmask(), expected);
        assert_eq!(matrix_cmp_bitmask(va, vb, a, b, may_contain_zero), expected);
    }

    fn check_matrix_cmp_width<S: Simd, V: U16Vector<S>>(simd: S) {
        let n = V::LEN as u16;
        let odd: Vec<u16> = (1..=n).collect();
        let even: Vec<u16> = (1..=n).map(|x| 2 * x).collect();
        check_matrix_cmp::<S, V>(simd, &odd, &even, false);
        check_matrix_cmp::<S, V>(simd, &odd, &even, true);
        let from_zero: Vec<u16> = (0..n).collect();
        let even_from_zero: Vec<u16> = (0..n).map(|x| 2 * x).collect();
        check_matrix_cmp::<S, V>(simd, &from_zero, &even_from_zero, true);
        let reversed: Vec<u16> = (0..n).rev().collect();
        check_matrix_cmp::<S, V>(simd, &even_from_zero, &reversed, true);
    }

    fn check_matrix_cmp_widths<S: Simd>(simd: S) {
        check_matrix_cmp_width::<S, U16s<S>>(simd);
        check_matrix_cmp_width::<S, u16x8<S>>(simd);
    }

    #[test]
    fn matrix_cmp_finds_lanes_of_a_in_b() {
        dispatch!(level(), simd => check_matrix_cmp_widths(simd));
    }
}

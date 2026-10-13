mod scalar;
mod vector;
mod visitor;

use crate::bitmap::store::array_store::visitor::{
    BinaryOperationVisitor, CardinalityCounter, VecWriter,
};
use core::cmp::Ordering;
use core::cmp::Ordering::*;
use core::convert::Infallible;
use core::fmt::{Display, Formatter};
use core::mem::size_of;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitXor, ControlFlow, RangeInclusive, Sub, SubAssign};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use super::bitmap_store::{bit, key, BitmapStore, BITMAP_LENGTH};
use super::interval_store::{for_each_run_overlap, is_sparse_array};
use super::{Interval, IntervalStore};

pub(crate) const ARRAY_ELEMENT_BYTES: usize = 2;

/// Arrays whose lengths differ by more than this factor are combined by searching the longer array
/// for each element of the shorter one, rather than by merging them
const SKEWED_LEN_RATIO: usize = 64;

fn is_skewed(small: &[u16], large: &[u16]) -> bool {
    small.len() * SKEWED_LEN_RATIO < large.len()
}

#[cfg(not(feature = "simd"))]
fn is_skewed_either_way(lhs: &[u16], rhs: &[u16]) -> bool {
    let (small, large) = shorter_first(lhs, rhs);
    is_skewed(small, large)
}

fn shorter_first<'a>(lhs: &'a [u16], rhs: &'a [u16]) -> (&'a [u16], &'a [u16]) {
    if lhs.len() <= rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

fn or(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let (small, large) = shorter_first(lhs, rhs);
    if is_skewed(small, large) {
        return scalar::or_skewed(small, large, visitor);
    }
    #[cfg(feature = "simd")]
    vector::or(lhs, rhs, visitor);
    #[cfg(not(feature = "simd"))]
    scalar::or(lhs, rhs, visitor);
}

fn and(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let (small, large) = shorter_first(lhs, rhs);
    if is_skewed(small, large) {
        return scalar::and_skewed(small, large, visitor);
    }
    #[cfg(feature = "simd")]
    vector::and(lhs, rhs, visitor);
    #[cfg(not(feature = "simd"))]
    scalar::and(lhs, rhs, visitor);
}

fn sub(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    if is_skewed(lhs, rhs) {
        return scalar::sub_skewed_small_lhs(lhs, rhs, visitor);
    }
    if is_skewed(rhs, lhs) {
        return scalar::sub_skewed_small_rhs(lhs, rhs, visitor);
    }
    #[cfg(feature = "simd")]
    vector::sub(lhs, rhs, visitor);
    #[cfg(not(feature = "simd"))]
    scalar::sub(lhs, rhs, visitor);
}

fn xor(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let (small, large) = shorter_first(lhs, rhs);
    if is_skewed(small, large) {
        return scalar::xor_skewed(small, large, visitor);
    }
    #[cfg(feature = "simd")]
    vector::xor(lhs, rhs, visitor);
    #[cfg(not(feature = "simd"))]
    scalar::xor(lhs, rhs, visitor);
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct ArrayStore {
    vec: Vec<u16>,
}

/// Returns the index of the first element of `values` which is not greater than the one before it
fn first_not_increasing(values: &[u16]) -> Option<usize> {
    // Only checking for an early exit once per chunk lets the comparisons within a chunk be
    // vectorized. The scalar loop below finds the exact index.
    const CHUNK_LEN: usize = 16;
    let (lhs, _) = values.as_chunks::<CHUNK_LEN>();
    let (rhs, _) = values.get(1..).unwrap_or_default().as_chunks::<CHUNK_LEN>();
    let mut checked = 0;
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        let mut not_increasing = false;
        for i in 0..CHUNK_LEN {
            not_increasing |= lhs[i] >= rhs[i];
        }
        if not_increasing {
            break;
        }
        checked += CHUNK_LEN;
    }

    (checked + 1..values.len()).find(|&i| values[i - 1] >= values[i])
}

/// Return the first contiguous range of elements in a sorted slice.
pub(crate) fn first_contiguous_range_len(slice: &[u16]) -> usize {
    let [first, rest @ ..] = slice else {
        // Explicitly empty range
        return 0;
    };
    let len = rest.partition_point(|item| {
        let item_ptr = core::ptr::addr_of!(*item);
        // SAFETY: `item` is guaranteed to be in bounds of `slice`.
        let elem_distance = usize::try_from(unsafe { item_ptr.offset_from(first) }).unwrap();
        let value_distance = item.checked_sub(*first).expect("array must be sorted");
        elem_distance == usize::from(value_distance)
    });
    len + 1 // +1 for the first element
}

/// Return the last contiguous range of elements in a sorted slice.
pub(crate) fn last_contiguous_range_len(slice: &[u16]) -> usize {
    let [rest @ .., last] = slice else {
        // Explicitly empty range
        return 0;
    };
    let last_ptr = core::ptr::addr_of!(*last);
    let len_from_start = rest.partition_point(|item| {
        // SAFETY: `item` is guaranteed to be in bounds of `slice`.
        let elem_distance = usize::try_from(unsafe { last_ptr.offset_from(item) }).unwrap();
        let value_distance = last.checked_sub(*item).expect("array must be sorted");
        elem_distance != usize::from(value_distance)
    });
    slice.len() - len_from_start
}

impl ArrayStore {
    pub fn new() -> ArrayStore {
        ArrayStore { vec: vec![] }
    }

    pub fn serialized_byte_size(cardinality: u64) -> usize {
        cardinality as usize * ARRAY_ELEMENT_BYTES
    }

    pub fn byte_size(&self) -> usize {
        Self::serialized_byte_size(self.len())
    }

    #[cfg(feature = "std")]
    pub fn with_capacity(capacity: usize) -> ArrayStore {
        ArrayStore { vec: Vec::with_capacity(capacity) }
    }

    /// The number of total values that can be inserted without needing to reallocate.
    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    ///
    /// Create a new SortedU16Vec from a given vec
    /// It is up to the caller to ensure the vec is sorted and deduplicated
    /// Favor `try_from` / `try_into` for cases in which these invariants should be checked
    ///
    /// # Panics
    ///
    /// When debug_assertions are enabled and the above invariants are not met
    #[inline]
    pub fn from_vec_unchecked(vec: Vec<u16>) -> ArrayStore {
        if cfg!(debug_assertions) {
            vec.try_into().unwrap()
        } else {
            ArrayStore { vec }
        }
    }

    pub fn from_lsb0_bytes(bytes: &[u8], byte_offset: usize, bits_set: u64) -> Self {
        type Word = u64;

        let mut vec = Vec::with_capacity(bits_set as usize);

        let (chunks, remainder) = bytes.as_chunks::<{ size_of::<Word>() }>();
        for (index, chunk) in chunks.iter().enumerate() {
            let bit_index = (byte_offset + index * size_of::<Word>()) * 8;
            let mut word = Word::from_le_bytes(*chunk);

            while word != 0 {
                vec.push((word.trailing_zeros() + bit_index as u32) as u16);
                word &= word - 1;
            }
        }
        for (index, mut byte) in remainder.iter().copied().enumerate() {
            let bit_index = (byte_offset + (bytes.len() - remainder.len()) + index) * 8;
            while byte != 0 {
                vec.push((byte.trailing_zeros() + bit_index as u32) as u16);
                byte &= byte - 1;
            }
        }

        Self::from_vec_unchecked(vec)
    }

    #[inline]
    pub fn insert(&mut self, index: u16) -> bool {
        self.vec.binary_search(&index).map_err(|loc| self.vec.insert(loc, index)).is_err()
    }

    pub fn insert_range(&mut self, range: RangeInclusive<u16>) -> u64 {
        let start = *range.start();
        let end = *range.end();

        // Figure out the starting/ending position in the vec.
        let pos_start = self.vec.binary_search(&start).unwrap_or_else(|x| x);
        let pos_end = pos_start
            + match self.vec[pos_start..].binary_search(&end) {
                Ok(x) => x + 1,
                Err(x) => x,
            };

        // Overwrite the range in the middle - there's no need to take
        // into account any existing elements between start and end, as
        // they're all being added to the set.
        let dropped = self.vec.splice(pos_start..pos_end, start..=end);

        end as u64 - start as u64 + 1 - dropped.len() as u64
    }

    pub fn push(&mut self, index: u16) -> bool {
        if self.max().is_none_or(|max| max < index) {
            self.vec.push(index);
            true
        } else {
            false
        }
    }

    ///
    /// Pushes `index` at the end of the store.
    /// It is up to the caller to have validated index > self.max()
    ///
    /// # Panics
    ///
    /// If debug_assertions enabled and index is > self.max()
    pub(crate) fn push_unchecked(&mut self, index: u16) {
        if cfg!(debug_assertions) {
            if let Some(max) = self.max() {
                assert!(index > max, "store max >= index")
            }
        }
        self.vec.push(index);
    }

    pub fn remove(&mut self, index: u16) -> bool {
        self.vec.binary_search(&index).map(|loc| self.vec.remove(loc)).is_ok()
    }

    pub fn remove_range(&mut self, range: RangeInclusive<u16>) -> u64 {
        let start = *range.start();
        let end = *range.end();

        // Figure out the starting/ending position in the vec.
        let pos_start = self.vec.binary_search(&start).unwrap_or_else(|x| x);
        let pos_end = pos_start
            + match self.vec[pos_start..].binary_search(&end) {
                Ok(x) => x + 1,
                Err(x) => x,
            };
        self.vec.drain(pos_start..pos_end);
        (pos_end - pos_start) as u64
    }

    pub fn remove_smallest(&mut self, n: u64) {
        self.vec.rotate_left(n as usize);
        self.vec.truncate(self.vec.len() - n as usize);
    }

    pub fn remove_biggest(&mut self, n: u64) {
        self.vec.truncate(self.vec.len() - n as usize);
    }

    pub fn contains(&self, index: u16) -> bool {
        self.vec.binary_search(&index).is_ok()
    }

    pub fn contains_range(&self, range: RangeInclusive<u16>) -> bool {
        let start = *range.start();
        let end = *range.end();
        let range_count = usize::from(end - start) + 1;
        if self.vec.len() < range_count {
            return false;
        }
        let start_i = match self.vec.binary_search(&start) {
            Ok(i) => i,
            Err(_) => return false,
        };

        // If there are `range_count` items, last item in the next range_count should be the
        // expected end value, because this vec is sorted and has no duplicates
        self.vec.get(start_i + range_count - 1) == Some(&end)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        let (small, large) = shorter_first(self.as_slice(), other.as_slice());
        if is_skewed(small, large) {
            return scalar::is_disjoint_skewed(small, large);
        }
        let (mut i1, mut i2) = (self.vec.iter(), other.vec.iter());
        let (mut value1, mut value2) = (i1.next(), i2.next());
        loop {
            match value1.and_then(|v1| value2.map(|v2| v1.cmp(v2))) {
                None => return true,
                Some(Equal) => return false,
                Some(Less) => value1 = i1.next(),
                Some(Greater) => value2 = i2.next(),
            }
        }
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        if is_skewed(self.as_slice(), other.as_slice()) {
            return scalar::is_subset_skewed(self.as_slice(), other.as_slice());
        }
        let (mut i1, mut i2) = (self.iter(), other.iter());
        let (mut value1, mut value2) = (i1.next(), i2.next());
        loop {
            match (value1, value2) {
                (None, _) => return true,
                (Some(..), None) => return false,
                (Some(v1), Some(v2)) => match v1.cmp(v2) {
                    Equal => {
                        value1 = i1.next();
                        value2 = i2.next();
                    }
                    Less => return false,
                    Greater => value2 = i2.next(),
                },
            }
        }
    }

    pub fn intersection_len(&self, other: &Self) -> u64 {
        let mut visitor = CardinalityCounter::new();
        and(self.as_slice(), other.as_slice(), &mut visitor);
        visitor.into_inner()
    }

    pub fn intersection_len_interval(&self, interval: &Interval) -> u64 {
        if interval.is_full() {
            return self.len();
        }
        let start_id = self.vec.partition_point(|&f| f < interval.start());
        let end_id = self.vec.partition_point(|&f| f <= interval.end());
        (end_id.saturating_sub(start_id)) as u64
    }

    pub fn to_bitmap_store(&self) -> BitmapStore {
        let mut bits = Box::new([0; BITMAP_LENGTH]);
        let len = self.len();

        for &index in self.iter() {
            bits[key(index)] |= 1 << bit(index);
        }
        BitmapStore::from_unchecked(len, bits)
    }

    pub fn len(&self) -> u64 {
        self.vec.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn min(&self) -> Option<u16> {
        self.vec.first().copied()
    }

    #[inline]
    pub fn max(&self) -> Option<u16> {
        self.vec.last().copied()
    }

    pub fn rank(&self, index: u16) -> u64 {
        match self.vec.binary_search(&index) {
            Ok(i) => i as u64 + 1,
            Err(i) => i as u64,
        }
    }

    pub fn select(&self, n: u16) -> Option<u16> {
        self.vec.get(n as usize).cloned()
    }

    pub fn iter(&'_ self) -> core::slice::Iter<'_, u16> {
        self.vec.iter()
    }

    pub fn into_iter(self) -> alloc::vec::IntoIter<u16> {
        self.vec.into_iter()
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.vec
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain(&mut self, mut f: impl FnMut(u16) -> bool) {
        // Idea to avoid branching from "Engineering Fast Indexes for Big Data
        // Applications" talk by Daniel Lemire
        // (https://youtu.be/1QMgGxiCFWE?t=1242).
        let slice = self.vec.as_mut_slice();
        let mut pos = 0;
        for i in 0..slice.len() {
            let val = slice[i];
            // We want to do `slice[pos] = val` but we don't need the bounds check.
            // SAFETY: pos is always at most i because `f(val) as usize` is at most 1.
            unsafe { *slice.get_unchecked_mut(pos) = val }
            pos += f(val) as usize;
        }
        self.vec.truncate(pos);
    }

    pub(crate) fn internal_validate(&self) -> Result<(), &'static str> {
        if self.vec.is_empty() {
            return Err("zero cardinality in array container");
        }
        if self.vec.len() > super::ARRAY_LIMIT as usize {
            return Err("array cardinality exceeds ARRAY_LIMIT");
        }
        if first_not_increasing(&self.vec).is_some() {
            return Err("array elements not strictly increasing");
        }

        Ok(())
    }
}

impl Default for ArrayStore {
    fn default() -> Self {
        ArrayStore::new()
    }
}

#[derive(Debug)]
pub struct Error {
    index: usize,
    kind: ErrorKind,
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    Duplicate,
    OutOfOrder,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            ErrorKind::Duplicate => {
                write!(f, "Duplicate element found at index: {}", self.index)
            }
            ErrorKind::OutOfOrder => {
                write!(f, "An element was out of order at index: {}", self.index)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl TryFrom<Vec<u16>> for ArrayStore {
    type Error = Error;

    fn try_from(value: Vec<u16>) -> Result<Self, Self::Error> {
        let Some(index) = first_not_increasing(&value) else {
            return Ok(ArrayStore { vec: value });
        };
        let kind = match value[index].cmp(&value[index - 1]) {
            Ordering::Less => ErrorKind::OutOfOrder,
            Ordering::Equal => ErrorKind::Duplicate,
            Ordering::Greater => unreachable!(),
        };
        Err(Error { index, kind })
    }
}

impl BitOr<Self> for &ArrayStore {
    type Output = ArrayStore;

    fn bitor(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        let capacity = self.vec.len() + rhs.vec.len();
        let mut visitor = VecWriter::new(capacity);
        or(self.as_slice(), rhs.as_slice(), &mut visitor);
        ArrayStore::from_vec_unchecked(visitor.into_inner())
    }
}

impl BitAnd<Self> for &ArrayStore {
    type Output = ArrayStore;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut visitor = VecWriter::new(self.vec.len().min(rhs.vec.len()));
        and(self.as_slice(), rhs.as_slice(), &mut visitor);
        ArrayStore::from_vec_unchecked(visitor.into_inner())
    }
}

impl BitAndAssign<&Self> for ArrayStore {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn bitand_assign(&mut self, rhs: &Self) {
        #[cfg(not(feature = "simd"))]
        if !is_skewed_either_way(self.as_slice(), rhs.as_slice()) {
            let mut i = 0;
            self.retain(|x| {
                i += rhs.iter().skip(i).position(|y| *y >= x).unwrap_or(rhs.vec.len());
                rhs.vec.get(i).is_some_and(|y| x == *y)
            });
            return;
        }
        let mut visitor = VecWriter::new(self.vec.len().min(rhs.vec.len()));
        and(self.as_slice(), rhs.as_slice(), &mut visitor);
        self.vec = visitor.into_inner()
    }
}

impl BitAndAssign<&BitmapStore> for ArrayStore {
    fn bitand_assign(&mut self, rhs: &BitmapStore) {
        self.retain(|x| rhs.contains(x));
    }
}

impl BitAndAssign<&IntervalStore> for ArrayStore {
    fn bitand_assign(&mut self, rhs: &IntervalStore) {
        if is_sparse_array(rhs, &self.vec) {
            self.retain(|f| rhs.contains(f));
            return;
        }
        let mut len = 0;
        let ControlFlow::Continue(()) =
            for_each_run_overlap::<_, Infallible>(rhs, &mut self.vec, |vec, overlap| {
                let overlap_len = overlap.len();
                vec.copy_within(overlap, len);
                len += overlap_len;
                ControlFlow::Continue(())
            });
        self.vec.truncate(len);
    }
}

impl SubAssign<&IntervalStore> for ArrayStore {
    fn sub_assign(&mut self, rhs: &IntervalStore) {
        if is_sparse_array(rhs, &self.vec) {
            self.retain(|f| !rhs.contains(f));
            return;
        }
        let mut len = 0;
        let mut kept_from = 0;
        let ControlFlow::Continue(()) =
            for_each_run_overlap::<_, Infallible>(rhs, &mut self.vec, |vec, overlap| {
                vec.copy_within(kept_from..overlap.start, len);
                len += overlap.start - kept_from;
                kept_from = overlap.end;
                ControlFlow::Continue(())
            });
        let vec_len = self.vec.len();
        self.vec.copy_within(kept_from..vec_len, len);
        self.vec.truncate(len + vec_len - kept_from);
    }
}

impl Sub<Self> for &ArrayStore {
    type Output = ArrayStore;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut visitor = VecWriter::new(self.vec.len());
        sub(self.as_slice(), rhs.as_slice(), &mut visitor);
        ArrayStore::from_vec_unchecked(visitor.into_inner())
    }
}

impl SubAssign<&Self> for ArrayStore {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: &Self) {
        #[cfg(not(feature = "simd"))]
        if !is_skewed_either_way(self.as_slice(), rhs.as_slice()) {
            let mut i = 0;
            self.retain(|x| {
                i += rhs.iter().skip(i).position(|y| *y >= x).unwrap_or(rhs.vec.len());
                rhs.vec.get(i).is_none_or(|y| x != *y)
            });
            return;
        }
        let mut visitor = VecWriter::new(self.vec.len());
        sub(self.as_slice(), rhs.as_slice(), &mut visitor);
        self.vec = visitor.into_inner()
    }
}

impl SubAssign<&BitmapStore> for ArrayStore {
    fn sub_assign(&mut self, rhs: &BitmapStore) {
        self.retain(|x| !rhs.contains(x));
    }
}

impl BitXor<Self> for &ArrayStore {
    type Output = ArrayStore;

    fn bitxor(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        let capacity = self.vec.len() + rhs.vec.len();
        let mut visitor = VecWriter::new(capacity);
        xor(self.as_slice(), rhs.as_slice(), &mut visitor);
        ArrayStore::from_vec_unchecked(visitor.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::store::Store;

    fn into_vec(s: Store) -> Vec<u16> {
        match s {
            Store::Array(vec) => vec.vec,
            Store::Bitmap(bits) => bits.to_array_store().vec,
            Store::Run(runs) => runs.iter().collect(),
        }
    }

    fn into_bitmap_store(s: Store) -> Store {
        match s {
            Store::Array(vec) => Store::Bitmap(vec.to_bitmap_store()),
            Store::Bitmap(..) => s,
            Store::Run(runs) => Store::Bitmap(runs.to_bitmap()),
        }
    }

    #[test]
    fn test_first_not_increasing() {
        for len in 0..40 {
            let sorted: Vec<u16> = (0..len).map(|x| x * 3).collect();
            assert_eq!(first_not_increasing(&sorted), None, "len {len}");
            for i in 1..len as usize {
                let mut duplicate = sorted.clone();
                duplicate[i] = duplicate[i - 1];
                assert_eq!(first_not_increasing(&duplicate), Some(i), "len {len}, dup at {i}");
                let mut out_of_order = sorted.clone();
                out_of_order[i] = 0;
                assert_eq!(first_not_increasing(&out_of_order), Some(i), "len {len}, at {i}");
            }
        }
    }

    #[test]
    fn test_array_try_from_reports_first_error() {
        let err = ArrayStore::try_from(vec![1, 2, 3, 3, 2]).unwrap_err();
        assert_eq!((err.index, &err.kind), (3, &ErrorKind::Duplicate));
        let err = ArrayStore::try_from(vec![1, 2, 3, 2, 2]).unwrap_err();
        assert_eq!((err.index, &err.kind), (3, &ErrorKind::OutOfOrder));
        assert!(ArrayStore::try_from(vec![]).is_ok());
        assert!(ArrayStore::try_from(vec![u16::MAX]).is_ok());
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_array_insert_invalid_range() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));

        // Insert a range with start > end.
        let new = store.insert_range(6..=1);
        assert_eq!(new, 0);

        assert_eq!(into_vec(store), vec![1, 2, 8, 9]);
    }

    #[test]
    fn test_array_insert_range() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));

        let new = store.insert_range(4..=5);
        assert_eq!(new, 2);

        assert_eq!(into_vec(store), vec![1, 2, 4, 5, 8, 9]);
    }

    #[test]
    fn test_array_insert_range_left_overlap() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));

        let new = store.insert_range(2..=5);
        assert_eq!(new, 3);

        assert_eq!(into_vec(store), vec![1, 2, 3, 4, 5, 8, 9]);
    }

    #[test]
    fn test_array_insert_range_right_overlap() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));

        let new = store.insert_range(4..=8);
        assert_eq!(new, 4);

        assert_eq!(into_vec(store), vec![1, 2, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_array_contains_range() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![]));
        assert!(!store.contains_range(0..=0));
        assert!(!store.contains_range(0..=1));
        assert!(!store.contains_range(1..=u16::MAX));

        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![0, 1, 2, 3, 4, 5, 100]));
        assert!(store.contains_range(0..=0));
        assert!(store.contains_range(0..=5));
        assert!(!store.contains_range(0..=6));
        assert!(store.contains_range(100..=100));
    }

    #[test]
    fn test_array_insert_range_full_overlap() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));

        let new = store.insert_range(1..=9);
        assert_eq!(new, 5);

        assert_eq!(into_vec(store), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_bitmap_insert_invalid_range() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 8, 9]));
        let mut store = into_bitmap_store(store);

        // Insert a range with start > end.
        let new = store.insert_range(6..=1);
        assert_eq!(new, 0);

        assert_eq!(into_vec(store), vec![1, 2, 8, 9]);
    }

    #[test]
    fn test_bitmap_insert_same_key_overlap() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 3, 62, 63]));
        let mut store = into_bitmap_store(store);

        let new = store.insert_range(1..=62);
        assert_eq!(new, 58);

        assert_eq!(into_vec(store), (1..64).collect::<Vec<_>>());
    }

    #[test]
    fn test_bitmap_insert_range() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130]));
        let mut store = into_bitmap_store(store);

        let new = store.insert_range(4..=128);
        assert_eq!(new, 125);

        let mut want = vec![1, 2];
        want.extend(4..129);
        want.extend([130]);

        assert_eq!(into_vec(store), want);
    }

    #[test]
    fn test_bitmap_insert_range_left_overlap() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130]));
        let mut store = into_bitmap_store(store);

        let new = store.insert_range(1..=128);
        assert_eq!(new, 126);

        let mut want = Vec::new();
        want.extend(1..129);
        want.extend([130]);

        assert_eq!(into_vec(store), want);
    }

    #[test]
    fn test_bitmap_insert_range_right_overlap() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130]));
        let mut store = into_bitmap_store(store);

        let new = store.insert_range(4..=132);
        assert_eq!(new, 128);

        let mut want = vec![1, 2];
        want.extend(4..133);

        assert_eq!(into_vec(store), want);
    }

    #[test]
    fn test_bitmap_insert_range_full_overlap() {
        let store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130]));
        let mut store = into_bitmap_store(store);

        let new = store.insert_range(1..=134);
        assert_eq!(new, 131);

        let mut want = Vec::new();
        want.extend(1..135);

        assert_eq!(into_vec(store), want);
    }

    #[test]
    fn test_bitmap_remove_smallest() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130, 500]));
        store.remove_smallest(3);
        assert_eq!(into_vec(store), vec![500]);
    }

    #[test]
    fn test_bitmap_remove_biggest() {
        let mut store = Store::Array(ArrayStore::from_vec_unchecked(vec![1, 2, 130, 500]));
        store.remove_biggest(2);
        assert_eq!(into_vec(store), vec![1, 2]);
    }
}

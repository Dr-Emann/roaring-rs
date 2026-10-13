//! Scalar arithmetic binary set operations on `ArrayStore`'s inner types

use crate::bitmap::store::array_store::visitor::BinaryOperationVisitor;
use core::cmp::Ordering::*;

#[inline]
pub fn or(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // Traverse both arrays
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        let a = unsafe { lhs.get_unchecked(i) };
        let b = unsafe { rhs.get_unchecked(j) };
        match a.cmp(b) {
            Less => {
                visitor.visit_scalar(*a);
                i += 1;
            }
            Greater => {
                visitor.visit_scalar(*b);
                j += 1;
            }
            Equal => {
                visitor.visit_scalar(*a);
                i += 1;
                j += 1;
            }
        }
    }

    // Store remaining elements of the arrays
    visitor.visit_slice(&lhs[i..]);
    visitor.visit_slice(&rhs[j..]);
}

#[inline]
pub fn and(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // Traverse both arrays
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        let a = unsafe { lhs.get_unchecked(i) };
        let b = unsafe { rhs.get_unchecked(j) };
        match a.cmp(b) {
            Less => i += 1,
            Greater => j += 1,
            Equal => {
                visitor.visit_scalar(*a);
                i += 1;
                j += 1;
            }
        }
    }
}

#[inline]
pub fn sub(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // Traverse both arrays
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        let a = unsafe { lhs.get_unchecked(i) };
        let b = unsafe { rhs.get_unchecked(j) };
        match a.cmp(b) {
            Less => {
                visitor.visit_scalar(*a);
                i += 1;
            }
            Greater => j += 1,
            Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    // Store remaining elements of the left array
    visitor.visit_slice(&lhs[i..]);
}

#[inline]
pub fn xor(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    // Traverse both arrays
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        let a = unsafe { lhs.get_unchecked(i) };
        let b = unsafe { rhs.get_unchecked(j) };
        match a.cmp(b) {
            Less => {
                visitor.visit_scalar(*a);
                i += 1;
            }
            Greater => {
                visitor.visit_scalar(*b);
                j += 1;
            }
            Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    // Store remaining elements of the arrays
    visitor.visit_slice(&lhs[i..]);
    visitor.visit_slice(&rhs[j..]);
}

/// Returns the index of the first element of `slice` which is not less than `value`
///
/// Probes at exponentially increasing offsets before binary searching the last gap, so this is
/// fast when that element is near the start of `slice`.
#[inline]
pub fn advance_until(slice: &[u16], value: u16) -> usize {
    if slice.first().is_none_or(|&first| first >= value) {
        return 0;
    }
    // invariant: `slice[lo] < value`
    let mut lo = 0;
    let mut step = 1;
    loop {
        let hi = lo + step;
        if hi >= slice.len() || slice[hi] >= value {
            let hi = hi.min(slice.len());
            return lo + 1 + slice[lo + 1..hi].partition_point(|&x| x < value);
        }
        lo = hi;
        step *= 2;
    }
}

/// Like [`and`], for when `small` is much shorter than `large`
#[inline]
pub fn and_skewed(small: &[u16], large: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let mut large = large;
    for &value in small {
        large = &large[advance_until(large, value)..];
        let [first, rest @ ..] = large else { return };
        if *first == value {
            visitor.visit_scalar(value);
            large = rest;
        }
    }
}

/// Like [`sub`], for when `lhs` is much shorter than `rhs`
#[inline]
pub fn sub_skewed_small_lhs(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let mut rhs = rhs;
    for (i, &value) in lhs.iter().enumerate() {
        rhs = &rhs[advance_until(rhs, value)..];
        match rhs {
            [] => {
                visitor.visit_slice(&lhs[i..]);
                return;
            }
            [first, rest @ ..] if *first == value => rhs = rest,
            [_, ..] => visitor.visit_scalar(value),
        }
    }
}

/// Like [`sub`], for when `rhs` is much shorter than `lhs`
#[inline]
pub fn sub_skewed_small_rhs(lhs: &[u16], rhs: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let mut lhs = lhs;
    for &value in rhs {
        let (before, after) = lhs.split_at(advance_until(lhs, value));
        visitor.visit_slice(before);
        lhs = after;
        if let [first, rest @ ..] = lhs {
            if *first == value {
                lhs = rest;
            }
        }
    }
    visitor.visit_slice(lhs);
}

/// Like [`or`], for when `small` is much shorter than `large`
#[inline]
pub fn or_skewed(small: &[u16], large: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let mut large = large;
    for &value in small {
        let (before, after) = large.split_at(advance_until(large, value));
        visitor.visit_slice(before);
        large = after;
        if let [first, rest @ ..] = large {
            if *first == value {
                large = rest;
            }
        }
        visitor.visit_scalar(value);
    }
    visitor.visit_slice(large);
}

/// Like [`xor`], for when `small` is much shorter than `large`
#[inline]
pub fn xor_skewed(small: &[u16], large: &[u16], visitor: &mut impl BinaryOperationVisitor) {
    let mut large = large;
    for &value in small {
        let (before, after) = large.split_at(advance_until(large, value));
        visitor.visit_slice(before);
        large = after;
        match large {
            [first, rest @ ..] if *first == value => large = rest,
            [..] => visitor.visit_scalar(value),
        }
    }
    visitor.visit_slice(large);
}

/// Returns true if `small` and `large` have no elements in common, for when `small` is much
/// shorter than `large`
#[inline]
pub fn is_disjoint_skewed(small: &[u16], large: &[u16]) -> bool {
    let mut large = large;
    for &value in small {
        large = &large[advance_until(large, value)..];
        let Some(&first) = large.first() else { return true };
        if first == value {
            return false;
        }
    }
    true
}

/// Returns true if every element of `small` is in `large`, for when `small` is much shorter than
/// `large`
#[inline]
pub fn is_subset_skewed(small: &[u16], large: &[u16]) -> bool {
    let mut large = large;
    for &value in small {
        large = &large[advance_until(large, value)..];
        let [first, rest @ ..] = large else { return false };
        if *first != value {
            return false;
        }
        large = rest;
    }
    true
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bitmap::store::array_store::visitor::VecWriter;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn run(op: impl Fn(&[u16], &[u16], &mut VecWriter), lhs: &[u16], rhs: &[u16]) -> Vec<u16> {
        let mut visitor = VecWriter::new(lhs.len() + rhs.len());
        op(lhs, rhs, &mut visitor);
        visitor.into_inner()
    }

    fn check_skewed(small: &[u16], large: &[u16]) {
        assert_eq!(run(and_skewed, small, large), run(and, small, large));
        assert_eq!(run(or_skewed, small, large), run(or, small, large));
        assert_eq!(run(xor_skewed, small, large), run(xor, small, large));
        assert_eq!(run(sub_skewed_small_lhs, small, large), run(sub, small, large));
        assert_eq!(run(sub_skewed_small_rhs, large, small), run(sub, large, small));
        let intersection = run(and, small, large);
        assert_eq!(is_disjoint_skewed(small, large), intersection.is_empty());
        assert_eq!(is_subset_skewed(small, large), intersection.len() == small.len());
    }

    proptest! {
        #[test]
        fn advance_until_matches_partition_point(
            slice in prop::collection::btree_set(any::<u16>(), 0..300),
            value in any::<u16>(),
        ) {
            let slice = Vec::from_iter(slice);
            prop_assert_eq!(advance_until(&slice, value), slice.partition_point(|&x| x < value));
        }

        #[test]
        fn skewed_ops_match_merges_dense(
            small in prop::collection::btree_set(0u16..2048, 0..16),
            large in prop::collection::btree_set(0u16..2048, 0..2048),
        ) {
            check_skewed(&Vec::from_iter(small), &Vec::from_iter(large));
        }

        #[test]
        fn skewed_ops_match_merges_sparse(
            small in prop::collection::btree_set(any::<u16>(), 0..16),
            large in prop::collection::btree_set(any::<u16>(), 0..2048),
        ) {
            check_skewed(&Vec::from_iter(small), &Vec::from_iter(large));
        }
    }

    #[test]
    fn skewed_ops_match_merges_at_edges() {
        let large: Vec<u16> = (10..1000).collect();
        for small in [&[][..], &[0], &[10], &[999], &[1000], &[0, 10, 500, 999, 1000, u16::MAX]] {
            check_skewed(small, &large);
        }
        check_skewed(&[1, 2, 3], &[]);
    }
}

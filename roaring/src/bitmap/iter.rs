use alloc::vec;
use core::iter::FusedIterator;
use core::slice;

use super::container::{self, Container};
use crate::{NonSortedIntegers, RoaringBitmap};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// An iterator for `RoaringBitmap`.
pub struct Iter<'a> {
    front: Option<container::Iter<'a>>,
    containers: slice::Iter<'a, Container>,
    back: Option<container::Iter<'a>>,
}

/// An iterator for `RoaringBitmap`.
pub struct IntoIter {
    front: Option<container::Iter<'static>>,
    containers: vec::IntoIter<Container>,
    back: Option<container::Iter<'static>>,
}

#[inline]
fn and_then_or_clear<T, U>(opt: &mut Option<T>, f: impl FnOnce(&mut T) -> Option<U>) -> Option<U> {
    let x = f(opt.as_mut()?);
    if x.is_none() {
        *opt = None;
    }
    x
}

impl Iter<'_> {
    fn new(containers: &[Container]) -> Iter {
        Iter { front: None, containers: containers.iter(), back: None }
    }
}

impl IntoIter {
    fn new(containers: Vec<Container>) -> IntoIter {
        IntoIter { front: None, containers: containers.into_iter(), back: None }
    }
}

fn size_hint_impl(
    front: &Option<container::Iter<'_>>,
    containers: &impl AsRef<[Container]>,
    back: &Option<container::Iter<'_>>,
) -> (usize, Option<usize>) {
    let first_size = front.as_ref().map_or(0, |it| it.len());
    let last_size = back.as_ref().map_or(0, |it| it.len());
    let mut size = first_size + last_size;
    for container in containers.as_ref() {
        match size.checked_add(container.len() as usize) {
            Some(new_size) => size = new_size,
            None => return (usize::MAX, None),
        }
    }
    (size, Some(size))
}

impl Iterator for Iter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        loop {
            if let Some(x) = and_then_or_clear(&mut self.front, Iterator::next) {
                return Some(x);
            }
            self.front = match self.containers.next() {
                Some(inner) => Some(inner.into_iter()),
                None => return and_then_or_clear(&mut self.back, Iterator::next),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        size_hint_impl(&self.front, &self.containers, &self.back)
    }

    #[inline]
    fn fold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        if let Some(iter) = &mut self.front {
            init = iter.fold(init, &mut f);
        }
        init = self.containers.fold(init, |acc, container| {
            let iter = <&Container>::into_iter(container);
            iter.fold(acc, &mut f)
        });
        if let Some(iter) = &mut self.back {
            init = iter.fold(init, &mut f);
        };
        init
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        let mut count = self.front.map_or(0, Iterator::count);
        count += self.containers.map(|container| container.len() as usize).sum::<usize>();
        count += self.back.map_or(0, Iterator::count);
        count
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut n = n;
        let nth_advance = |it: &mut container::Iter| {
            let len = it.len();
            if n < len {
                it.nth(n)
            } else {
                n -= len;
                None
            }
        };
        if let Some(x) = and_then_or_clear(&mut self.front, nth_advance) {
            return Some(x);
        }
        for container in self.containers.by_ref() {
            let len = container.len() as usize;
            if n < len {
                let mut front_iter = container.into_iter();
                let result = front_iter.nth(n);
                self.front = Some(front_iter);
                return result;
            }
            n -= len;
        }
        and_then_or_clear(&mut self.back, |it| it.nth(n))
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(x) = and_then_or_clear(&mut self.back, DoubleEndedIterator::next_back) {
                return Some(x);
            }
            self.back = match self.containers.next_back() {
                Some(inner) => Some(inner.into_iter()),
                None => return and_then_or_clear(&mut self.front, DoubleEndedIterator::next_back),
            }
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(iter) = &mut self.back {
            init = iter.rfold(init, &mut fold);
        }
        init = self.containers.rfold(init, |acc, container| {
            let iter = container.into_iter();
            iter.rfold(acc, &mut fold)
        });
        if let Some(iter) = &mut self.front {
            init = iter.rfold(init, &mut fold);
        };
        init
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let mut n = n;
        let nth_advance = |it: &mut container::Iter| {
            let len = it.len();
            if n < len {
                it.nth_back(n)
            } else {
                n -= len;
                None
            }
        };
        if let Some(x) = and_then_or_clear(&mut self.back, nth_advance) {
            return Some(x);
        }
        for container in self.containers.by_ref().rev() {
            let len = container.len() as usize;
            if n < len {
                let mut front_iter = container.into_iter();
                let result = front_iter.nth_back(n);
                self.back = Some(front_iter);
                return result;
            }
            n -= len;
        }
        and_then_or_clear(&mut self.front, |it| it.nth_back(n))
    }
}

#[cfg(target_pointer_width = "64")]
impl ExactSizeIterator for Iter<'_> {}
impl FusedIterator for Iter<'_> {}

impl Iterator for IntoIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        loop {
            if let Some(x) = and_then_or_clear(&mut self.front, Iterator::next) {
                return Some(x);
            }
            match self.containers.next() {
                Some(inner) => self.front = Some(inner.into_iter()),
                None => return and_then_or_clear(&mut self.back, Iterator::next),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        size_hint_impl(&self.front, &self.containers, &self.back)
    }

    #[inline]
    fn fold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        if let Some(iter) = &mut self.front {
            init = iter.fold(init, &mut f);
        }
        init = self.containers.fold(init, |acc, container| {
            let iter = <Container>::into_iter(container);
            iter.fold(acc, &mut f)
        });
        if let Some(iter) = &mut self.back {
            init = iter.fold(init, &mut f);
        };
        init
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        let mut count = self.front.map_or(0, Iterator::count);
        count += self.containers.map(|container| container.len() as usize).sum::<usize>();
        count += self.back.map_or(0, Iterator::count);
        count
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut n = n;
        let nth_advance = |it: &mut container::Iter| {
            let len = it.len();
            if n < len {
                it.nth(n)
            } else {
                n -= len;
                None
            }
        };
        if let Some(x) = and_then_or_clear(&mut self.front, nth_advance) {
            return Some(x);
        }
        for container in self.containers.by_ref() {
            let len = container.len() as usize;
            if n < len {
                let mut front_iter = container.into_iter();
                let result = front_iter.nth(n);
                self.front = Some(front_iter);
                return result;
            }
            n -= len;
        }
        and_then_or_clear(&mut self.back, |it| it.nth(n))
    }
}

impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(x) = and_then_or_clear(&mut self.back, DoubleEndedIterator::next_back) {
                return Some(x);
            }
            match self.containers.next_back() {
                Some(inner) => self.back = Some(inner.into_iter()),
                None => return and_then_or_clear(&mut self.front, DoubleEndedIterator::next_back),
            }
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Some(iter) = &mut self.back {
            init = iter.rfold(init, &mut fold);
        }
        init = self.containers.rfold(init, |acc, container| {
            let iter = container.into_iter();
            iter.rfold(acc, &mut fold)
        });
        if let Some(iter) = &mut self.front {
            init = iter.rfold(init, &mut fold);
        };
        init
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let mut n = n;
        let nth_advance = |it: &mut container::Iter| {
            let len = it.len();
            if n < len {
                it.nth_back(n)
            } else {
                n -= len;
                None
            }
        };
        if let Some(x) = and_then_or_clear(&mut self.back, nth_advance) {
            return Some(x);
        }
        for container in self.containers.by_ref().rev() {
            let len = container.len() as usize;
            if n < len {
                let mut front_iter = container.into_iter();
                let result = front_iter.nth_back(n);
                self.back = Some(front_iter);
                return result;
            }
            n -= len;
        }
        and_then_or_clear(&mut self.front, |it| it.nth_back(n))
    }
}

#[cfg(target_pointer_width = "64")]
impl ExactSizeIterator for IntoIter {}
impl FusedIterator for IntoIter {}

impl RoaringBitmap {
    /// Iterator over each value stored in the RoaringBitmap, guarantees values are ordered by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    /// use core::iter::FromIterator;
    ///
    /// let bitmap = (1..3).collect::<RoaringBitmap>();
    /// let mut iter = bitmap.iter();
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter {
        Iter::new(&self.containers)
    }
}

impl<'a> IntoIterator for &'a RoaringBitmap {
    type Item = u32;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl IntoIterator for RoaringBitmap {
    type Item = u32;
    type IntoIter = IntoIter;

    fn into_iter(self) -> IntoIter {
        IntoIter::new(self.containers)
    }
}

impl<const N: usize> From<[u32; N]> for RoaringBitmap {
    fn from(arr: [u32; N]) -> Self {
        RoaringBitmap::from_iter(arr)
    }
}

impl FromIterator<u32> for RoaringBitmap {
    fn from_iter<I: IntoIterator<Item = u32>>(iterator: I) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        rb.extend(iterator);
        rb
    }
}

impl<'a> FromIterator<&'a u32> for RoaringBitmap {
    fn from_iter<I: IntoIterator<Item = &'a u32>>(iterator: I) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        rb.extend(iterator);
        rb
    }
}

impl Extend<u32> for RoaringBitmap {
    fn extend<I: IntoIterator<Item = u32>>(&mut self, iterator: I) {
        for value in iterator {
            self.insert(value);
        }
    }
}

impl<'a> Extend<&'a u32> for RoaringBitmap {
    fn extend<I: IntoIterator<Item = &'a u32>>(&mut self, iterator: I) {
        for value in iterator {
            self.insert(*value);
        }
    }
}

impl RoaringBitmap {
    /// Create the set from a sorted iterator. Values must be sorted and deduplicated.
    ///
    /// The values of the iterator must be ordered and strictly greater than the greatest value
    /// in the set. If a value in the iterator doesn't satisfy this requirement, it is not added
    /// and the append operation is stopped.
    ///
    /// Returns `Ok` with the requested `RoaringBitmap`, `Err` with the number of elements
    /// that were correctly appended before failure.
    ///
    /// # Example: Create a set from an ordered list of integers.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb = RoaringBitmap::from_sorted_iter(0..10).unwrap();
    ///
    /// assert!(rb.iter().eq(0..10));
    /// ```
    ///
    /// # Example: Try to create a set from a non-ordered list of integers.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let integers = 0..10u32;
    /// let error = RoaringBitmap::from_sorted_iter(integers.rev()).unwrap_err();
    ///
    /// assert_eq!(error.valid_until(), 1);
    /// ```
    pub fn from_sorted_iter<I: IntoIterator<Item = u32>>(
        iterator: I,
    ) -> Result<RoaringBitmap, NonSortedIntegers> {
        let mut rb = RoaringBitmap::new();
        rb.append(iterator).map(|_| rb)
    }

    /// Extend the set with a sorted iterator.
    ///
    /// The values of the iterator must be ordered and strictly greater than the greatest value
    /// in the set. If a value in the iterator doesn't satisfy this requirement, it is not added
    /// and the append operation is stopped.
    ///
    /// Returns `Ok` with the number of elements appended to the set, `Err` with
    /// the number of elements we effectively appended before an error occurred.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb = RoaringBitmap::new();
    /// assert_eq!(rb.append(0..10), Ok(10));
    ///
    /// assert!(rb.iter().eq(0..10));
    /// ```
    pub fn append<I: IntoIterator<Item = u32>>(
        &mut self,
        iterator: I,
    ) -> Result<u64, NonSortedIntegers> {
        // Name shadowed to prevent accidentally referencing the param
        let mut iterator = iterator.into_iter();

        let mut prev = match (iterator.next(), self.max()) {
            (None, _) => return Ok(0),
            (Some(first), Some(max)) if first <= max => {
                return Err(NonSortedIntegers { valid_until: 0 })
            }
            (Some(first), _) => first,
        };

        // It is now guaranteed that so long as the values of the iterator are
        // monotonically increasing they must also be the greatest in the set.

        self.push_unchecked(prev);

        let mut count = 1;

        for value in iterator {
            if value <= prev {
                return Err(NonSortedIntegers { valid_until: count });
            } else {
                self.push_unchecked(value);
                prev = value;
                count += 1;
            }
        }

        Ok(count)
    }
}

//! Selection of the SIMD level to compile hot loops for

#[cfg(feature = "simd")]
use fearless_simd::{dispatch, Level, Simd};

/// The SIMD level to use.
///
/// With `std`, this is detected at runtime on x86/x86-64 (and cached by `fearless_simd`).
/// Otherwise, it is determined by the target features enabled at compile time.
#[cfg(feature = "simd")]
#[inline]
pub(crate) fn level() -> Level {
    #[cfg(feature = "std")]
    return Level::new();
    #[cfg(not(feature = "std"))]
    return Level::baseline();
}

/// Calls `f` compiled with the target features of the SIMD level from [`level`]
///
/// This lets the compiler auto-vectorize `f` with instructions beyond the target's baseline, such
/// as POPCNT and AVX2 on x86-64. `f` must be `#[inline(always)]`, or it is compiled without them.
/// Without the `simd` feature, this just calls `f`.
#[inline(always)]
pub(crate) fn with_simd_features<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "simd")]
    return dispatch!(level(), simd => simd.vectorize(f));
    #[cfg(not(feature = "simd"))]
    return f();
}

//! # Minkowski Distance
//!
//! The Minkowski distance  of order _p_ (where _p_ is an integer) is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance.
//! The Manhattan distance between two points \\(x \in ℝ^n \\) and \\( y \in ℝ^n \\) in n-dimensional space is defined as:
//!
//! \\[ d(x, y) = \left(\sum_{i=0}^n \lvert x_i - y_i \rvert^p\right)^{1/p} \\]
//!
//! Example:
//!
//! ```
//! use smartcore::error::SmartCoreResult;
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::minkowski::Minkowski;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! # fn main() -> SmartCoreResult<()> {
//! let l1: f64 = Minkowski::new(1).distance(&x, &y)?;
//! let l2: f64 = Minkowski::new(2).distance(&x, &y)?;
//! # Ok(())
//! # }
//! ```
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::error::Failed;
use crate::error::SmartCoreResult;
use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;

use super::Distance;

/// Defines the Minkowski distance of order `p`
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Minkowski<T: Number> {
    /// order, integer
    pub p: u16,
    _t: PhantomData<T>,
}

impl<T: Number> Minkowski<T> {
    /// instatiate the initial structure
    pub fn new(p: u16) -> Minkowski<T> {
        Minkowski { p, _t: PhantomData }
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Minkowski<T> {
    fn distance(&self, x: &A, y: &A) -> SmartCoreResult<f64> {
        if x.shape() != y.shape() {
            return Err(Failed::input("Input vector sizes are different"));
        }
        if self.p < 1 {
            return Err(Failed::input("p must be at least 1"));
        }

        let p_t = self.p as f64;

        let dist = x
            .iterator(0)
            .zip(y.iterator(0))
            .try_fold(0.0_f64, |sum, (&a, &b)| {
                let delta = (a - b)
                    .to_f64()
                    .ok_or_else(|| Failed::invalid_state("Unable to convert Minkowski delta to f64"))?;
                Ok(sum + delta.abs().powf(p_t))
            })?;

        Ok(dist.powf(1f64 / p_t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SmartCoreResult;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn minkowski_distance() -> SmartCoreResult<()> {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Minkowski::new(1).distance(&a, &b)?;
        let l2: f64 = Minkowski::new(2).distance(&a, &b)?;
        let l3: f64 = Minkowski::new(3).distance(&a, &b)?;

        assert!((l1 - 9.0).abs() < 1e-8);
        assert!((l2 - 5.19615242).abs() < 1e-8);
        assert!((l3 - 4.32674871).abs() < 1e-8);
        Ok(())
    }

    #[test]
    fn minkowski_distance_negative_p() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let result = Minkowski::new(0).distance(&a, &b);
        assert!(result.is_err());
    }
}

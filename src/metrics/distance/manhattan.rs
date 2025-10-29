//! # Manhattan Distance
//!
//! The Manhattan distance between two points \\(x \in ℝ^n \\) and \\( y \in ℝ^n \\) in n-dimensional space is the sum of the distances in each dimension.
//!
//! \\[ d(x, y) = \sum_{i=0}^n \lvert x_i - y_i \rvert \\]
//!
//! Example:
//!
//! ```
//! use smartcore::error::SmartCoreResult;
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::manhattan::Manhattan;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! # fn main() -> SmartCoreResult<()> {
//! let l1: f64 = Manhattan::new().distance(&x, &y)?;
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

/// Manhattan distance
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Manhattan<T: Number> {
    _t: PhantomData<T>,
}

impl<T: Number> Manhattan<T> {
    /// instatiate the initial structure
    pub fn new() -> Manhattan<T> {
        Manhattan { _t: PhantomData }
    }
}

impl<T: Number> Default for Manhattan<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Manhattan<T> {
    fn distance(&self, x: &A, y: &A) -> SmartCoreResult<f64> {
        if x.shape() != y.shape() {
            return Err(Failed::input("Input vector sizes are different"));
        }

        x.iterator(0)
            .zip(y.iterator(0))
            .try_fold(0.0_f64, |sum, (&a, &b)| {
                let diff = (a - b)
                    .to_f64()
                    .ok_or_else(|| Failed::invalid_state("Unable to convert L1 difference to f64"))?;
                Ok(sum + diff.abs())
            })
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
    fn manhattan_distance() -> SmartCoreResult<()> {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Manhattan::new().distance(&a, &b)?;

        assert!((l1 - 9.0).abs() < 1e-8);
        Ok(())
    }
}

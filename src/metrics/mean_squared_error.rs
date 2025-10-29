//! # Mean Squared Error
//!
//! MSE measures the average magnitude of the errors in a set of predictions, without considering their direction.
//!
//! \\[mse(y, \hat{y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} (y_i - \hat{y_i})^2 \\]
//!
//! where \\(\hat{y}\\) are predictions and \\(y\\) are true target values.
//!
//! Example:
//!
//! ```
//! use smartcore::error::SmartCoreResult;
//! use smartcore::metrics::mean_squared_error::MeanSquareError;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! # fn main() -> SmartCoreResult<()> {
//! let mse: f64 = MeanSquareError::new().get_score(&y_true, &y_pred)?;
//! # Ok(())
//! # }
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, SmartCoreResult};
use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::metrics::Metrics;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Mean Squared Error
pub struct MeanSquareError<T> {
    _phantom: PhantomData<T>,
}

impl<T: Number + FloatNumber> Metrics<T> for MeanSquareError<T> {
    /// create a typed object to call MeanSquareError functions
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    fn new_with(_parameter: f64) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    /// Computes mean squared error
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    fn get_score(
        &self,
        y_true: &dyn ArrayView1<T>,
        y_pred: &dyn ArrayView1<T>,
    ) -> SmartCoreResult<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(Failed::input(
                "Mean squared error requires y_true and y_pred to have the same length",
            ));
        }

        let n = y_true.shape();
        if n == 0 {
            return Err(Failed::input(
                "Mean squared error requires at least one observation to evaluate",
            ));
        }
        let mut rss = T::zero();
        for i in 0..n {
            let res = *y_true.get(i) - *y_pred.get(i);
            rss += res * res;
        }

        let rss_f64 = rss.to_f64().ok_or_else(|| {
            Failed::invalid_state(
                "Mean squared error could not convert the residual sum of squares to f64",
            )
        })?;

        Ok(rss_f64 / n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn mean_squared_error() -> Result<(), Failed> {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = MeanSquareError::new().get_score(&y_true, &y_pred)?;
        let score2: f64 = MeanSquareError::new().get_score(&y_true, &y_true)?;

        assert!((score1 - 0.375).abs() < 1e-8);
        assert!((score2 - 0.0).abs() < 1e-8);
        Ok(())
    }
}

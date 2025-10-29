//! # Recall score
//!
//! How many relevant items are selected?
//!
//! \\[recall = \frac{tp}{tp + fn}\\]
//!
//! where tp (true positive) - correct result, fn (false negative) - missing result
//!
//! Example:
//!
//! ```
//! use smartcore::error::SmartCoreResult;
//! use smartcore::metrics::recall::Recall;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![0., 1., 1., 0.];
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//!
//! # fn main() -> SmartCoreResult<()> {
//! let score: f64 = Recall::new().get_score(&y_true, &y_pred)?;
//! # Ok(())
//! # }
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::collections::HashSet;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, SmartCoreResult};
use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::realnum::RealNumber;

use crate::metrics::Metrics;

/// Recall metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Recall<T> {
    _phantom: PhantomData<T>,
}

impl<T: RealNumber> Metrics<T> for Recall<T> {
    /// create a typed object to call Recall functions
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
    /// Calculated recall score
    /// * `y_true` - ground truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    fn get_score(
        &self,
        y_true: &dyn ArrayView1<T>,
        y_pred: &dyn ArrayView1<T>,
    ) -> SmartCoreResult<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(Failed::input(
                "Recall requires y_true and y_pred to have the same length",
            ));
        }

        let len = y_true.shape();
        if len == 0 {
            return Err(Failed::input(
                "Recall requires at least one observation to evaluate",
            ));
        }

        let mut classes = HashSet::new();
        for i in 0..len {
            classes.insert(y_true.get(i).to_f64_bits());
        }
        let classes = classes.len();
        if classes == 0 {
            return Err(Failed::input(
                "Recall requires at least one distinct class label",
            ));
        }

        let mut tp: usize = 0;
        let mut fn_count: usize = 0;
        for i in 0..len {
            let actual = y_true.get(i);
            let predicted = y_pred.get(i);
            if predicted == actual {
                if classes == 2 {
                    if *actual == T::one() {
                        tp += 1;
                    }
                } else {
                    tp += 1;
                }
            } else if classes == 2 {
                if *actual == T::one() {
                    fn_count += 1;
                }
            } else {
                fn_count += 1;
            }
        }

        let denominator = tp + fn_count;
        if denominator == 0 {
            return Err(Failed::input(
                "Recall is undefined because there are no relevant samples",
            ));
        }

        Ok(tp as f64 / denominator as f64)
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
    fn recall() -> Result<(), Failed> {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Recall::new().get_score(&y_true, &y_pred)?;
        let score2: f64 = Recall::new().get_score(&y_pred, &y_pred)?;

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);

        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];

        let score3: f64 = Recall::new().get_score(&y_true, &y_pred)?;
        assert!((score3 - 0.5).abs() < 1e-8);
        Ok(())
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn recall_multiclass() -> Result<(), Failed> {
        let y_true: Vec<f64> = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];

        let score1: f64 = Recall::new().get_score(&y_true, &y_pred)?;
        let score2: f64 = Recall::new().get_score(&y_pred, &y_pred)?;

        assert!((score1 - 0.333333333).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
        Ok(())
    }
}

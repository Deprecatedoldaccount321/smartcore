//! # XGBoost Binary Classifier
//!
//! Provides a thin wrapper around [`XGRegressor`] configured with the binary
//! logistic objective, exposing ergonomic helpers for probability and label
//! prediction in classification settings.

use super::xgb_regressor::{sigmoid, Objective, XGRegressor, XGRegressorParameters};
use crate::{
    api::{PredictorBorrow, SupervisedEstimatorBorrow},
    error::{Failed, FailedError},
    linalg::basic::arrays::Array2,
    numbers::basenum::Number,
};

/// Hyperparameters for [`XGClassifier`].
#[derive(Clone, Debug)]
pub struct XGClassifierParameters {
    inner: XGRegressorParameters,
}

impl Default for XGClassifierParameters {
    fn default() -> Self {
        Self {
            inner: XGRegressorParameters::for_binary_classification(),
        }
    }
}

impl XGClassifierParameters {
    /// Sets the number of boosting rounds or trees to build.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.inner.n_estimators = n_estimators;
        self
    }

    /// Sets the step size shrinkage used to prevent overfitting.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.inner.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum depth of each individual tree.
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.inner.max_depth = max_depth;
        self
    }

    /// Sets the minimum sum of instance weight (hessian) needed in a child node.
    pub fn with_min_child_weight(mut self, min_child_weight: usize) -> Self {
        self.inner.min_child_weight = min_child_weight;
        self
    }

    /// Sets the L2 regularization term on weights (`lambda`).
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.inner.lambda = lambda;
        self
    }

    /// Sets the minimum loss reduction required to make a further partition on a leaf node.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.inner.gamma = gamma;
        self
    }

    /// Sets the initial prediction score for all instances (interpreted as logit).
    pub fn with_base_score(mut self, base_score: f64) -> Self {
        self.inner.base_score = base_score;
        self
    }

    /// Sets the fraction of samples to be used for fitting individual base learners.
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.inner.subsample = subsample;
        self
    }

    /// Sets the seed for the random number generator for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed;
        self
    }

    /// Access the underlying [`XGRegressorParameters`].
    pub fn as_regressor_parameters(&self) -> &XGRegressorParameters {
        &self.inner
    }

    pub(crate) fn into_regressor_parameters(self) -> XGRegressorParameters {
        self.inner
    }
}

fn validate_binary_labels(labels: &[f64]) -> Result<(), Failed> {
    if labels
        .iter()
        .any(|&label| !(0.0..=1.0).contains(&label) || !label.is_finite())
    {
        return Err(Failed::because(
            FailedError::ParametersError,
            "Binary logistic objective requires labels in [0, 1]",
        ));
    }
    Ok(())
}

/// Gradient boosted tree classifier using the binary logistic objective.
pub struct XGClassifier<TX: Number + PartialOrd, X: Array2<TX>> {
    regressor: XGRegressor<TX, f64, X, Vec<f64>>,
}

impl<TX: Number + PartialOrd, X: Array2<TX>> XGClassifier<TX, X> {
    /// Train a classifier on the provided dataset.
    ///
    /// * `data` - _NxM_ matrix with _N_ observations and _M_ features.
    /// * `labels` - Binary targets encoded as `0.0` or `1.0`.
    /// * `parameters` - Hyperparameters controlling the training procedure.
    pub fn fit(
        data: &X,
        labels: &[f64],
        parameters: XGClassifierParameters,
    ) -> Result<Self, Failed> {
        validate_binary_labels(labels)?;
        let params = parameters.into_regressor_parameters();
        if params.objective != Objective::BinaryLogistic {
            return Err(Failed::because(
                FailedError::ParametersError,
                "XGClassifier expects the BinaryLogistic objective.",
            ));
        }

        let labels_vec = labels.to_vec();
        let regressor = XGRegressor::fit(data, &labels_vec, params)?;
        Ok(Self { regressor })
    }

    /// Predict raw logits for the supplied samples.
    pub fn predict_margin(&self, data: &X) -> Result<Vec<f64>, Failed> {
        let logits = self.regressor.predict(data)?;
        logits
            .into_iter()
            .map(|margin| {
                margin
                    .to_f64()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::InvalidStateError,
                            "Failed to convert prediction margin to f64",
                        )
                    })
            })
            .collect()
    }

    /// Predict class probabilities for the supplied samples.
    pub fn predict_proba(&self, data: &X) -> Result<Vec<f64>, Failed> {
        let margins = self.predict_margin(data)?;
        Ok(margins.into_iter().map(sigmoid).collect())
    }

    /// Predict binary class labels (`0` or `1`) for the supplied samples.
    pub fn predict(&self, data: &X) -> Result<Vec<u8>, Failed> {
        let probabilities = self.predict_proba(data)?;
        Ok(probabilities
            .into_iter()
            .map(|p| if p >= 0.5 { 1 } else { 0 })
            .collect())
    }

    /// Expose the underlying regressor (logit booster). Useful for advanced scenarios.
    pub fn into_regressor(self) -> XGRegressor<TX, f64, X, Vec<f64>> {
        self.regressor
    }
}

impl<'a, TX: Number + PartialOrd, X: Array2<TX>>
    SupervisedEstimatorBorrow<'a, X, Vec<f64>, XGClassifierParameters> for XGClassifier<TX, X>
{
    fn new() -> Self {
        Self {
            regressor: XGRegressor::new(),
        }
    }

    fn fit(
        x: &'a X,
        y: &'a Vec<f64>,
        parameters: &'a XGClassifierParameters,
    ) -> Result<Self, Failed> {
        validate_binary_labels(y)?;
        let mut params = parameters.inner.clone();
        params.objective = Objective::BinaryLogistic;
        let regressor = XGRegressor::fit(x, y, params)?;
        Ok(Self { regressor })
    }
}

impl<'a, TX: Number + PartialOrd, X: Array2<TX>> PredictorBorrow<'a, X, f64>
    for XGClassifier<TX, X>
{
    fn predict(&self, x: &'a X) -> Result<Vec<f64>, Failed> {
        self.predict_proba(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn test_classifier_end_to_end() {
        let data = DenseMatrix::from_2d_array(&[
            &[-3.0_f32],
            &[-2.0_f32],
            &[-1.0_f32],
            &[1.0_f32],
            &[2.0_f32],
            &[3.0_f32],
        ])
        .unwrap();
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let params = XGClassifierParameters::default()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(3);

        let classifier = XGClassifier::fit(&data, &labels, params).unwrap();

        let probabilities = classifier.predict_proba(&data).unwrap();
        assert_eq!(probabilities.len(), labels.len());
        for probability in &probabilities {
            assert!((0.0..=1.0).contains(probability));
        }

        let predictions = classifier.predict(&data).unwrap();
        assert_eq!(predictions.len(), labels.len());
        assert!(predictions.iter().any(|&label| label == 1));
    }
}

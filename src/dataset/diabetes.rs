//! # Diabetes Data
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 442 | 10 | No | Regression |
//!
//! [Diabetes Data](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) was collected by Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani for the "Least Angle Regression" paper.
//! Predictive variables have been mean centered and scaled to unit variance.
//! The dataset has following attributes:
//!
//! | Predictor | Data Type | Target? |
//! |-|-|-|
//! | Age | Numerical | No |
//! | Sex | Numerical | No |
//! | Body mass index (BMI) | Numerical | No |
//! | Average blood pressure (BP) | Numerical | No |
//! | Six blood serum measurements (SR1 - SR6) | Numerical | No |
//! | A quantitative measure of disease progression one year after baseline | Numerical | Yes |
//!
//! ## References:
//! * ["Least Angle Regression", Efron B., Hastie T., Johnstone I., Tibshirani R., 2004, Annals of Statistics (with discussion), 407-499](http://statweb.stanford.edu/~tibs/ftp/lars.pdf)
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;
use crate::error::{Failed, SmartCoreResult};

/// Get dataset
pub fn load_dataset() -> SmartCoreResult<Dataset<f32, u32>> {
    let (x, y, num_samples, num_features) =
        deserialize_data(std::include_bytes!("diabetes.xy")).map_err(|why| {
            let msg = format!("Can't deserialize diabetes.xy. {why}");
            Failed::invalid_state(&msg)
        })?;
    let y: Vec<u32> = y.into_iter().map(|value| value as u32).collect();

    Ok(Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: [
            "Age", "Sex", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: vec!["Disease progression".to_string()],
        description: "Diabetes Data: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html"
            .to_string(),
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::error::Failed;

    // TODO: fix serialization
    // #[cfg(not(target_arch = "wasm32"))]
    // #[test]
    // #[ignore]
    // fn refresh_diabetes_dataset() {
    //     // run this test to generate diabetes.xy file.
    //     let dataset = load_dataset();
    //     assert!(serialize_data(&dataset, "diabetes.xy").is_ok());
    // }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn boston_dataset() -> Result<(), Failed> {
        let dataset = load_dataset()?;
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 10);
        assert_eq!(dataset.num_samples, 442);
        Ok(())
    }
}

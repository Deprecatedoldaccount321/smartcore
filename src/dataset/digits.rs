//! # Optical Recognition of Handwritten Digits Dataset
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 1797 | 64 | No | Classification, Clusteing |
//!
//! [Digits dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) contains normalized bitmaps of handwritten digits (0-9) from a preprinted form.
//! This multivariate dataset is frequently used to demonstrate various machine learning algorithms.
//!
//! All input attributes are integers in the range 0..16.
//!
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;
use crate::error::{Failed, SmartCoreResult};

/// Get dataset
pub fn load_dataset() -> SmartCoreResult<Dataset<f32, f32>> {
    let (x, y, num_samples, num_features) =
        deserialize_data(std::include_bytes!("digits.xy")).map_err(|why| {
            let msg = format!("Can't deserialize digits.xy. {why}");
            Failed::invalid_state(&msg)
        })?;

    Ok(Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: ["sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: ["setosa", "versicolor", "virginica"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        description: "Digits dataset: https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits".to_string(),
    })
}

#[cfg(test)]
mod tests {

    #[cfg(not(target_arch = "wasm32"))]
    use super::super::*;
    use super::*;
    use crate::error::Failed;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore]
    fn refresh_digits_dataset() -> Result<(), Failed> {
        // run this test to generate digits.xy file.
        let dataset = load_dataset()?;
        serialize_data(&dataset, "digits.xy")
            .map_err(|err| Failed::invalid_state(&format!("Failed to serialize digits: {err}")))?;
        Ok(())
    }
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn digits_dataset() -> Result<(), Failed> {
        let dataset = load_dataset()?;
        assert_eq!(dataset.data.len(), 1797 * 64);
        assert_eq!(dataset.target.len(), 1797);
        assert_eq!(dataset.num_features, 64);
        assert_eq!(dataset.num_samples, 1797);
        Ok(())
    }
}

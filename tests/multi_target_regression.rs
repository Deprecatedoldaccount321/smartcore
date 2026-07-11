#[path = "support/counting_matrix.rs"]
mod counting_matrix;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};
use smartcore::linear::ridge_regression::{
    RidgeRegression, RidgeRegressionParameters, RidgeRegressionSolverName,
};

use counting_matrix::{
    factorization_counts, reset_factorization_counts, CountingMatrix, FactorizationCounts,
};

type Matrix = DenseMatrix<f64>;
type Target = Vec<f64>;
type LinearModel = LinearRegression<f64, f64, Matrix, Target>;
type RidgeModel = RidgeRegression<f64, f64, Matrix, Target>;
type LassoModel = Lasso<f64, f64, Matrix, Target>;
type CountingLinearModel = LinearRegression<f64, f64, CountingMatrix, Target>;
type CountingRidgeModel = RidgeRegression<f64, f64, CountingMatrix, Target>;

fn fixture() -> (Matrix, Vec<Target>) {
    let rows = [
        &[0.0, 1.0, 2.0][..],
        &[1.0, 0.5, 3.0][..],
        &[2.0, 1.5, 0.0][..],
        &[3.0, 2.0, 1.0][..],
        &[4.0, 3.5, 2.5][..],
        &[5.0, 2.5, 4.0][..],
        &[6.0, 4.0, 3.0][..],
        &[7.0, 5.0, 5.0][..],
    ];
    let matrix = Matrix::from_2d_array(&rows).unwrap();
    let targets = vec![
        vec![3.0, 5.5, 5.0, 8.0, 11.5, 14.0, 15.0, 19.0],
        vec![4.0, 4.0, 7.5, 9.0, 12.5, 12.5, 15.0, 17.0],
        vec![2.0, 0.0, 5.5, 6.5, 7.0, 6.5, 9.5, 9.5],
        vec![8.0, 12.5, 3.0, 7.0, 12.25, 17.5, 13.0, 20.0],
    ];
    (matrix, targets)
}

fn assert_predictions_match_exact(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "prediction bits differ at index {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn linear_regression_multi_target_matches_four_independent_fits_exactly() {
    let (matrix, targets) = fixture();

    for solver in [
        LinearRegressionSolverName::QR,
        LinearRegressionSolverName::SVD,
    ] {
        let parameters = LinearRegressionParameters::default().with_solver(solver);
        let models = LinearModel::fit_multi_target(&matrix, &targets, parameters.clone()).unwrap();

        assert_eq!(models.len(), targets.len());
        for (model, target) in models.iter().zip(&targets) {
            let independent = LinearModel::fit(&matrix, target, parameters.clone()).unwrap();
            assert_predictions_match_exact(
                &model.predict(&matrix).unwrap(),
                &independent.predict(&matrix).unwrap(),
            );
        }
    }
}

#[test]
fn ridge_multi_target_matches_four_independent_fits_exactly() {
    let (matrix, targets) = fixture();

    for solver in [
        RidgeRegressionSolverName::Cholesky,
        RidgeRegressionSolverName::SVD,
    ] {
        for normalize in [false, true] {
            let parameters = RidgeRegressionParameters {
                solver: solver.clone(),
                alpha: 0.25,
                normalize,
            };
            let models =
                RidgeModel::fit_multi_target(&matrix, &targets, parameters.clone()).unwrap();

            assert_eq!(models.len(), targets.len());
            for (model, target) in models.iter().zip(&targets) {
                let independent = RidgeModel::fit(&matrix, target, parameters.clone()).unwrap();
                assert_predictions_match_exact(
                    &model.predict(&matrix).unwrap(),
                    &independent.predict(&matrix).unwrap(),
                );
            }
        }
    }
}

#[test]
fn ols_and_ridge_multi_target_factorize_once_for_four_targets() {
    let (matrix, targets) = fixture();
    assert_eq!(targets.len(), 4);
    let matrix = CountingMatrix::from_dense(matrix);

    for solver in [
        LinearRegressionSolverName::QR,
        LinearRegressionSolverName::SVD,
    ] {
        let expected_counts = match solver {
            LinearRegressionSolverName::QR => FactorizationCounts::QR_ONCE,
            LinearRegressionSolverName::SVD => FactorizationCounts::SVD_ONCE,
        };
        let solver_name = format!("{solver:?}");
        let parameters = LinearRegressionParameters::default().with_solver(solver);

        reset_factorization_counts();
        let models = CountingLinearModel::fit_multi_target(&matrix, &targets, parameters).unwrap();

        assert_eq!(models.len(), targets.len());
        assert_eq!(
            factorization_counts(),
            expected_counts,
            "OLS {solver_name} should factorize once for all four targets"
        );
    }

    for solver in [
        RidgeRegressionSolverName::Cholesky,
        RidgeRegressionSolverName::SVD,
    ] {
        for normalize in [false, true] {
            let expected_counts = match solver {
                RidgeRegressionSolverName::Cholesky => FactorizationCounts::CHOLESKY_ONCE,
                RidgeRegressionSolverName::SVD => FactorizationCounts::SVD_ONCE,
            };
            let solver_name = format!("{solver:?}");
            let parameters = RidgeRegressionParameters {
                solver: solver.clone(),
                alpha: 0.25,
                normalize,
            };

            reset_factorization_counts();
            let models =
                CountingRidgeModel::fit_multi_target(&matrix, &targets, parameters).unwrap();

            assert_eq!(models.len(), targets.len());
            assert_eq!(
                factorization_counts(),
                expected_counts,
                "Ridge {solver_name} with normalize={normalize} should factorize once for all four targets"
            );
        }
    }
}

#[test]
fn lasso_multi_target_matches_four_independent_fits_exactly() {
    let (matrix, targets) = fixture();
    let parameters = LassoParameters {
        alpha: 0.1,
        normalize: true,
        tol: 1e-4,
        max_iter: 100,
    };

    let models = LassoModel::fit_multi_target(&matrix, &targets, parameters.clone()).unwrap();

    assert_eq!(models.len(), targets.len());
    for (model, target) in models.iter().zip(&targets) {
        let independent = LassoModel::fit(&matrix, target, parameters.clone()).unwrap();
        assert_predictions_match_exact(
            &model.predict(&matrix).unwrap(),
            &independent.predict(&matrix).unwrap(),
        );
    }
}

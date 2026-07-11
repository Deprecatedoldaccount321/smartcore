use std::cell::Cell;

use smartcore::error::Failed;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::optimization_control::OPTIMIZATION_INTERRUPTED;

type Matrix = DenseMatrix<f64>;
type Target = Vec<f64>;
type LassoModel = Lasso<f64, f64, Matrix, Target>;

fn fixture() -> (Matrix, Target) {
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
    let target = vec![3.0, 5.5, 5.0, 8.0, 11.5, 14.0, 15.0, 19.0];
    (matrix, target)
}

#[test]
fn lasso_control_returns_a_stable_fit_error() {
    let (matrix, target) = fixture();
    let cancelled = || true;

    let error = LassoModel::fit_with_control(
        &matrix,
        &target,
        LassoParameters::default(),
        &cancelled,
    )
    .unwrap_err();

    assert_eq!(error, Failed::fit(OPTIMIZATION_INTERRUPTED));
}

#[test]
fn lasso_polls_control_during_active_optimization() {
    let (matrix, target) = fixture();
    let polls = Cell::new(0usize);
    let cancel_after_work_starts = || {
        polls.set(polls.get() + 1);
        polls.get() >= 6
    };

    let error = LassoModel::fit_with_control(
        &matrix,
        &target,
        LassoParameters {
            alpha: 0.1,
            normalize: false,
            tol: 1e-12,
            max_iter: 1000,
        },
        &cancel_after_work_starts,
    )
    .unwrap_err();

    assert_eq!(error, Failed::fit(OPTIMIZATION_INTERRUPTED));
    assert_eq!(polls.get(), 6);
}

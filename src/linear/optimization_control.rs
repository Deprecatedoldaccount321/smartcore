//! Cooperative control for iterative optimization.

use crate::error::Failed;

/// Stable error message returned when optimization is interrupted.
pub const OPTIMIZATION_INTERRUPTED: &str = "Optimization interrupted by control";

/// A cooperatively polled cancellation or deadline control.
///
/// Closures implement this trait, so callers can check an atomic cancellation
/// flag, a deadline, or both without tying the optimizer to a runtime.
pub trait OptimizationControl {
    /// Return `true` when the current optimization should stop.
    fn should_stop(&self) -> bool;
}

impl<F> OptimizationControl for F
where
    F: Fn() -> bool,
{
    fn should_stop(&self) -> bool {
        self()
    }
}

pub(crate) fn check_control<C: OptimizationControl + ?Sized>(
    control: &C,
) -> Result<(), Failed> {
    if control.should_stop() {
        Err(Failed::fit(OPTIMIZATION_INTERRUPTED))
    } else {
        Ok(())
    }
}

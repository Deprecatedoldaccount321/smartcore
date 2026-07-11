use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use smartcore::error::Failed;
use smartcore::linalg::basic::arrays::{
    Array, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;

static CHOLESKY_SOLVE_COUNT: AtomicUsize = AtomicUsize::new(0);
static QR_SOLVE_COUNT: AtomicUsize = AtomicUsize::new(0);
static SVD_SOLVE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FactorizationCounts {
    pub cholesky: usize,
    pub qr: usize,
    pub svd: usize,
}

impl FactorizationCounts {
    pub const CHOLESKY_ONCE: Self = Self {
        cholesky: 1,
        qr: 0,
        svd: 0,
    };
    pub const QR_ONCE: Self = Self {
        cholesky: 0,
        qr: 1,
        svd: 0,
    };
    pub const SVD_ONCE: Self = Self {
        cholesky: 0,
        qr: 0,
        svd: 1,
    };
}

pub fn factorization_counts() -> FactorizationCounts {
    FactorizationCounts {
        cholesky: CHOLESKY_SOLVE_COUNT.load(Ordering::SeqCst),
        qr: QR_SOLVE_COUNT.load(Ordering::SeqCst),
        svd: SVD_SOLVE_COUNT.load(Ordering::SeqCst),
    }
}

pub fn reset_factorization_counts() {
    CHOLESKY_SOLVE_COUNT.store(0, Ordering::SeqCst);
    QR_SOLVE_COUNT.store(0, Ordering::SeqCst);
    SVD_SOLVE_COUNT.store(0, Ordering::SeqCst);
}

#[derive(Clone, Debug)]
pub struct CountingMatrix {
    inner: DenseMatrix<f64>,
}

impl CountingMatrix {
    pub fn from_dense(inner: DenseMatrix<f64>) -> Self {
        Self { inner }
    }
}

impl Array<f64, (usize, usize)> for CountingMatrix {
    fn get(&self, position: (usize, usize)) -> &f64 {
        <DenseMatrix<f64> as Array<f64, (usize, usize)>>::get(&self.inner, position)
    }

    fn shape(&self) -> (usize, usize) {
        <DenseMatrix<f64> as Array<f64, (usize, usize)>>::shape(&self.inner)
    }

    fn is_empty(&self) -> bool {
        <DenseMatrix<f64> as Array<f64, (usize, usize)>>::is_empty(&self.inner)
    }

    fn iterator<'a>(&'a self, axis: u8) -> Box<dyn Iterator<Item = &'a f64> + 'a> {
        <DenseMatrix<f64> as Array<f64, (usize, usize)>>::iterator(&self.inner, axis)
    }
}

impl MutArray<f64, (usize, usize)> for CountingMatrix {
    fn set(&mut self, position: (usize, usize), value: f64) {
        <DenseMatrix<f64> as MutArray<f64, (usize, usize)>>::set(&mut self.inner, position, value);
    }

    fn iterator_mut<'a>(&'a mut self, axis: u8) -> Box<dyn Iterator<Item = &'a mut f64> + 'a> {
        <DenseMatrix<f64> as MutArray<f64, (usize, usize)>>::iterator_mut(&mut self.inner, axis)
    }
}

impl ArrayView2<f64> for CountingMatrix {}
impl MutArrayView2<f64> for CountingMatrix {}

impl Array2<f64> for CountingMatrix {
    fn fill(rows: usize, columns: usize, value: f64) -> Self {
        Self::from_dense(<DenseMatrix<f64> as Array2<f64>>::fill(
            rows, columns, value,
        ))
    }

    fn slice<'a>(
        &'a self,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> Box<dyn ArrayView2<f64> + 'a> {
        <DenseMatrix<f64> as Array2<f64>>::slice(&self.inner, rows, columns)
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> Box<dyn MutArrayView2<f64> + 'a> {
        <DenseMatrix<f64> as Array2<f64>>::slice_mut(&mut self.inner, rows, columns)
    }

    fn from_iterator<I: Iterator<Item = f64>>(
        iterator: I,
        rows: usize,
        columns: usize,
        axis: u8,
    ) -> Self {
        Self::from_dense(<DenseMatrix<f64> as Array2<f64>>::from_iterator(
            iterator, rows, columns, axis,
        ))
    }

    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<f64> + 'a> {
        <DenseMatrix<f64> as Array2<f64>>::get_row(&self.inner, row)
    }

    fn get_col<'a>(&'a self, column: usize) -> Box<dyn ArrayView1<f64> + 'a> {
        <DenseMatrix<f64> as Array2<f64>>::get_col(&self.inner, column)
    }

    fn transpose(&self) -> Self {
        Self::from_dense(<DenseMatrix<f64> as Array2<f64>>::transpose(&self.inner))
    }
}

impl CholeskyDecomposable<f64> for CountingMatrix {
    fn cholesky_solve_mut(self, right_hand_side: Self) -> Result<Self, Failed> {
        CHOLESKY_SOLVE_COUNT.fetch_add(1, Ordering::SeqCst);
        <DenseMatrix<f64> as CholeskyDecomposable<f64>>::cholesky_solve_mut(
            self.inner,
            right_hand_side.inner,
        )
        .map(Self::from_dense)
    }
}

impl QRDecomposable<f64> for CountingMatrix {
    fn qr_solve_mut(self, right_hand_side: Self) -> Result<Self, Failed> {
        QR_SOLVE_COUNT.fetch_add(1, Ordering::SeqCst);
        <DenseMatrix<f64> as QRDecomposable<f64>>::qr_solve_mut(self.inner, right_hand_side.inner)
            .map(Self::from_dense)
    }
}

impl SVDDecomposable<f64> for CountingMatrix {
    fn svd_solve_mut(self, right_hand_side: Self) -> Result<Self, Failed> {
        SVD_SOLVE_COUNT.fetch_add(1, Ordering::SeqCst);
        <DenseMatrix<f64> as SVDDecomposable<f64>>::svd_solve_mut(self.inner, right_hand_side.inner)
            .map(Self::from_dense)
    }
}

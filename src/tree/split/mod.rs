pub mod vector_datasets;

use super::loss_fn::{
    split_values::{NullDirection, SplitInfo},
    Score, ScoreError,
};

#[derive(Debug, thiserror::Error)]
pub enum BestSplitNotFound {
    #[error("Split not found: {0}")]
    Score(#[from] ScoreError),
    #[error("Split not found: cannot compare score {} and {}", 0.0, 0.1)]
    ScoreNotComparable((SplitInfo, SplitInfo)),
    #[error("Split not found: split not needed")]
    NoSplitRequired,
}

#[derive(Debug, thiserror::Error)]
pub enum DataSetRowsError {
    #[error("Dataset Row Error: Ill formed dataframe, {0}, {1}")]
    IllFormedColumn(String, usize),
    #[error("Dataset Row Error: DF has no columns")]
    EmptyDF,
}

pub trait Splittable {
    fn len(&self) -> usize;
    fn split(
        &mut self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> Self;
}

pub trait Feature<T: PartialOrd>: Splittable {
    fn find_splits(&self) -> impl Iterator<Item = T> + '_;
    fn mask<'a>(&'a self, split: T) -> impl Iterator<Item = Option<bool>> + 'a;
}

pub trait Target<T>: Splittable + Sync {
    fn iter(&self) -> impl Iterator<Item = T>;
}

pub trait DataSet: Splittable {
    fn find_best_split<T, S: Score<T>>(
        &self,
        target: &impl Target<T>,
        score_function: &S,
    ) -> Result<SplitInfo, BestSplitNotFound>;
    fn num_rows(&self) -> Result<usize, DataSetRowsError>;
    fn rows(
        &self,
    ) -> Result<
        impl Iterator<Item = Result<Vec<(&str, Option<impl Into<f64> + Copy>)>, DataSetRowsError>>,
        DataSetRowsError,
    >;
}

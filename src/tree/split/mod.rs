pub mod vector_datasets;

use super::loss_fn::{
    split_values::{NullDirection, SplitInfo},
    Score, ScoreError,
};

pub enum BestSplitNotFound {
    Score(ScoreError),
    ScoreNotComparable((SplitInfo, SplitInfo)),
    NoSplitRequired,
}

impl std::fmt::Display for BestSplitNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BestSplitNotFound::Score(err) => write!(f, "{}", err),
            BestSplitNotFound::ScoreNotComparable((s1, s2)) => {
                write!(f, "Cannot compare score {} and score {}", s1, s2)
            }
            BestSplitNotFound::NoSplitRequired => write!(f, "No need for further splitting"),
        }
    }
}

impl From<ScoreError> for BestSplitNotFound {
    fn from(value: ScoreError) -> Self {
        BestSplitNotFound::Score(value)
    }
}


pub trait Feature<T: PartialOrd> {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &'_ T;
    fn find_splits(&self) -> impl Iterator<Item = T> + '_;
    fn mask<'a>(&'a self, split: T) -> impl Iterator<Item = Option<bool>> + 'a;
}

pub trait Target<T> {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &'_ T;
    fn iter(&self) -> impl Iterator<Item = T>;
    fn split(
        &mut self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> Self;
}

pub trait DataSet {
    fn find_best_split<T, S: Score<T>>(
        &self,
        target: &impl Target<T>,
        score_function: &S,
    ) -> Result<SplitInfo, BestSplitNotFound>;
    fn num_rows(&self) -> usize;
    fn split(
        &mut self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> Self;
    fn rows(&self) -> impl Iterator<Item = Vec<(&str, impl Into<f64> + Copy)>>;
}

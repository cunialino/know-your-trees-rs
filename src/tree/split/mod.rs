pub mod vector_datasets;

use super::loss_fn::{
    split_values::{NullDirection, SplitInfo},
    Score,
};

pub trait Feature {
    type Item: PartialOrd;
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &'_ Self::Item;
    fn find_splits(&self) -> impl Iterator<Item = Self::Item> + '_;
    fn mask<'a>(&'a self, split: Self::Item) -> impl Iterator<Item = Option<bool>> + 'a;
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
    ) -> Option<SplitInfo>;
    fn num_rows(&self) -> usize;
    fn split(
        &mut self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> Self;
    fn rows(&self) -> impl Iterator<Item = Vec<(&str, impl Into<f64> + Copy)>>;
}

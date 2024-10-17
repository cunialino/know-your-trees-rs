use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;

use crate::tree::loss_fn::split_values::NullDirection;
use crate::tree::loss_fn::split_values::SplitInfo;
use crate::tree::loss_fn::Score;
use crate::tree::split::Feature;
use core::cmp::Ordering;
use std::collections::HashMap;

use super::BestSplitNotFound;
use super::DataSet;
use super::DataSetRowsError;
use super::Splittable;
use super::Target;

impl<T> Splittable for std::vec::Vec<T> 
where
    T: Copy
{
    fn len(&self) -> usize {
        self.len()
    }
    fn split(
        &self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> (Self, Self) {
        let mut right_values = Vec::new();
        let mut left_values = Vec::new();

        for (value, should_go_left) in self.into_iter().zip(mask.map(|m| match m {
            Some(b) => b,
            None => matches!(null_direction, NullDirection::Left),
        })) {
            if should_go_left {
                left_values.push(value.to_owned());
            } else {
                right_values.push(value.to_owned());
            }
        }

        (left_values, right_values)
    }
}

impl<T> Feature<T> for std::vec::Vec<T>
where
    T: Into<f64> + PartialOrd + Copy,
{
    fn mask<'a>(&'a self, split: T) -> impl Iterator<Item = Option<bool>> + 'a + Clone {
        self.iter().map(move |v| match v.partial_cmp(&split) {
            Some(Ordering::Less) => Some(true),
            Some(Ordering::Equal) => Some(false),
            Some(Ordering::Greater) => Some(false),
            _ => None,
        })
    }
    fn find_splits(&self) -> impl Iterator<Item = T> + '_ {
        self.iter().copied()
    }
}
impl<T> Feature<T> for std::vec::Vec<Option<T>>
where
    T: Into<f64> + PartialOrd + Copy,
{
    fn mask<'a>(&'a self, split: T) -> impl Iterator<Item = Option<bool>> + 'a + Clone {
        self.iter().map(move |v| {
            if let Some(v) = v {
                match v.partial_cmp(&split) {
                    Some(Ordering::Less) => Some(true),
                    Some(Ordering::Equal) => Some(false),
                    Some(Ordering::Greater) => Some(false),
                    _ => None,
                }
            } else {
                None
            }
        })
    }
    fn find_splits(&self) -> impl Iterator<Item = T> + '_ {
        self.iter().filter_map(|v| v.to_owned())
    }
}

impl Target<bool> for std::vec::Vec<bool> {
    fn iter(&self) -> impl Iterator<Item = bool> {
        self.as_slice().iter().copied()
    }
}

impl<F> Splittable for HashMap<String, std::vec::Vec<F>>
where
    F: Into<f64> + PartialOrd + Copy,
{
    fn len(&self) -> usize {
        self.values().map(|vec| vec.len()).max().unwrap()
    }
    fn split(
        &self,
        mask: impl Iterator<Item = Option<bool>>,
        null_direction: NullDirection,
    ) -> (Self, Self) {
        let mut left = HashMap::with_capacity(self.len());
        let mut right = HashMap::with_capacity(self.len());

        let mask: Vec<_> = mask.collect();

        for (column_name, values) in self.into_iter() {
            let (left_vals, right_values) = values.split(mask.iter().copied(), null_direction);
            left.insert(column_name.clone(), left_vals);
            right.insert(column_name.clone(), right_values);
        }
        (left, right)
    }
}

impl<F> DataSet for HashMap<String, std::vec::Vec<F>>
where
    F: Into<f64> + PartialOrd + Copy + Send + Copy + Sync,
{
    fn find_best_split<T, S: Score<T>>(
        &self,
        target: &impl Target<T>,
        score_function: &S,
    ) -> Result<(SplitInfo, impl Iterator<Item = Option<bool>> + Clone), BestSplitNotFound> {
        let min_sp = |s1: (SplitInfo, _), s2: (SplitInfo, _)| match s1.0.partial_cmp(&s2.0) {
            Some(Ordering::Less) | Some(Ordering::Equal) => Ok(s1),
            Some(Ordering::Greater) => Ok(s2),
            None => Err(BestSplitNotFound::ScoreNotComparable((s1.0, s2.0))),
        };
        self.par_iter()
            .flat_map(|(name, values)| {
                values.find_splits().par_bridge().map(move |split_val| {
                    let mask = values.mask(split_val);
                    let score = score_function.split_score(target, mask.clone())?;
                    Ok(( SplitInfo::new(name.clone(), split_val.into(), score), mask))
                })
            })
            .reduce(
                || Err(BestSplitNotFound::NoSplitRequired),
                |acc, el| match (acc, el) {
                    (Ok(acc), Ok(el)) => min_sp(acc, el),
                    (Ok(acc), Err(_)) => Ok(acc),
                    (Err(_), Ok(el)) => Ok(el),
                    (Err(acc), Err(_)) => Err(BestSplitNotFound::from(acc)),
                },
            )
    }
    fn num_rows(&self) -> Result<usize, DataSetRowsError> {
        let max = self.values().map(|vec| vec.len()).max();
        match max {
            Some(m) => Ok(m),
            None => Err(DataSetRowsError::EmptyDF),
        }
    }
    fn rows(
        &self,
    ) -> Result<
        impl Iterator<Item = Result<Vec<(&str, Option<impl Into<f64> + Copy>)>, DataSetRowsError>>,
        DataSetRowsError,
    > {
        let indices = 0..self.num_rows()?;
        Ok(indices.into_iter().map(|idx| {
            self.iter()
                .map(|(name, col)| match col.get(idx) {
                    Some(v) => Ok((name.as_str(), Some(v.to_owned()))),
                    None => Err(DataSetRowsError::IllFormedColumn(name.to_owned(), idx)),
                })
                // Should be ok to collect here, we are collecting
                // at most the number of columns of df (usually not very high)
                .collect()
        }))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tree::loss_fn::Logit;
    use crate::tree::loss_fn::ScoringFunction;
    #[test]
    fn test_logit_split() {
        let df = HashMap::from([("f1".to_owned(), vec![1., 2., 3.])]);
        let tar = vec![true, true, false];
        let score_fn = ScoringFunction::Logit(Logit::new(0.5));
        if let Ok(( split_info, _)) = df.find_best_split(&tar, &score_fn) {
            println!(
                "Split col: {}\nSplit val: {}",
                split_info.name, split_info.value
            );
            assert_eq!("f1".to_string(), split_info.name, "Wrong split col");
            assert_eq!(3., split_info.value, "Wrong split point");
        } else {
            panic!("Cannot find split")
        };
    }
    #[test]
    fn test_logit_split_score() {
        let tar = vec![true, true, false];
        let init_prd: f64 = 0.5;
        let score_fn = ScoringFunction::Logit(Logit::new(init_prd));
        let g1 = init_prd - 1.;
        let g2 = init_prd - 1.;
        let g3 = init_prd;
        let h = init_prd.powi(2);

        let curr = (g1 + g2 + g3).powi(2) / (3. * h);
        let filter_1 = vec![Some(true), Some(false), Some(false)];
        let output_1 = (g1).powi(2) / h + (g2 + g3).powi(2) / (init_prd.powi(2) * 2.) - curr;
        let res_1 = score_fn.split_score(&tar, filter_1.into_iter());
        assert_eq!(-output_1, res_1.unwrap().score, "Wrong Score");

        let filter_2 = vec![Some(true), Some(true), Some(false)];
        let output_2 = (g1 + g2).powi(2) / (2. * h) + (g3).powi(2) / h - curr;
        let res_2 = score_fn.split_score(&tar, filter_2.into_iter());
        assert_eq!(-output_2, res_2.unwrap().score, "Wrong Score");
    }
    #[test]
    fn test_null_feat() {
        let feat_split = vec![Some(1.), None, None].find_splits().next().unwrap();
        assert_eq!(1., feat_split, "Wrong splits for null vals")
    }
}

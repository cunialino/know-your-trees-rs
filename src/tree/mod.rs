use loss_fn::{split_values::SplitInfo, Score};
use split::{DataSet, Target};

pub mod loss_fn;
pub mod split;

#[derive(Debug, Default)]
pub struct TreeConfig {
    pub max_depth: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    #[error("Tree Error: {0}")]
    DataSetRowsError(#[from] crate::tree::split::DataSetRowsError),
    #[error("Could not find feature {0} in dataset")]
    CouldNotFindFeature(String),
    #[error("Found leaf with no prediction")]
    NoPredictionInLeaf,
}

#[derive(Debug, PartialEq)]
pub struct Tree {
    pub split_info: Option<SplitInfo>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub prediction: Option<f64>, // Optional: only used at leaf nodes
}

impl Tree {
    pub fn fit<T, S: Score<T>>(
        samples: &impl DataSet,
        target: &impl Target<T>,
        tree_config: &TreeConfig,
        score_fn: &S,
    ) -> Result<Tree, TreeError> {
        let max_depth = tree_config.max_depth.clone();
        Tree::build_tree_recursive(samples, target, max_depth, score_fn, None)
    }
    fn build_leaf<T, S: Score<T>>(target: &impl Target<T>, split_function: &S) -> Tree {
        let pred = split_function.pred(target);
        Tree {
            split_info: None,
            left: None,
            right: None,
            prediction: Some(pred),
        }
    }
    fn build_tree_recursive<T, S: Score<T>>(
        samples: &impl DataSet,
        target: &impl Target<T>,
        max_depth: usize,
        split_function: &S,
        split_info_parent: Option<&SplitInfo>,
    ) -> Result<Tree, TreeError> {
        if max_depth == 0 {
            return Ok(Tree::build_leaf(target, split_function));
        }
        match samples.find_best_split(target, split_function) {
            Ok((split_info, mask)) => {
                //Not really sure why logit does not fit correctly with this one
                if let Some(_) = split_info_parent {
                    if split_info.score.score == 0. {
                        return Ok(Tree::build_leaf(target, split_function));
                    }
                }
                let (left_samples, right_samples) =
                    samples.split(mask.clone(), split_info.score.null_direction);
                let (left_tar, right_tar) = target.split(mask, split_info.score.null_direction);
                let left_tree = Self::build_tree_recursive(
                    &left_samples,
                    &left_tar,
                    max_depth,
                    split_function,
                    Some(&split_info),
                )?;
                let right_tree = Self::build_tree_recursive(
                    &right_samples,
                    &right_tar,
                    max_depth,
                    split_function,
                    Some(&split_info),
                )?;
                Ok(Tree {
                    split_info: Some(split_info),
                    left: Some(Box::new(left_tree)),
                    right: Some(Box::new(right_tree)),
                    prediction: None,
                })
            }
            Err(error) => match error {
                split::BestSplitNotFound::NoSplitRequired => {
                    Ok(Tree::build_leaf(target, split_function))
                }
                split::BestSplitNotFound::Score(score_err) => match score_err {
                    loss_fn::ScoreError::InvalidSplit(_) | loss_fn::ScoreError::PerfectSplit => {
                        Ok(Tree::build_leaf(target, split_function))
                    }
                    _ => panic!("Could not split data: {}", score_err),
                },
                _ => panic!("Could not split data: {}", error),
            },
        }
    }
    fn predict_single_value<'a, T: Into<f64> + Copy>(
        &'a self,
        sample: &'a [(&'a str, Option<T>)],
    ) -> Result<f64, TreeError> {
        if let (Some(split_info), Some(l), Some(r)) = (
            self.split_info.as_ref(),
            self.left.as_ref(),
            self.right.as_ref(),
        ) {
            let (_, val) = sample
                .iter()
                .find(|(name, _)| split_info.name.eq(name))
                .expect(format!("Feature {} not in dataset", split_info.name).as_str());
            match val {
                Some(val) => {
                    if (*val).into() < split_info.value {
                        l.predict_single_value(sample)
                    } else {
                        r.predict_single_value(sample)
                    }
                }
                None => match split_info.score.null_direction {
                    loss_fn::split_values::NullDirection::Left => l.predict_single_value(sample),
                    loss_fn::split_values::NullDirection::Right => r.predict_single_value(sample),
                },
            }
        } else {
            self.prediction.ok_or(TreeError::NoPredictionInLeaf)
        }
    }
    pub fn predict(&self, samples: &impl DataSet) -> Result<Vec<f64>, TreeError> {
        samples
            .rows()?
            .map(|row| self.predict_single_value(row?.as_slice()))
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use loss_fn::{split_values::SplitScore, ScoringFunction};

    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_tree() {
        let data = HashMap::from([("F1".to_string(), vec![1., 2., 3.])]);
        let target = vec![true, false, false];
        let tree_config = TreeConfig { max_depth: 2 };
        let score_fn = ScoringFunction::Gini(loss_fn::Gini);
        let tree = Tree::fit(&data, &target, &tree_config, &score_fn);
        let output_tree = Tree {
            split_info: Some(SplitInfo::new(
                "F1".to_string(),
                2.,
                SplitScore {
                    score: 0.,
                    null_direction: loss_fn::split_values::NullDirection::Left,
                },
            )),
            left: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(1.0),
            })),
            right: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(0.0),
            })),
            prediction: None,
        };
        assert_eq!(
            output_tree,
            tree.expect("Tree did not fit correctly"),
            "Tree did not fit correctly"
        );
    }
    #[test]
    fn test_with_logit() {
        let data = HashMap::from([("F1".to_string(), vec![1., 2., 3.])]);
        let target = vec![true, false, false];
        let tree_config = TreeConfig { max_depth: 2 };
        let score_fn = ScoringFunction::Logit(loss_fn::Logit::new(0.5));
        let tree = Tree::fit(&data, &target, &tree_config, &score_fn);
        let output_tree = Tree {
            split_info: Some(SplitInfo::new(
                "F1".to_string(),
                2.,
                SplitScore {
                    score: -8. / 3.,
                    null_direction: loss_fn::split_values::NullDirection::Left,
                },
            )),
            left: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(2.0),
            })),
            right: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(-2.0),
            })),
            prediction: None,
        };
        assert_eq!(
            output_tree,
            tree.expect("Tree did not fit correctly"),
            "Tree did not fit correctly"
        );
    }
    #[test]
    fn test_prediction() {
        let output_tree = Tree {
            split_info: Some(SplitInfo::new(
                "F1".to_string(),
                2.,
                SplitScore {
                    score: -8. / 3.,
                    null_direction: loss_fn::split_values::NullDirection::Left,
                },
            )),
            left: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(2.0),
            })),
            right: Some(Box::new(Tree {
                split_info: None,
                left: None,
                right: None,
                prediction: Some(-2.0),
            })),
            prediction: None,
        };
        let dataset = HashMap::from([("F1".to_string(), vec![1., 3.])]);
        let pred = output_tree.predict(&dataset).unwrap();
        assert_eq!(vec![2., -2.], pred, "Wrong predictions")
    }
}

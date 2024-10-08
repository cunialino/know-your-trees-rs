use loss_fn::{split_values::SplitInfo, Score};
use split::{DataSet, Target};

pub mod loss_fn;
pub mod split;

#[derive(Debug, Default)]
pub struct TreeConfig {
    pub max_depth: usize,
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
        samples: &mut impl DataSet,
        target: &mut impl Target<T>,
        tree_config: &TreeConfig,
        score_fn: &S,
    ) -> Option<Box<Tree>> {
        let max_depth = tree_config.max_depth.clone();
        Tree::build_tree_recursive(samples, target, max_depth, score_fn)
    }
    fn build_leaf<T, S: Score<T>>(target: &impl Target<T>, split_function: &S) -> Box<Tree> {
        let pred = split_function.pred(target);
        Box::new(Tree {
            split_info: None,
            left: None,
            right: None,
            prediction: Some(pred),
        })
    }
    fn add_or_stop<T, S: Score<T>>(
        split_info: &SplitInfo,
        node: Tree,
        target: &impl Target<T>,
        score_fn: &S,
    ) -> Tree {
        if node.prediction.is_some() {
            node
        } else if let Some(ns) = node.split_info.as_ref() {
            if ns.score.score != split_info.score.score {
                *Self::build_leaf(target, score_fn)
            } else {
                node
            }
        } else {
            *Self::build_leaf(target, score_fn)
        }
    }
    fn build_tree_recursive<T, S: Score<T>>(
        samples: &mut impl DataSet,
        target: &mut impl Target<T>,
        max_depth: usize,
        split_function: &S,
    ) -> Option<Box<Tree>> {
        if max_depth == 0 {
            return Some(Tree::build_leaf(target, split_function));
        }
        match samples.find_best_split(target, split_function) {
            Ok(split_info) => {
                // this feels like a nightmare came true, collecting AND cloning.
                // There MUST be a smarter way. For the sake of movin on, I'll just
                // do it this way for now.
                let mask = samples
                    .rows()
                    .map(|row| {
                        let value = row
                            .iter()
                            .find(|(name, _)| **name == split_info.name)
                            .map(|(_, val)| (*val).into());

                        value.map(|v| v < split_info.value)
                    })
                    .collect::<Vec<Option<bool>>>();
                let mut right_samples =
                    samples.split(mask.clone().into_iter(), split_info.score.null_direction);
                let mut right_tar =
                    target.split(mask.clone().into_iter(), split_info.score.null_direction);
                let max_depth = if split_info.score.score == 0. {
                    0
                } else {
                    max_depth - 1
                };
                let left_tree =
                    Self::build_tree_recursive(samples, target, max_depth, split_function);
                let right_tree = Self::build_tree_recursive(
                    &mut right_samples,
                    &mut right_tar,
                    max_depth,
                    split_function,
                );
                if let (Some(left_tree), Some(right_tree)) = (left_tree, right_tree) {
                    let left_tree =
                        Self::add_or_stop(&split_info, *left_tree, target, split_function);
                    let right_tree =
                        Self::add_or_stop(&split_info, *right_tree, &right_tar, split_function);
                    Some(Box::new(Tree {
                        split_info: Some(split_info),
                        left: Some(Box::new(left_tree)),
                        right: Some(Box::new(right_tree)),
                        prediction: None,
                    }))
                } else {
                    Some(Tree::build_leaf(target, split_function))
                }
            }
            Err(error) => match error {
                split::BestSplitNotFound::NoSplitRequired => {
                    Some(Tree::build_leaf(target, split_function))
                }
                split::BestSplitNotFound::Score(score_err) => match score_err {
                    loss_fn::ScoreError::InvalidSplit(_) | loss_fn::ScoreError::PerfectSplit => {
                        Some(Tree::build_leaf(target, split_function))
                    }
                    _ => panic!("Could not split data: {}", error),
                },
                _ => panic!("Could not split data: {}", error),
            },
        }
    }
    fn predict_single_value<'a, T: Into<f64> + Copy>(&'a self, sample: &'a [(&'a str, T)]) -> f64 {
        if let (Some(split_info), Some(l), Some(r)) = (
            self.split_info.as_ref(),
            self.left.as_ref(),
            self.right.as_ref(),
        ) {
            let (_, val) = sample
                .iter()
                .find(|(name, _)| split_info.name.eq(name))
                .expect(format!("Feature {} not in dataset", split_info.name).as_str());
            if (*val).into() < split_info.value {
                l.predict_single_value(sample)
            } else {
                r.predict_single_value(sample)
            }
        } else {
            self.prediction
                .expect("Something went wrong in building the tree")
        }
    }
    pub fn predict(&self, samples: impl DataSet) -> Vec<f64> {
        samples
            .rows()
            .map(|row| self.predict_single_value(row.as_slice()))
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
        let mut data = HashMap::from([("F1".to_string(), vec![1., 2., 3.])]);
        let mut target = vec![true, false, false];
        let tree_config = TreeConfig { max_depth: 2 };
        let score_fn = ScoringFunction::Gini(loss_fn::Gini);
        let tree = Tree::fit(&mut data, &mut target, &tree_config, &score_fn);
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
            Some(Box::new(output_tree)),
            tree,
            "Tree did not fit correctly"
        );
    }
    #[test]
    fn test_with_logit() {
        let mut data = HashMap::from([("F1".to_string(), vec![1., 2., 3.])]);
        let mut target = vec![true, false, false];
        let tree_config = TreeConfig { max_depth: 2 };
        let score_fn = ScoringFunction::Logit(loss_fn::Logit::new(0.5));
        let tree = Tree::fit(&mut data, &mut target, &tree_config, &score_fn);
        let output_tree = Tree {
            split_info: Some(SplitInfo::new(
                "F1".to_string(),
                2.,
                SplitScore {
                    score: -3.,
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
            Some(Box::new(output_tree)),
            tree,
            "Tree did not fit correctly"
        );
    }
}

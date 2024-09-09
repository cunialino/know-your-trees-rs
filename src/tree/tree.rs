use super::predictions::{generate_prediction_function, PredictionFn};
use super::scores::{generate_score_function, ScoreConfig, SplitScoreFn};
use super::split::{best_split, SplitValue};
use arrow::array::Array;
use arrow::compute::{filter, filter_record_batch, not};
use arrow::record_batch::RecordBatch;
use std::usize;

#[derive(Debug, Default)]
pub struct TreeConfig {
    max_depth: usize,
}

#[derive(Debug, PartialEq)]
pub struct Tree {
    pub feature_index: Option<usize>,
    pub threshold: Option<SplitValue>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub prediction: Option<f64>, // Optional: only used at leaf nodes
}

impl Tree {
    pub fn fit(
        samples: RecordBatch,
        target: &dyn Array,
        tree_config: &TreeConfig,
        score_config: &ScoreConfig,
    ) -> Option<Box<Tree>> {
        let max_depth = tree_config.max_depth.clone();
        let split_function = generate_score_function(score_config);
        let prediction_function = generate_prediction_function();
        Tree::build_tree_recursive(samples, target, max_depth, &split_function, &prediction_function)
    }
    fn build_tree_recursive(
        samples: RecordBatch,
        target: &dyn Array,
        max_depth: usize,
        split_function: &SplitScoreFn,
        prediction_function: &PredictionFn,
    ) -> Option<Box<Tree>> {
        if max_depth == 0 || samples.num_rows() == 0 {
            return None;
        }
        if let Some((_, col_index, data_mask, th)) = best_split(&samples, target, split_function)
        {
            let tree = Tree {
                feature_index: Some(col_index),
                threshold: Some(th),
                left: Self::build_tree_recursive(
                    filter_record_batch(&samples, &data_mask).unwrap(),
                    filter(target, &data_mask).unwrap().as_ref(),
                    max_depth - 1,
                    split_function,
                    prediction_function,
                ),
                right: Self::build_tree_recursive(
                    filter_record_batch(&samples, &not(&data_mask).unwrap()).unwrap(),
                    filter(target, &not(&data_mask).unwrap()).unwrap().as_ref(),
                    max_depth - 1,
                    split_function,
                    prediction_function,
                ),
                prediction: None,
            };
            return Some(Box::new(tree));
        }

        return Some(Box::new(Tree {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            prediction: Some(prediction_function(target)),
        }));
    }
}

#[cfg(test)]
mod tests {

    use crate::tree::scores::SplitScores;
    use crate::tree::scores::WeightedSplitScores;

    use super::*;
    use arrow::array::ArrayRef;
    use arrow::{
        array::{BooleanArray, Float32Array},
        datatypes::{DataType, Field, Schema},
    };
    use std::sync::Arc;

    #[test]
    fn test_tree() {
        let my_schema: Arc<Schema> = Arc::new(Schema::new(vec![Field::new(
            "sample",
            DataType::Float32,
            false,
        )]));

        let data: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let target: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, false]));
        let tree_config = TreeConfig { max_depth: 2 };
        let score_config = ScoreConfig {
            score_function: SplitScores::Weighted(WeightedSplitScores::Gini),
            initial_prediction: None,
        };
        let tree = Tree::fit(
            RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            &target,
            &tree_config,
            &score_config,
        );
        let output_tree = Tree {
            feature_index: Some(0),
            threshold: Some(SplitValue::Numeric(2.0)),
            left: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(1.0),
            })),
            right: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(0.0),
            })),
            prediction: None,
        };
        assert_eq!(
            tree,
            Some(Box::new(output_tree)),
            "Tree did not fit correctly"
        );
    }
}

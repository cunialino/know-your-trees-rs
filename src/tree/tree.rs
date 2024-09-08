use super::predictions::Prediction;
use super::scores::ScoreFunction;
use super::split::{best_split, SplitValue};
use arrow::array::ArrayRef;
use arrow::compute::{filter, filter_record_batch, not};
use arrow::record_batch::RecordBatch;
use std::usize;


#[derive(Debug, PartialEq)]
pub struct Tree {
    pub feature_index: Option<usize>,
    pub threshold: Option<SplitValue>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub prediction: Option<f64>, // Optional: only used at leaf nodes
}

impl Tree {
    pub fn build_tree(
        samples: RecordBatch,
        target: ArrayRef,
        max_depth: usize,
        split_function: &ScoreFunction,
    ) -> Option<Box<Tree>> {
        if max_depth == 0 || samples.num_rows() == 0 {
            return None;
        }
        if let Some((_, col_index, data_mask, th)) =
            best_split(&samples, target.clone(), split_function, None)
        {
            let tree = Tree {
                feature_index: Some(col_index),
                threshold: Some(th),
                left: Self::build_tree(
                    filter_record_batch(&samples, &data_mask).unwrap(),
                    filter(&target, &data_mask).unwrap(),
                    max_depth - 1,
                    split_function,
                ),
                right: Self::build_tree(
                    filter_record_batch(&samples, &not(&data_mask).unwrap()).unwrap(),
                    filter(&target, &not(&data_mask).unwrap()).unwrap(),
                    max_depth - 1,
                    split_function,
                ),
                prediction: None,
            };
            return Some(Box::new(tree));
        }
        target.frequency();

        return Some(Box::new(Tree {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            prediction: Some(target.frequency()),
        }));
    }
}

#[cfg(test)]
mod tests {

    use crate::tree::scores::WeightedScoreFunction;

    use super::*;
    use arrow::{
        array::{Float32Array, BooleanArray},
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
        let tree = Tree::build_tree(
            RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            target,
            2,
            &ScoreFunction::Weighted(WeightedScoreFunction::Gini),
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

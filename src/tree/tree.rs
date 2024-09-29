use super::array_traits::ArrayConversions;
use super::scores::{generate_score_function, NullDirection, PredFnType, ScoreConfig, SplitFnType};
use super::split::{best_split, SplitValue};
use arrow::array::{Array, Float64Array};
use arrow::compute::{filter, filter_record_batch, not};
use arrow::record_batch::RecordBatch;
use std::usize;

#[derive(Debug, Default)]
pub struct TreeConfig {
    pub max_depth: usize,
}

#[derive(Debug, PartialEq)]
pub struct Tree {
    pub feature_index: Option<String>,
    pub threshold: Option<SplitValue>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub null_direction: Option<NullDirection>,
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
        let (split_function, prediction_function) = generate_score_function(score_config);
        Tree::build_tree_recursive(
            samples,
            target,
            max_depth,
            &split_function,
            &prediction_function,
        )
    }
    fn build_tree_recursive(
        samples: RecordBatch,
        target: &dyn Array,
        max_depth: usize,
        split_function: &SplitFnType,
        prediction_function: &PredFnType,
    ) -> Option<Box<Tree>> {
        if max_depth == 0 || samples.num_rows() == 0 {
            return None;
        }
        if let Some((split_score, col_index, data_mask, th)) =
            best_split(&samples, target, split_function)
        {
            Some(Box::new(Tree {
                feature_index: Some(col_index),
                threshold: Some(th),
                null_direction: Some(split_score.null_direction),
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
            }))
        } else {
            Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                null_direction: None,
                prediction: Some(prediction_function(target)),
            }))
        }
    }
    fn predict_single_value(&self, samples: &RecordBatch) -> f64 {
        assert!(
            samples.num_rows() == 1,
            "Expected one record only in predict_single_value"
        );
        if let (Some(feat_name), Some(l), Some(r), Some(split_value), Some(null_direction)) = (
            self.feature_index.as_ref(),
            self.left.as_ref(),
            self.right.as_ref(),
            self.threshold.as_ref(),
            self.null_direction.as_ref(),
        ) {
            let col = samples
                .column_by_name(feat_name)
                .expect(format!("Column {feat_name} not present for prediction").as_str());
            match split_value {
                SplitValue::String(_) => todo!("Prediction on strins not implemented yet"),
                SplitValue::Numeric(sv) => {
                    let val = col.try_into_iter_f64().nth(0).unwrap();
                    val.map_or_else(
                        || match null_direction {
                            NullDirection::Left => l.predict_single_value(&samples),
                            NullDirection::Right => r.predict_single_value(&samples),
                        },
                        |v| {
                            if v < *sv {
                                l.predict_single_value(&samples)
                            } else {
                                r.predict_single_value(&samples)
                            }
                        },
                    )
                }
            }
        } else {
            self.prediction
                .expect("Something went wrong in building the tree")
        }
    }
    pub fn predict(&self, samples: &RecordBatch) -> Float64Array {
        (0..samples.num_rows())
            .into_iter()
            .map(|row_num| {
                let row = samples.slice(row_num, 1);
                self.predict_single_value(&row)
            })
            .collect()
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
            true,
        )]));

        let data: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.), Some(2.), None]));
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
            feature_index: Some("sample".to_string()),
            threshold: Some(SplitValue::Numeric(2.0)),
            null_direction: Some(NullDirection::Left),
            left: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(1.0),
                null_direction: None,
            })),
            right: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                null_direction: None,
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
    #[test]
    fn test_prediction() {
        let tree = Tree {
            feature_index: Some("sample".to_string()),
            threshold: Some(SplitValue::Numeric(2.0)),
            null_direction: Some(NullDirection::Left),
            left: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(1.0),
                null_direction: None,
            })),
            right: Some(Box::new(Tree {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                prediction: Some(0.0),
                null_direction: None,
            })),
            prediction: None,
        };
        let my_schema: Arc<Schema> = Arc::new(Schema::new(vec![Field::new(
            "sample",
            DataType::Float32,
            true,
        )]));

        let data: ArrayRef = Arc::new(Float32Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            None,
        ]));
        let samples = RecordBatch::try_new(my_schema, vec![data]).unwrap();
        let out = tree.predict(&samples);
        assert_eq!(out, Float64Array::from(vec![1., 0., 0., 1.]));
    }
}

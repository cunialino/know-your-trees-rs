use super::scores::{Score, ScoreFunction};
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::compute::{filter, filter_record_batch, not};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float32Type, Float64Type, Int32Type};
use arrow::record_batch::RecordBatch;
use std::usize;

#[derive(Debug, PartialEq)]
pub enum SplitValue {
    Numeric(f64),
    String(String),
}

trait SplitType {
    fn filter_mask(&self, split_point: &SplitValue) -> BooleanArray;
    fn possible_splits_iter(&self) -> impl Iterator<Item = (SplitValue, BooleanArray)>;
}

impl<D> SplitType for PrimitiveArray<D>
where
    D: ArrowPrimitiveType,
    D::Native: Into<f64>,
{
    fn filter_mask(&self, split_point: &SplitValue) -> BooleanArray {
        if let SplitValue::Numeric(numeric_point) = split_point {
            let split_point = *numeric_point;
            BooleanArray::from(
                self.values()
                    .iter()
                    .map(|&value| value.into() < split_point)
                    .collect::<Vec<bool>>(),
            )
        } else {
            panic!("Mismatched split point type for numeric array")
        }
    }
    fn possible_splits_iter(&self) -> impl Iterator<Item = (SplitValue, BooleanArray)> {
        self.values().iter().map(move |&split_point| {
            (
                SplitValue::Numeric(split_point.into()),
                self.filter_mask(&SplitValue::Numeric(split_point.into())),
            )
        })
    }
}

struct Split {
    target: ArrayRef,
    data: RecordBatch,
}

impl Split {
    fn split_points_iterator(
        &self,
        column_index: usize,
    ) -> impl Iterator<Item = (f64, BooleanArray, SplitValue)> + '_ {
        let data_ref = self.data.column(column_index);
        let possible_splits_iter: Box<dyn Iterator<Item = (SplitValue, BooleanArray)>> =
            match data_ref.data_type() {
                DataType::Float32 => Box::new(
                    data_ref
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Float32Type>>()
                        .unwrap()
                        .possible_splits_iter(),
                ),
                DataType::Float64 => Box::new(
                    data_ref
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Float64Type>>()
                        .unwrap()
                        .possible_splits_iter(),
                ),
                DataType::Int32 => Box::new(
                    data_ref
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Int32Type>>()
                        .unwrap()
                        .possible_splits_iter(),
                ),
                _ => panic!("Invalid data type"),
            };
        possible_splits_iter.map(|(split_value, boolean_array)| {
            let split_score = self
                .target
                .split_score(&boolean_array, ScoreFunction::Gini);
            (split_score, boolean_array, split_value)
        })
    }

    fn best_split(&self) -> Option<(usize, BooleanArray, SplitValue)> {
        (0..self.data.num_columns())
            .map(|column_index| {
                self.split_points_iterator(column_index).map(
                    move |(split_score, filter_mask, split_value)| {
                        (split_score, column_index, filter_mask, split_value)
                    },
                )
            })
            .flatten()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, column_index, mask, threshold)| (column_index, mask, threshold))
    }
}
#[derive(Debug)]
pub struct Tree {
    pub feature_index: Option<usize>,
    pub threshold: Option<SplitValue>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub prediction: Option<bool>, // Optional: only used at leaf nodes
}

impl Tree {
    pub fn new(feature_index: usize, threshold: f64) -> Self {
        Tree {
            feature_index: Some(feature_index),
            threshold: Some(SplitValue::Numeric(threshold)),
            left: None,
            right: None,
            prediction: None,
        }
    }
    pub fn build_tree(
        samples: RecordBatch,
        target: ArrayRef,
        max_depth: usize,
    ) -> Option<Box<Tree>> {
        if max_depth == 0 || samples.num_rows() == 0 {
            return None;
        }
        let split = Split {
            data: samples.clone(),
            target: target.clone(),
        };
        if let Some((col_index, data_mask, th)) = split.best_split() {
            let tree = Tree {
                feature_index: Some(col_index),
                threshold: Some(th),
                left: Self::build_tree(
                    filter_record_batch(&samples, &data_mask).unwrap(),
                    filter(&target, &data_mask).unwrap(),
                    max_depth - 1,
                ),
                right: Self::build_tree(
                    filter_record_batch(&samples, &not(&data_mask).unwrap()).unwrap(),
                    filter(&target, &not(&data_mask).unwrap()).unwrap(),
                    max_depth - 1,
                ),
                prediction: None,
            };
            return Some(Box::new(tree));
        }
        None
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow::{
        array::{Float32Array, Int32Array},
        datatypes::{DataType, Field, Schema},
    };
    use std::sync::Arc;

    #[test]
    fn test_best_split_ints() {
        let my_schema: Arc<Schema> = Arc::new(Schema::new(vec![Field::new(
            "sample",
            DataType::Int32,
            false,
        )]));

        let data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let target: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let nn = Split {
            target,
            data: RecordBatch::try_new(my_schema, vec![data]).unwrap(),
        };

        // Since best_split may return None, handle it properly
        let (row_index, filter_mask, threshold) = nn.best_split().expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(row_index, 0, "Wrong column index");
        assert_eq!(filter_mask, BooleanArray::from(vec![true, false]));
        assert_eq!(threshold, SplitValue::Numeric(2.), "Wrong threshold");
    }
    #[test]
    fn test_best_split() {
        let my_schema: Arc<Schema> = Arc::new(Schema::new(vec![Field::new(
            "sample",
            DataType::Float32,
            false,
        )]));

        let data: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0]));
        let target: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let nn = Split {
            target,
            data: RecordBatch::try_new(my_schema, vec![data]).unwrap(),
        };

        // Since best_split may return None, handle it properly
        let (row_index, filter_mask, threshold) = nn.best_split().expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(row_index, 0, "Wrong column index");
        assert_eq!(filter_mask, BooleanArray::from(vec![true, false]));
        assert_eq!(threshold, SplitValue::Numeric(2.), "Wrong threshold");
    }

    #[test]
    fn test_tree() {
        let my_schema: Arc<Schema> = Arc::new(Schema::new(vec![Field::new(
            "sample",
            DataType::Float32,
            false,
        )]));

        let data: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0]));
        let target: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let tree = Tree::build_tree(
            RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            target,
            2,
        );
        dbg!(tree);
    }
}

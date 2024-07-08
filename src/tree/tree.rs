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

trait Split {
    fn filter_mask(&self, split_point: &SplitValue) -> BooleanArray;
    fn possible_splits_iter(&self) -> impl Iterator<Item = (SplitValue, BooleanArray)>;
}

impl<D> Split for PrimitiveArray<D>
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
fn downcast_and_possible_splits<T>(
    data_ref: &ArrayRef,
) -> Box<dyn Iterator<Item = (SplitValue, BooleanArray)> + '_>
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64>,
{
    let array = data_ref
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap();
    Box::new(array.possible_splits_iter())
}

fn best_split(data: &RecordBatch, target: ArrayRef) -> Option<(f64, usize, BooleanArray, SplitValue)> {
    let iters = (0..data.num_columns())
        .flat_map(|column_index| {
            let col = data.column(column_index);
            match col.data_type() {
                DataType::Float32 => downcast_and_possible_splits::<Float32Type>(col),
                DataType::Float64 => downcast_and_possible_splits::<Float64Type>(col),
                DataType::Int32 => downcast_and_possible_splits::<Int32Type>(col),
                _ => panic!("Invalid data type"),
            }
            .map(|(split_value, filter_mask)| {
                let split_score = target.split_score(&filter_mask, ScoreFunction::Gini);
                (split_score, filter_mask, split_value)
            })
            .map(move |(split_score, filter_mask, split_value)| {
                (split_score, column_index, filter_mask, split_value)
            })
        });
        iters.max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
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
        if let Some((_, col_index, data_mask, th)) = best_split(&samples, target.clone()) {
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

        let (_, row_index, filter_mask, threshold) = best_split(&RecordBatch::try_new(my_schema, vec![data]).unwrap(), target).expect("No split found");

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

        // Since best_split may return None, handle it properly
        let (_, row_index, filter_mask, threshold) = best_split(&RecordBatch::try_new(my_schema, vec![data]).unwrap(), target).expect("No split found");

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

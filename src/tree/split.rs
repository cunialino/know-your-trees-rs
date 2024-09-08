use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float32Type, Float64Type, Int32Type};
use arrow::record_batch::RecordBatch;

use std::usize;

use super::scores::ScoreFn;

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
pub fn best_split(
    data: &RecordBatch,
    target: &ArrayRef,
    split_function: &ScoreFn,
) -> Option<(f64, usize, BooleanArray, SplitValue)> {
    let iters = (0..data.num_columns()).flat_map(move |column_index| {
        let col = data.column(column_index);
        match col.data_type() {
            DataType::Float32 => downcast_and_possible_splits::<Float32Type>(col),
            DataType::Float64 => downcast_and_possible_splits::<Float64Type>(col),
            DataType::Int32 => downcast_and_possible_splits::<Int32Type>(col),
            _ => panic!("Invalid data type"),
        }
        .filter_map(move |(split_value, filter_mask)| {
            let score = split_function(target, &filter_mask);
            if score.is_some() {
                Some((score.unwrap(), column_index, filter_mask, split_value))
            } else {
                None
            }
        })
    });

    iters.min_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect(format!("Cannot compare {} with {}.", a.0, b.0).as_str())
    })
}

#[cfg(test)]
mod tests {

    use crate::tree::scores::generate_score_function;
    use crate::tree::scores::ScoreConfig;
    use crate::tree::scores::ScoreFunction;
    use crate::tree::scores::WeightedScoreFunction;

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

        let score_config = ScoreConfig {
            score_function: ScoreFunction::Weighted(WeightedScoreFunction::Gini),
            initial_prediction: None,
        };
        let score_fn = generate_score_function(&score_config);
        let (_, row_index, filter_mask, threshold) = best_split(
            &RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            &target,
            &score_fn,
        )
        .expect("No split found");

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
        let score_config = ScoreConfig {
            score_function: ScoreFunction::Weighted(WeightedScoreFunction::Gini),
            initial_prediction: None,
        };
        let score_fn = generate_score_function(&score_config);
        let (_, row_index, filter_mask, threshold) = best_split(
            &RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            &target,
            &score_fn,
        )
        .expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(row_index, 0, "Wrong column index");
        assert_eq!(filter_mask, BooleanArray::from(vec![true, false]));
        assert_eq!(threshold, SplitValue::Numeric(2.), "Wrong threshold");
    }
}

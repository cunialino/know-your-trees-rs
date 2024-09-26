use core::f64;

use arrow::array::{Array, BooleanArray};
use arrow::record_batch::RecordBatch;

use super::array_traits::ArrayConversions;
use super::scores::{SplitFnType, SplitScore};

#[derive(Debug, PartialEq)]
pub enum SplitValue {
    Numeric(f64),
    String(Vec<String>),
}

fn filter_mask(feature: &dyn Array, split_value: &SplitValue) -> BooleanArray {
    match split_value {
        SplitValue::Numeric(split_point) => BooleanArray::from(
            feature
                .try_into_iter_f64()
                .map(|v| v.map(|sm| sm < *split_point))
                .collect::<std::vec::Vec<_>>(),
        ),
        SplitValue::String(_) => todo!("Categorical features not yet implemented"),
    }
}
fn possible_splits_iter(
    feature: &dyn Array,
) -> impl Iterator<Item = (SplitValue, BooleanArray)> + '_ {
    match feature.data_type().is_numeric() {
        true => Box::new(feature.try_into_iter_f64().filter_map(|split_point| {
            if split_point.is_none() {
                None
            } else {
                let split_pt = SplitValue::Numeric(split_point.unwrap());
                let bl_mask = filter_mask(feature, &split_pt);
                Some((split_pt, bl_mask))
            }
        })),
        false => todo!("Categorical features not yet implemented"),
    }
}
pub fn best_split(
    data: &RecordBatch,
    target: &dyn Array,
    split_function: &SplitFnType,
) -> Option<(SplitScore, String, BooleanArray, SplitValue)> {
    data.schema()
        .fields()
        .iter()
        .flat_map(|field| {
            let col = data.column_by_name(field.name()).unwrap();
            possible_splits_iter(col)
                .map(|(split_value, filter_mask)| (field.name(), split_value, filter_mask))
        })
        .filter_map(|(name, split_value, filter_mask)| {
            let score = split_function(target, &filter_mask);
            if score.is_some() {
                Some((score.unwrap(), name.to_string(), filter_mask, split_value))
            } else {
                None
            }
        })
        .min_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect(format!("Cannot compare {} with {}.", a.0, b.0).as_str())
        })
}

#[cfg(test)]
mod tests {

    use crate::tree::scores::generate_score_function;
    use crate::tree::scores::ScoreConfig;
    use crate::tree::scores::SplitScores;
    use crate::tree::scores::WeightedSplitScores;

    use super::*;
    use arrow::{
        array::{ArrayRef, Float32Array, Int32Array},
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
            score_function: SplitScores::Weighted(WeightedSplitScores::Gini),
            initial_prediction: None,
        };
        let (score_fn, _) = generate_score_function(&score_config);
        let (_, row_index, filter_mask, threshold) = best_split(
            &RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            &target,
            &score_fn,
        )
        .expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(row_index, "sample", "Wrong column index");
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
            score_function: SplitScores::Weighted(WeightedSplitScores::Gini),
            initial_prediction: None,
        };
        let (score_fn, _) = generate_score_function(&score_config);
        let (_, row_index, filter_mask, threshold) = best_split(
            &RecordBatch::try_new(my_schema, vec![data]).unwrap(),
            &target,
            &score_fn,
        )
        .expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(row_index, "sample", "Wrong column index");
        assert_eq!(filter_mask, BooleanArray::from(vec![true, false]));
        assert_eq!(threshold, SplitValue::Numeric(2.), "Wrong threshold");
    }
    #[test]
    fn test_mask_split() {
        let data: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.), None, Some(2.0)]));
        let split_v = SplitValue::Numeric(2.0);
        let result = filter_mask(&data, &split_v);
        let output: BooleanArray = vec![Some(true), None, Some(false)].into();
        assert_eq!(output, result, "Filtermask broken")
    }
    #[test]
    fn test_possible_splits() {
        let data: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.), None, Some(2.0)]));
        let mut possible_splits_iters = possible_splits_iter(&data);
        let first_output = Some((
            SplitValue::Numeric(1.),
            vec![Some(false), None, Some(false)].into(),
        ));
        let second_result = Some((
            SplitValue::Numeric(2.),
            vec![Some(true), None, Some(false)].into(),
        ));

        assert_eq!(
            first_output,
            possible_splits_iters.nth(0),
            "You dumb little shit"
        );
        assert_eq!(
            second_result,
            possible_splits_iters.nth(0),
            "You dumb little shit"
        );
    }
}

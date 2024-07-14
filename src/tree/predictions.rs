use arrow::array::{Array, ArrayRef, BooleanArray};
use std::collections::HashMap;
#[derive(Debug, PartialEq)]
pub enum PredictionValue {
    Binary(bool),
}
pub trait Prediction {
    fn frequency(&self) -> PredictionValue;
}

impl Prediction for ArrayRef {
    fn frequency(&self) -> PredictionValue {
        let mut class_counts = HashMap::new();
        let boolean_array = self
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        for label in boolean_array.values().iter().map(|x| x) {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        PredictionValue::Binary(
            *class_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1))
                .map(|(k, _v)| k)
                .unwrap(),
        )
    }
}

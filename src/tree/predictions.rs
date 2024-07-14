use arrow::array::{Array, ArrayRef, BooleanArray};

pub trait Prediction {
    fn frequency(&self) -> f64;
}

impl Prediction for ArrayRef {
    fn frequency(&self) -> f64 {
        let boolean_array = self
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        boolean_array.true_count() as f64 / boolean_array.len() as f64
    }
}

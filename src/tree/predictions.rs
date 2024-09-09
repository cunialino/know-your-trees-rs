use arrow::array::{Array, BooleanArray};

pub type PredictionFn = dyn Fn(&dyn Array) -> f64;

fn frequency(arr: &dyn Array) -> f64 {
    let boolean_array = arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Gini index is valid only for boolean array");
    boolean_array.true_count() as f64 / boolean_array.len() as f64
}

pub fn generate_prediction_function() -> Box<PredictionFn> {
    Box::new(|arr: &dyn Array| frequency(arr))
}

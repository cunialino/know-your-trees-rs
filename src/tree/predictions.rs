use arrow::array::{Array, BooleanArray};

use super::scores::{generate_loss_function, LossFn, ScoreConfig};

fn frequency(arr: &dyn Array) -> f64 {
    let boolean_array = arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Gini index is valid only for boolean array");
    boolean_array.true_count() as f64 / boolean_array.len() as f64
}

pub fn generate_prediction_function(score_config: &ScoreConfig) -> Box<LossFn> {
    let (score_fn, is_wgt) = generate_loss_function(score_config);
    if is_wgt {
        Box::new(|arr| frequency(arr))
    } else {
        Box::new(move |arr| score_fn(arr))
    }
}

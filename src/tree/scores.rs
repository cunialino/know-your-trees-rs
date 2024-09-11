use arrow::array::{Array, BooleanArray};
use arrow::compute::{filter, not};
use std::collections::HashMap;

pub type SplitScoreFn = dyn Fn(&dyn Array, &BooleanArray) -> Option<f64>;

#[derive(Debug, Clone, Copy)]
pub enum WeightedSplitScores {
    Gini,
}
#[derive(Debug, Clone, Copy)]
pub enum DifferentiableSplitScores {
    Logit,
}
#[derive(Debug, Clone, Copy)]
pub enum SplitScores {
    Weighted(WeightedSplitScores),
    Differentiable(DifferentiableSplitScores),
}

#[derive(Debug)]
pub struct ScoreConfig {
    pub score_function: SplitScores,
    pub initial_prediction: Option<f64>,
}

fn logit(target: &dyn Array, prediction: f64) -> f64 {
    let boolean_array = target
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Logit loss is valid only for boolean array");

    let (gradients, hessians) =
        boolean_array
            .iter()
            .fold((0.0, 0.0), |(grad_sum, hess_sum), target| {
                let target_value = target.map_or(0.0, |b| if b { 1.0 } else { 0.0 });
                let grad = prediction - target_value;
                let hess = prediction * (1.0 - prediction);
                (grad_sum + grad, hess_sum + hess)
            });

    if hessians == 0.0 {
        0.0
    } else {
        gradients.powi(2) / hessians
    }
}

fn gini(target: &dyn Array) -> f64 {
    let mut class_counts = HashMap::new();
    let boolean_array = target
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Gini index is valid only for boolean array");
    for label in boolean_array.values().iter() {
        *class_counts.entry(label).or_insert(0) += 1;
    }
    let total = target.len() as f64;
    let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
        let proportion = count as f64 / total;
        acc + proportion * proportion
    });
    1.0 - sum_of_squares
}
pub fn generate_score_function(score_config: &ScoreConfig) -> Box<SplitScoreFn> {
    let (score_fn, is_wgt): (Box<dyn Fn(&dyn Array) -> f64>, bool) =
        match score_config.score_function {
            SplitScores::Weighted(weighted_fn) => {
                if score_config.initial_prediction.is_some() {
                    panic!("Prediction should not be provided for weighted score functions");
                }
                (
                    match weighted_fn {
                        WeightedSplitScores::Gini => Box::new(|arr| gini(arr)),
                    },
                    true,
                )
            }
            SplitScores::Differentiable(diff_fn) => {
                let pred = score_config
                    .initial_prediction
                    .expect("Prediction must be provided for differentiable score functions");
                (
                    match diff_fn {
                        DifferentiableSplitScores::Logit => {
                            let pred = pred.clone();
                            Box::new(move |arr| logit(arr, pred))
                        }
                    },
                    false,
                )
            }
        };
    {
        let is_wgt = is_wgt.clone();
        Box::new(move |arr: &dyn Array, filter_mask: &BooleanArray| {
            let left_array = filter(arr, filter_mask).unwrap();
            let right_array = filter(arr, &not(filter_mask).unwrap()).unwrap();
            let (left_wgt, right_wgt) = match is_wgt {
                true => (1.0, 1.0),
                false => {
                    let left_len = filter_mask.true_count();
                    let total_len = filter_mask.len();
                    let right_len = total_len - left_len;
                    (
                        left_len as f64 / total_len as f64,
                        right_len as f64 / total_len as f64,
                    )
                }
            };
            if is_wgt && score_fn(arr) == 0. {
                None
            } else {
                Some(left_wgt * score_fn(&left_array) + right_wgt * score_fn(&right_array))
            }
        })
    }
}

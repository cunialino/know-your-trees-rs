use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::{filter, not};
use std::collections::HashMap;

#[derive(Debug)]
pub enum WeightedScoreFunction {
    Gini,
}
pub enum DifferentiableScoreFunction {
    Logit,
}
pub enum ScoreFunction {
    Weighted(WeightedScoreFunction),
    Differentiable(DifferentiableScoreFunction),
}
pub trait Score {
    fn gini(&self) -> f64;
    fn logit(&self, prediction: f64) -> f64;
    fn split_score(
        &self,
        filter_mask: &BooleanArray,
        score_function: &ScoreFunction,
        prediction: Option<f64>,
    ) -> Option<f64>;
}
impl Score for ArrayRef {
    fn gini(&self) -> f64 {
        let mut class_counts = HashMap::new();
        let boolean_array = self
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        for label in boolean_array.values().iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let total = self.len() as f64;
        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });
        1.0 - sum_of_squares
    }

    fn logit(&self, prediction: f64) -> f64 {
        let boolean_array = self
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

    fn split_score(
        &self,
        filter_mask: &BooleanArray,
        score_function: &ScoreFunction,
        prediction: Option<f64>,
    ) -> Option<f64> {
        let left_array = filter(self, filter_mask).unwrap();
        let right_array = filter(self, &not(filter_mask).unwrap()).unwrap();

        let calculate_score = |score_fn: Box<dyn Fn(&ArrayRef) -> f64>, is_wgt: bool| {
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
            Some(left_wgt * score_fn(&left_array) + right_wgt * score_fn(&right_array))
        };

        match score_function {
            ScoreFunction::Weighted(weighted_fn) => {
                if prediction.is_some() {
                    panic!("Prediction should not be provided for weighted score functions");
                }
                match weighted_fn {
                    WeightedScoreFunction::Gini => {
                        if self.gini() > 0. {
                            calculate_score(Box::new(|arr| arr.gini()), true)
                        } else {
                            None
                        }
                    }
                }
            }
            ScoreFunction::Differentiable(diff_fn) => {
                let pred = prediction
                    .expect("Prediction must be provided for differentiable score functions");
                match diff_fn {
                    DifferentiableScoreFunction::Logit => {
                        calculate_score(Box::new(|arr| arr.logit(pred)), true)
                    } // Add other differentiable score functions here
                }
            }
        }
    }
}

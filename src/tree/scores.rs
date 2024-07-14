use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::{filter, not};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ScoreFunction {
    Gini,
    Logit,
}
pub trait Score {
    fn gini(&self) -> f64;
    fn logit(&self, prediction: f64) -> f64;
    fn score(&self, score_function: &ScoreFunction) -> f64;
    fn split_score(&self, filter_mask: &BooleanArray, score_function: &ScoreFunction) -> f64;
}

impl Score for ArrayRef {
    fn gini(&self) -> f64 {
        let mut class_counts = HashMap::new();
        let boolean_array = self
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        for label in boolean_array.values().iter().map(|x| x) {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let total = self.len() as f64;

        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });

        sum_of_squares - 1.
    }
    fn logit(&self, prediction: f64) -> f64 {
        let boolean_array = self
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Logit loss is valid only for boolean array");
        boolean_array.iter().map(|target| match target {
            Some(true) => 1.0,
            Some(false) => 0.0,
            _ => panic!("No idea how i can get here"),
        }).map(|v| v * prediction.ln() + (1.-v) * (1. - prediction).ln()).sum()
    }
    fn score(&self, score_function: &ScoreFunction) -> f64 {
        match score_function {
            ScoreFunction::Gini => self.gini(),
            ScoreFunction::Logit => self.logit(0.5),
        }
    }
    fn split_score(&self, filter_mask: &BooleanArray, score_function: &ScoreFunction) -> f64 {
        let left_len = filter_mask.true_count();
        let total_len = filter_mask.len();
        let right_len = total_len - left_len;

        let left_wgt = left_len as f64 / total_len as f64;
        let right_wgt = right_len as f64 / total_len as f64;

        let left_array = filter(self, filter_mask).unwrap();
        let right_array = filter(self, &not(filter_mask).unwrap()).unwrap();
        left_wgt * left_array.score(score_function) + right_wgt * right_array.score(score_function)
    }
}

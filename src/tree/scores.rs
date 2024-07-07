use arrow::array::{Array, BooleanArray};
use arrow::compute::{filter, not};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ScoreFunction {
    Gini,
}
pub trait Score {
    fn gini(&self) -> f64;
    fn split_score(&self, filter_mask: &BooleanArray, score_function: ScoreFunction) -> f64;
}

impl Score for BooleanArray {
    fn gini(&self) -> f64 {
        let mut class_counts = HashMap::new();
        for label in self.values().iter().map(|x| x) {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let total = self.len() as f64;

        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });

        sum_of_squares - 1.
    }
    fn split_score(&self, filter_mask: &BooleanArray, score_function: ScoreFunction) -> f64 {
        let left_len = filter_mask.values().iter().filter(|&value| value).count();
        let total_len = filter_mask.len();
        let right_len = total_len - left_len;

        let left_wgt = left_len as f64 / total_len as f64;
        let right_wgt = right_len as f64 / total_len as f64;

        let left_array = filter(self, filter_mask).unwrap();
        let left_boolean_array = left_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Cannot downcast to BooleanArray");

        let right_array = filter(self, &not(filter_mask).unwrap()).unwrap();
        let right_boolean_array = right_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Cannot downcast to BooleanArray");

        match score_function {
            ScoreFunction::Gini => {
                let left_score = left_wgt * left_boolean_array.gini();
                let right_score = right_wgt * right_boolean_array.gini();
                left_score + right_score
            },
            // Add other score functions as needed
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow::array::BooleanArray;

    #[test]
    fn gini() {
        let target = BooleanArray::from(vec![true, false]);

        // Calculate Gini index
        let gini = target.gini();
        let expected_gini = -0.5; // 0.48
        let delta = 0.001;

        assert!(
            (gini - expected_gini).abs() < delta,
            "Calculated Gini index is incorrect: expected {:.3}, got {:.3}",
            expected_gini,
            gini
        );
    }
}

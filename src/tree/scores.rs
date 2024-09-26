use arrow::array::{Array, BooleanArray};
use arrow::compute::{filter, is_null, not};
use std::collections::HashMap;

pub type SplitFnType = dyn Fn(&dyn Array, &BooleanArray) -> Option<SplitScore>;
pub type PredFnType = dyn Fn(&dyn Array) -> f64;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum NullDirection {
    #[default]
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitScore {
    pub score: f64,
    pub null_direction: NullDirection,
}

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

impl SplitScore {
    fn new(score: f64, null_direction: NullDirection) -> SplitScore {
        SplitScore {
            score,
            null_direction,
        }
    }
}

trait DiffScore: Copy + Clone {
    fn grad_and_hess(&self, arr: &dyn Array) -> (f64, f64);
    fn pred(&self, arr: &dyn Array) -> f64 {
        let (grad, hess) = self.grad_and_hess(arr);
        if hess == 0.0 {
            0.0
        } else {
            grad.powi(2) / hess
        }
    }
    fn split_score(&self, arr: &dyn Array, split_mask: &BooleanArray) -> Option<SplitScore> {
        let left_arr = filter(arr, split_mask).expect("Cannot filter");
        let right_arr = filter(arr, split_mask).expect("Cannot filter");
        let null_arr =
            filter(arr, &is_null(split_mask).expect("Cannot is_null")).expect("Cannot Filter");
        let (l_gra, l_hes) = self.grad_and_hess(left_arr.as_ref());
        let (r_gra, r_hes) = self.grad_and_hess(right_arr.as_ref());
        let (n_gra, n_hes) = self.grad_and_hess(null_arr.as_ref());
        let score_nl = (l_gra + n_gra).powi(2) / (l_hes + n_hes) + r_gra.powi(2) / r_hes;
        let score_nr = l_gra.powi(2) / l_hes + (r_gra + n_gra).powi(2) / (r_hes + n_hes);
        if score_nl <= score_nr {
            Some(SplitScore::new(score_nl, NullDirection::Left))
        } else {
            Some(SplitScore::new(score_nr, NullDirection::Right))
        }
    }
}

trait StdScore: Copy + Clone {
    fn score(&self, arr: &dyn Array) -> f64;
    fn pred(&self, arr: &dyn Array) -> f64 {
        let boolean_array = arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        boolean_array.true_count() as f64 / boolean_array.len() as f64
    }
    fn weights(&self, split_mask: &BooleanArray) -> (usize, usize, usize, usize) {
        let left_len = split_mask.true_count();
        let null_len = split_mask.null_count();
        let total_len = split_mask.len();
        let right_len = total_len - left_len - null_len;
        return (left_len, right_len, null_len, total_len);
    }
    fn split_score(&self, arr: &dyn Array, split_mask: &BooleanArray) -> Option<SplitScore> {
        if self.score(arr) > 0. {
            let (l_w, r_w, n_w, total_len) = self.weights(split_mask);
            let l_score = self.score(filter(arr, split_mask).unwrap().as_ref());
            let r_score = self.score(filter(arr, &not(split_mask).unwrap()).unwrap().as_ref());
            let score =
                l_w as f64 / total_len as f64 * l_score + r_w as f64 / total_len as f64 * r_score;
            Some(SplitScore::new(score, NullDirection::default()))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Logit {
    pred: f64,
}

impl DiffScore for Logit {
    fn grad_and_hess(&self, arr: &dyn Array) -> (f64, f64) {
        let boolean_array = arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Logit loss is valid only for boolean array");

        let (gradients, hessians) =
            boolean_array
                .iter()
                .fold((0.0, 0.0), |(grad_sum, hess_sum), target| {
                    let target_value = target.map_or(0.0, |b| if b { 1.0 } else { 0.0 });
                    let grad = self.pred - target_value;
                    let hess = self.pred * (1.0 - self.pred);
                    (grad_sum + grad, hess_sum + hess)
                });
        (gradients, hessians)
    }
}

#[derive(Debug, Clone, Copy)]
struct Gini {}

impl StdScore for Gini {
    fn score(&self, arr: &dyn Array) -> f64 {
        let mut class_counts = HashMap::new();
        let boolean_array = arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Gini index is valid only for boolean array");
        for label in boolean_array.values().iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let total = arr.len() as f64;
        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });
        1.0 - sum_of_squares
    }
}

fn diff_score_selector(score_function: &DifferentiableSplitScores, pred: f64) -> impl DiffScore {
    match score_function {
        DifferentiableSplitScores::Logit => Logit { pred },
    }
}
fn wgt_score_selector(score_function: &WeightedSplitScores) -> impl StdScore {
    match score_function {
        WeightedSplitScores::Gini => Gini {},
    }
}
pub fn generate_score_function(score_config: &ScoreConfig) -> (Box<SplitFnType>, Box<PredFnType>) {
    match score_config.score_function {
        SplitScores::Weighted(wgt_score) => {
            let wgt_score = wgt_score_selector(&wgt_score);
            (
                Box::new(move |arr: &dyn Array, split_mask: &BooleanArray| {
                    wgt_score.split_score(arr, split_mask)
                }),
                Box::new(move |arr: &dyn Array| wgt_score.pred(arr)),
            )
        }
        SplitScores::Differentiable(diff_score) => {
            let pred = score_config
                .initial_prediction
                .expect("Differentiable Scores need initial prediction");
            let diff_score = diff_score_selector(&diff_score, pred);
            (
                Box::new(move |arr: &dyn Array, split_mask: &BooleanArray| {
                    diff_score.split_score(arr, split_mask)
                }),
                Box::new(move |arr: &dyn Array| diff_score.pred(arr)),
            )
        }
    }
}

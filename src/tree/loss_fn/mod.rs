pub mod split_values;

use core::cmp::Ordering;
use std::collections::HashMap;

use split_values::{NullDirection, SplitScore};

use super::split::Target;

pub trait Score<T> {
    fn split_score(
        &self,
        target: &impl Target<T>,
        filter_mask: impl Iterator<Item = Option<bool>>,
    ) -> Option<split_values::SplitScore>;
    fn pred(&self, target: &impl Target<T>) -> f64;
}

pub struct Gini;

impl Gini {
    fn gini(counts: &HashMap<bool, usize>, total: f64) -> f64 {
        if total == 0.0 {
            return 0.0;
        }

        let sum_of_squares = counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });

        1.0 - sum_of_squares
    }

    fn impurity(
        &self,
        left_counts: &HashMap<bool, usize>,
        right_counts: &HashMap<bool, usize>,
        null_counts: &HashMap<bool, usize>,
        left_total: f64,
        right_total: f64,
        null_total: f64,
    ) -> split_values::SplitScore {
        let mut left_with_nulls = left_counts.clone();
        for (&label, &count) in null_counts {
            *left_with_nulls.entry(label).or_insert(0) += count;
        }
        let left_gini_with_nulls = Gini::gini(&left_with_nulls, left_total + null_total);

        let mut right_with_nulls = right_counts.clone();
        for (&label, &count) in null_counts {
            *right_with_nulls.entry(label).or_insert(0) += count;
        }
        let right_gini_with_nulls = Gini::gini(&right_with_nulls, right_total + null_total);

        let total = left_total + right_total + null_total;
        let weighted_left_gini = (left_total + null_total) / total * left_gini_with_nulls
            + right_total / total * Gini::gini(right_counts, right_total);
        let weighted_right_gini = left_total / total * Gini::gini(left_counts, left_total)
            + (right_total + null_total) / total * right_gini_with_nulls;

        match weighted_left_gini
            .partial_cmp(&weighted_right_gini)
            .expect("Cannot Compare")
        {
            Ordering::Less | Ordering::Equal => split_values::SplitScore {
                score: weighted_left_gini,
                null_direction: NullDirection::Left,
            },
            _ => split_values::SplitScore {
                score: weighted_right_gini,
                null_direction: NullDirection::Right,
            },
        }
    }
}

impl Score<bool> for Gini {
    fn split_score(
        &self,
        target: &impl Target<bool>,
        filter_mask: impl Iterator<Item = Option<bool>>,
    ) -> Option<split_values::SplitScore> {
        let mut left_counts: HashMap<bool, usize> = HashMap::new();
        let mut right_counts: HashMap<bool, usize> = HashMap::new();
        let mut null_counts: HashMap<bool, usize> = HashMap::new();

        let mut left_total = 0.0;
        let mut right_total = 0.0;
        let mut null_total = 0.0;

        for (val, mask) in target.iter().zip(filter_mask) {
            match mask {
                Some(true) => {
                    *left_counts.entry(val).or_insert(0) += 1;
                    left_total += 1.;
                }
                Some(false) => {
                    *right_counts.entry(val).or_insert(0) += 1;
                    right_total += 1.;
                }
                None => {
                    *null_counts.entry(val).or_insert(0) += 1;
                    null_total += 1.;
                }
            }
        }
        let total_len = left_total + right_total + null_total;
        if total_len == left_total || total_len == right_total || total_len == null_total {
            None
        } else {
            Some(self.impurity(
                &left_counts,
                &right_counts,
                &null_counts,
                left_total,
                right_total,
                null_total,
            ))
        }
    }
    fn pred(&self, target: &impl Target<bool>) -> f64 {
        let (true_cnt, len) = target.iter().fold((0., 0.), |(tc, l), v| {
            let add_cnt = if v { 1. } else { 0. };
            (tc + add_cnt, l + 1.)
        });
        true_cnt / len
    }
}

#[derive(Copy, Clone)]
pub struct Logit {
    pred: f64,
}

impl Logit {
    pub fn new(pred: f64) -> Self {
        match pred > 0. && pred < 1. {
            true => Self { pred },
            false => panic!("Initial prediction for Logit must be gt than 0 a lt than 1"),
        }
    }
    fn grad_and_hes(&self, target: bool) -> (f64, f64) {
        let target_val = if target { 1. } else { 0. };
        let grad = self.pred - target_val;
        let hess = self.pred * (1.0 - self.pred);
        (grad, hess)
    }
}

impl Score<bool> for Logit {
    fn split_score(
        &self,
        target: &impl Target<bool>,
        filter_mask: impl Iterator<Item = Option<bool>>,
    ) -> Option<SplitScore> {
        let mut l_g = 0.;
        let mut l_h = 0.;
        let mut r_g = 0.;
        let mut r_h = 0.;
        let mut n_g = 0.;
        let mut n_h = 0.;
        for (bool_v, m) in target.iter().zip(filter_mask) {
            let (gv, hv) = self.grad_and_hes(bool_v);
            if let Some(true) = m {
                l_g += gv;
                l_h += hv;
            } else if let Some(false) = m {
                r_g += gv;
                r_h += hv;
            } else {
                n_g += gv;
                n_h += hv;
            }
        }
        let score_on_left = (l_g + n_g).powi(2) / (l_h + n_h) + r_g.powi(2) / r_h;
        let score_on_right = (r_g + n_g).powi(2) / (r_h + n_h) + l_g.powi(2) / l_h;
        if score_on_left >= score_on_right {
            Some(SplitScore {
                score: -score_on_left,
                null_direction: NullDirection::Left,
            })
        } else if score_on_right.is_finite() {
            Some(SplitScore {
                score: -score_on_right,
                null_direction: NullDirection::Right,
            })
        } else {
            None
        }
    }
    fn pred(&self, target: &impl Target<bool>) -> f64 {
        let (g, h) = target.iter().fold((0., 0.), |(g, h), v| {
            let (vg, vh) = self.grad_and_hes(v);
            (g + vg, h + vh)
        });
        - g / h
    }
}

pub enum ScoringFunction {
    Logit(Logit),
    Gini(Gini),
}

impl std::fmt::Display for ScoringFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let displayable = match self {
            ScoringFunction::Logit(_) => "Logit",
            ScoringFunction::Gini(_) => "Gini",
        };
        write!(f, "{}", displayable)
    }
}

impl Score<bool> for ScoringFunction {
    fn split_score(
        &self,
        target: &impl Target<bool>,
        filter_mask: impl Iterator<Item = Option<bool>>,
    ) -> Option<SplitScore> {
        match self {
            ScoringFunction::Gini(g) => g.split_score(target, filter_mask),
            ScoringFunction::Logit(l) => l.split_score(target, filter_mask),
        }
    }
    fn pred(&self, target: &impl Target<bool>) -> f64 {
        match self {
            ScoringFunction::Gini(g) => g.pred(target),
            ScoringFunction::Logit(l) => l.pred(target),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_stuff() {
        let init_prd: f64 = 0.5;
        let logit = Logit::new(init_prd);
        let g = init_prd - 1.;
        let h = init_prd.powi(2);
        let (g_res, h_res) = logit.grad_and_hes(true);
        assert_eq!(g, g_res, "Wrong grad for Logit");
        assert_eq!(h, h_res, "Wrong hess for Logit");
    }
}

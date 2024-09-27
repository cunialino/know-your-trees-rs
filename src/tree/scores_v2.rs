use std::collections::HashMap;

pub trait Score {
    type TargetType: Copy;
    fn score(&self, target: &[Self::TargetType]) -> f64;
    fn split_score(&self, target: &[Self::TargetType], filter_mask: &[bool]) -> f64 {
        1.
    }
    fn find_best_split<'a, T>(
        &'a self,
        target: &'a [Self::TargetType],
        feature: &'a [T],
    ) -> impl Iterator<Item = (f64, f64)> + 'a
    where
        T: PartialOrd + Into<f64> + Copy,
    {
        let splits = feature.iter().map(|val| {
            let bool_mask = feature.iter().map(|v2| val < v2);
            (val, bool_mask.collect::<Vec<bool>>())
        });
        splits.map(|(val, mask)| {
            (
                self.split_score(target, mask.as_slice()),
                val.clone().into(),
            )
        })
    }
}

struct Gini {}

impl Score for Gini {
    type TargetType = bool;
    fn score(&self, target: &[Self::TargetType]) -> f64 {
        let mut class_counts = HashMap::new();
        let mut total = 0.;
        for label in target.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
            total += 1.;
        }
        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });
        1.0 - sum_of_squares
    }
}

pub trait DiffScore: Score {
    fn grad_and_hess(&self, target: &[Self::TargetType]) -> (f64, f64);
    fn split_score(&self, target: &[Self::TargetType], split_mask: &[bool]) -> f64 {
        let left_arr: Vec<Self::TargetType> = target
            .iter()
            .zip(split_mask.iter())
            .filter(|(_, &p)| p)
            .map(|(&v, _)| v) // Map the filtered values (which are all true here)
            .collect();
        let right_arr: Vec<Self::TargetType> = target
            .iter()
            .zip(split_mask.iter())
            .filter(|(_, &p)| !p)
            .map(|(&v, _)| v) // Map the filtered values (which are all true here)
            .collect();

        let (l_gra, l_hes) = self.grad_and_hess(left_arr.as_ref());
        let (r_gra, r_hes) = self.grad_and_hess(right_arr.as_ref());
        l_gra.powi(2) / l_hes + r_gra.powi(2) / r_hes
    }
}

struct Logit {
    pred: f64,
}

impl Score for Logit {
    type TargetType = bool;
    fn score(&self, target: &[Self::TargetType]) -> f64 {
        let (g, h) = self.grad_and_hess(target);
        g.powi(2) / h
    }
}

impl DiffScore for Logit {
    fn grad_and_hess(&self, target: &[bool]) -> (f64, f64) {
        let (gradients, hessians) =
            target
                .iter()
                .fold((0.0, 0.0), |(grad_sum, hess_sum), target| {
                    let target_value = if *target { 1.0 } else { 0.0 };
                    let grad = self.pred - target_value;
                    let hess = self.pred * (1.0 - self.pred);
                    (grad_sum + grad, hess_sum + hess)
                });
        (gradients, hessians)
    }
}

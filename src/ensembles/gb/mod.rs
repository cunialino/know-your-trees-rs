use crate::tree::loss_fn::ScoringFunction;
use crate::tree::split::{DataSet, Target};
use crate::tree::Tree;
use crate::tree::TreeConfig;

pub struct GBConfig {
    pub num_boost: usize,
    pub tree_config: TreeConfig,
    pub alpha: f64,
}
pub struct GradientBoosting {
    pub trees: std::vec::Vec<Tree>,
    pub alpha: f64,
}

impl GradientBoosting {
    pub fn fit(samples: &impl DataSet, target: &impl Target<bool>, params: &GBConfig) -> Self {
        let mut trees = std::vec::Vec::with_capacity(params.num_boost);
        let mut initial_preds = vec![0.5; target.len()];
        let mut score_fn =
            ScoringFunction::Logit(crate::tree::loss_fn::Logit::new(initial_preds.as_slice()));
        for round in 0..params.num_boost {
            let tree = Tree::fit(samples, target, &params.tree_config, &score_fn).unwrap();
            // bad bad thing
            let predictions = tree.predict(samples).unwrap();
            initial_preds
                .iter_mut()
                .zip(predictions.iter())
                .for_each(|(ip, v)| *ip += params.alpha * v);
            score_fn =
                ScoringFunction::Logit(crate::tree::loss_fn::Logit::new(initial_preds.as_slice()));
            trees.insert(round, tree);
        }
        GradientBoosting {
            trees,
            alpha: params.alpha,
        }
    }
    pub fn predict(&self, samples: &impl DataSet) -> Vec<f64> {
        self.trees
            .iter()
            .map(|t| t.predict(samples).unwrap())
            .reduce(|acc, p| {
                acc.iter()
                    .zip(p.iter())
                    .map(|(ip, v)| *ip + self.alpha * v)
                    .collect()
            })
            .unwrap()
    }
}

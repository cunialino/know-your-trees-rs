use crate::tree::loss_fn::Score;
use crate::tree::loss_fn::ScoringFunction;
use crate::tree::split::{Target, DataSet};
use crate::tree::Tree;
use crate::tree::TreeConfig;

pub struct GBConfig {
   pub num_boost: usize,
    pub tree_config: TreeConfig,
}
struct GradientBoosting {
    pub trees: std::vec::Vec<Tree>,
}

impl GradientBoosting {
    fn fit<T, S: Score<T>>(samples: &impl DataSet, target: &impl Target<T>, params: GBConfig, score_fn: ScoringFunction) -> Self {
        let mut trees = std::vec::Vec::with_capacity(params.num_boost);
        for round in (0..params.num_boost) {
            let tree = Tree::fit(samples, target, &params.tree_config, &score_fn);
        };
    }
}

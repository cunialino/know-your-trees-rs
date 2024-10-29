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
                .zip(predictions)
                .for_each(|(ip, v)| *ip += params.alpha * v.unwrap());
            score_fn =
                ScoringFunction::Logit(crate::tree::loss_fn::Logit::new(initial_preds.as_slice()));
            trees.insert(round, tree);
        }
        GradientBoosting {
            trees,
            alpha: params.alpha,
        }
    }
    pub fn predict<'a>(&'a self, samples: &'a impl DataSet) -> impl Iterator<Item = f64> + 'a {
        self.trees.iter().flat_map(|tree| {
            match tree.predict(samples) {
                Ok(iter) => iter.map(|res| res.unwrap_or(0.0)), // Replace `0.0` with a fallback value if an error occurs
                Err(_) => panic!("tee"),
            }
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{loss_fn, TreeConfig};
    use crate::tree::loss_fn::ScoringFunction;

    #[test]
    fn test_gb_config_initialization() {
        let tree_config = TreeConfig { max_depth: 3 };
        let gb_config = GBConfig {
            num_boost: 10,
            tree_config,
            alpha: 0.1,
        };

        assert_eq!(gb_config.num_boost, 10, "Expected num_boost to be 10");
        assert_eq!(gb_config.alpha, 0.1, "Expected alpha to be 0.1");
        assert_eq!(gb_config.tree_config.max_depth, 3, "Expected max_depth to be 3");
    }

    #[test]
    fn test_gradient_boosting_fit() {
        let data = std::collections::HashMap::from([("F1".to_string(), vec![1., 2., 3.])]);
        let target = vec![true, false, false];
        let _score_fn = ScoringFunction::Logit(loss_fn::Logit::new(&[0.5, 0.5, 0.5]));
        let tree_config = TreeConfig { max_depth: 3 };
        let gb_config = GBConfig {
            num_boost: 5,
            tree_config,
            alpha: 0.1,
        };

        let model = GradientBoosting::fit(&data, &target, &gb_config);

        assert_eq!(model.trees.len(), 5, "Expected 5 trees in the gradient boosting model");
        assert_eq!(model.alpha, 0.1, "Expected alpha to be 0.1 in the model");
    }
}

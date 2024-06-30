use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::compute::{filter, not};
use arrow::datatypes::{ArrowNativeTypeOp, ArrowNumericType, ArrowPrimitiveType, Float64Type};

#[derive(Debug)]
struct Split {
    target: ArrayRef,
    data: ArrayRef,
}

impl Split {
    fn gini(&self, target: &BooleanArray) -> f64 {
        let mut class_counts = std::collections::HashMap::new();
        let target_ref = target;
        for label in target_ref.iter() {
            *class_counts.entry(format!("{:?}", label)).or_insert(0) += 1;
        }
        let total = self.target.len() as f64;

        let sum_of_squares = class_counts.values().fold(0.0, |acc, &count| {
            let proportion = count as f64 / total;
            acc + proportion * proportion
        });

        1.0 - sum_of_squares
    }

    fn best_split<D>(&self) -> Option<(BooleanArray, <D as ArrowPrimitiveType>::Native)>
    where
        D: ArrowNumericType,
    {
        if self.gini(self.target.as_any().downcast_ref().unwrap()) == 0. {
            return None;
        }
        let data_ref = self
            .data
            .as_any()
            .downcast_ref::<PrimitiveArray<D>>()
            .unwrap();
        let mut best_gini_index = f64::MAX;
        let mut best_mask = None;
        let mut threshold = <D as ArrowPrimitiveType>::Native::MAX_TOTAL_ORDER;
        for split_point in data_ref.values() {
            let filter_mask = BooleanArray::from(
                data_ref
                    .values()
                    .into_iter()
                    .map(|&i| i < *split_point)
                    .collect::<Vec<bool>>(),
            );

            let left_len = filter_mask.values().iter().filter(|&value| value).count();
            let total_len = filter_mask.len();
            let right_len = filter_mask.len() - left_len;

            let left_wgt = left_len as f64 / total_len as f64;
            let right_wgt = right_len as f64 / total_len as f64;

            let left_gini = left_wgt
                * self.gini(
                    filter(&self.target, &filter_mask)
                        .unwrap()
                        .as_any()
                        .downcast_ref()
                        .unwrap(),
                );
            let right_gini = right_wgt
                * self.gini(
                    filter(&self.target, &filter_mask)
                        .unwrap()
                        .as_any()
                        .downcast_ref()
                        .unwrap(),
                );

            let weighted_gini = left_gini + right_gini;

            if weighted_gini < best_gini_index {
                best_gini_index = weighted_gini;
                best_mask = Some(filter_mask);
                threshold = *split_point;
            }
        }

        best_mask.and_then(|l| Some((l, threshold)))
    }
}

#[derive(Debug)]
pub struct Tree {
    pub feature_index: Option<usize>,
    pub threshold: Option<f64>,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
    pub prediction: Option<bool>, // Optional: only used at leaf nodes
}

impl Tree {
    pub fn new(feature_index: usize, threshold: f64) -> Self {
        Tree {
            feature_index: Some(feature_index),
            threshold: Some(threshold),
            left: None,
            right: None,
            prediction: None,
        }
    }
    pub fn build_tree(samples: ArrayRef, target: ArrayRef, max_depth: usize) -> Option<Box<Tree>> {
        if max_depth == 0 || samples.len() == 0 {
            return None;
        }
        let split = Split {
            data: samples.clone(),
            target: target.clone(),
        };
        if let Some((data_mask, th)) = split.best_split::<Float64Type>() {
            let tree = Tree {
                feature_index: Some(0),
                threshold: Some(th),
                left: Self::build_tree(
                    filter(&samples, &data_mask).unwrap(),
                    filter(&target, &data_mask).unwrap(),
                    max_depth - 1,
                ),
                right: Self::build_tree(
                    filter(&samples, &not(&data_mask).unwrap()).unwrap(),
                    filter(&target, &not(&data_mask).unwrap()).unwrap(),
                    max_depth - 1,
                ),
                prediction: None,
            };
            return Some(Box::new(tree));
        }
        None
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow::array::Float64Array;
    use std::sync::Arc;

    #[test]
    fn gini() {
        let node = Split {
            data: Arc::new(Float64Array::from(vec![0.0])), // Not used for Gini index
            target: Arc::new(BooleanArray::from(vec![true, true, false, false, true])), // Example labels
        };

        // Calculate Gini index
        let gini = node.gini(node.target.as_any().downcast_ref().unwrap());
        let expected_gini = 1.0 - (3.0 / 5.0 * 3.0 / 5.0 + 2.0 / 5.0 * 2.0 / 5.0); // 0.48
        let delta = 0.001;

        assert!(
            (gini - expected_gini).abs() < delta,
            "Calculated Gini index is incorrect: expected {:.3}, got {:.3}",
            expected_gini,
            gini
        );
    }
    #[test]
    fn test_best_split() {
        let nn = Split {
            target: Arc::new(BooleanArray::from(vec![true, false])),
            data: Arc::new(Float64Array::from(vec![1.0, 2.0])),
        };

        // Since best_split may return None, handle it properly
        let (filter_mask, threshold) = nn.best_split::<Float64Type>().expect("No split found");

        // Verify the splits' contents are as expected
        assert_eq!(filter_mask, BooleanArray::from(vec![true, false]));
        assert_eq!(threshold, 2., "Wrong threshold");
    }

    #[test]
    fn test_tree() {
        let target = Arc::new(BooleanArray::from(vec![true, false, false]));
        let samples = Arc::new(Float64Array::from(vec![1.0, 2.0, 1.5]));
        let tree = Tree::build_tree(samples, target, 2);
        dbg!(tree);
    }
}

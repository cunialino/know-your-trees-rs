fn create_tree<'a, FeatureType, DataSetType, K, V>(df: DataSetType)
where
    FeatureType: PartialOrd + 'a,
    DataSetType: Iterator<Item = (K, V)>,
    K: AsRef<str> + 'a,
    V: IntoIterator<Item = &'a FeatureType>,
{
    todo!("Still needs implementation");
}

#[test]
fn test_ct() {
    let mut map: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    map.insert("a".to_string(), vec![1.0, 2.0, 3.0]);
    map.insert("b".to_string(), vec![4.0, 5.0]);
    create_tree(map.iter());

    let mut bu: std::vec::Vec<std::vec::Vec<f32>> = std::vec::Vec::new();

    bu.push(vec![0.1, 2.0, 3.0]);
    create_tree(bu.iter().enumerate().map(|(a, b)| (a.to_string(), b)));

}

use kyt::tree::scores::{ScoreConfig, SplitScores, WeightedSplitScores};
use kyt::tree::tree::{Tree, TreeConfig};

use arrow::array::ArrayRef;
use arrow::array::{BooleanArray, StringArray};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::Result;
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::sync::Arc;
fn load_and_modify_csv() -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("Id", DataType::UInt8, false),
        Field::new("sepal_length", DataType::Float64, false),
        Field::new("sepal_width", DataType::Float64, false),
        Field::new("petal_length", DataType::Float64, false),
        Field::new("petal_width", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false), // the column to modify
    ]));
    let file = File::open("data/Iris.csv").unwrap();

    // create a builder
    let mut reader = ReaderBuilder::new(schema.clone()).build(file).unwrap();

    // Read the first batch (in practice, you'd handle multiple batches if needed)
    let maybe_batch = reader.next().unwrap();
    let record_batch = maybe_batch?;

    // Access the 'species' column, assumed to be the last column
    let species_col = record_batch
        .column_by_name("species")
        .expect("Cannot find species column")
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Create a new Boolean array by applying the transformation
    let bool_col: BooleanArray = species_col
        .iter()
        .map(|v| v == Some("Iris-setosa"))
        .collect::<Vec<bool>>()
        .into();
    let new_schema = Arc::new(Schema::new(vec![
        Field::new("Id", DataType::UInt8, false),
        Field::new("sepal_length", DataType::Float64, false),
        Field::new("sepal_width", DataType::Float64, false),
        Field::new("petal_length", DataType::Float64, false),
        Field::new("petal_width", DataType::Float64, false),
        Field::new("species", DataType::Boolean, false), // the column to modify
    ]));

    // Create a new RecordBatch with the modified column (replacing 'species')
    let mut columns: Vec<ArrayRef> = record_batch.columns().to_vec();
    columns[5] = Arc::new(bool_col);

    let modified_batch = RecordBatch::try_new(new_schema.clone(), columns)?;
    Ok(modified_batch)
}

fn main() {
    // Load and prepare the data
    let mut batch = load_and_modify_csv().expect("Cannot read record batch");
    let target = batch.remove_column(5);
    batch.remove_column(0);

    // Configure the tree
    let tree_config = TreeConfig { max_depth: 5 };
    let score_config = ScoreConfig {
        score_function: SplitScores::Weighted(WeightedSplitScores::Gini),
        initial_prediction: Some(0.1),
    };

    // Fit the tree (this is what we want to profile)
    let _tree = Tree::fit(batch, target.as_ref(), &tree_config, &score_config);
}

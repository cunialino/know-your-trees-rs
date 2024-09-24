use arrow::{
    array::{Array, AsArray},
    datatypes::{DataType, Float32Type, Float64Type, Int32Type},
};

pub trait ArrayConversions {
    fn try_into_iter_f64(&self) -> Box<dyn Iterator<Item = Option<f64>> + '_>;
}

impl ArrayConversions for dyn Array + '_ {
    fn try_into_iter_f64(&self) -> Box<dyn Iterator<Item = Option<f64>> + '_> {
        match self.data_type() {
            DataType::Float32 => Box::new(
                self.as_primitive_opt::<Float32Type>()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(|f| f.into())),
            ),
            DataType::Float64 => Box::new(
                self.as_primitive_opt::<Float64Type>()
                    .unwrap()
                    .iter()
                    .map(|v| v),
            ),
            DataType::Int32 => Box::new(
                self.as_primitive_opt::<Int32Type>()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(|f| f.into())),
            ),
            _ => panic!("Invalid data type"),
        }
    }
}

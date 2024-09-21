use arrow::{
    array::{Array, AsArray},
    datatypes::{DataType, Float32Type, Float64Type, Int32Type},
};

pub trait ArrayConversions {
    fn try_into_iter_f64(&self) -> Option<Box<dyn Iterator<Item = f64> + '_>>;
}

impl ArrayConversions for dyn Array + '_ {
   fn try_into_iter_f64(&self) -> Option<Box<dyn Iterator<Item = f64> + '_>> {
        match self.data_type() {
            DataType::Float32 => Some(Box::new(
                self.as_primitive_opt::<Float32Type>()?
                    .iter()
                    .map(|v| v.unwrap_or(f32::NAN).into()),
            )),
            DataType::Float64 => Some(Box::new(
                self.as_primitive_opt::<Float64Type>()?
                    .iter()
                    .map(|v| v.unwrap_or(f64::NAN)),
            )),
            DataType::Int32 => Some(Box::new(
                self.as_primitive_opt::<Int32Type>()?
                    .iter()
                    .map(|v| v.unwrap().into()),
            )),
            _ => panic!("Invalid data type"),
        }
    }
}

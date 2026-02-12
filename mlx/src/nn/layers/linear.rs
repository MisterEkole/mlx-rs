// mlx/src/nn/layers/linear.rs

pub struct Linear {
    weight: Array,
    bias: Option<Array>,
}
//TODO: Add a method to initialize weights and bias (e.g., random normal for weights, zeros for bias)
impl Linear {
    pub fn new(input_dims: usize, output_dims: usize, has_bias: bool) -> crate::Result<Self> {
        let weight = Array::random_normal(&[output_dims, input_dims])?; 
        
        let bias = if has_bias {
            Some(Array::full(&[output_dims as i32], 0.0, crate::Dtype::Float32)?)
        } else {
            None
        };

        Ok(Self { weight, bias })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Array) -> crate::Result<Array> {
      
        let mut out = x.matmul(&self.weight.transpose_default()?)?;
        
        if let Some(ref b) = self.bias {
            out = out.add(b)?;
        }
        
        Ok(out)
    }
}
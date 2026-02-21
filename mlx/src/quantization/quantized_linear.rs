use crate::{Array, Result};
use crate::nn::{Module, ModuleParams};
use crate::tree::TreeFlatten;
use super::q_ops;

pub struct QuantizedLinear{
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Option<Array>,

    pub bits: i32,
    pub group_size: i32,
}

impl QuantizedLinear{
    pub fn new(w: Array, s: Array, b: Array, bias: Option<Array>, bits: i32, group_size: i32) ->Self{
        Self{
            weight: w,
            scales: s,
            biases: b,
            bias,
            bits,
            group_size,
        }
    }
    
    pub fn from_linear(linear:&crate::nn::Linear, bits: i32, group_size: i32) -> Result<Self>{
        let (qw, s, b)= q_ops::quantize(&linear.weight, bits, group_size)?;
        Ok(Self::new(qw, s, b, linear.bias.clone(), bits, group_size))
    }
}

impl TreeFlatten for QuantizedLinear {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = vec![
            self.weight.clone(), 
            self.scales.clone(), 
            self.biases.clone()
        ];
        
        if let Some(b) = &self.bias {
            flat.push(b.clone());
        }
        
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.weight = flat_arrays.next().unwrap().clone();
        self.scales = flat_arrays.next().unwrap().clone();
        self.biases = flat_arrays.next().unwrap().clone();
        
        if self.bias.is_some() {
            self.bias = Some(flat_arrays.next().unwrap().clone());
        }
    }
}

impl Module for QuantizedLinear{
    fn forward(&self, x: &Array) -> Result<Array> {
        let weight = q_ops::dequantize(&self.weight, &self.scales, &self.biases, self.bits, self.group_size)?;
        let weight_t = weight.transpose(&[])?;
        let mut out = x.matmul(&weight_t)?;

        if let Some(ref b)= self.bias{
            out = out.add(b)?;
        }
        Ok(out)
    }
}

impl ModuleParams for QuantizedLinear {
    fn parameters(&self) -> Vec<&Array>{
        let mut params = vec![&self.weight, &self.scales, &self.biases];
        if let Some(ref b) = self.bias{
            params.push(b);
        }
        params
    }
}
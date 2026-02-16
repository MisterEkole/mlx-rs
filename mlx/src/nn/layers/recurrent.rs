use crate::{Array, Result, Dtype};
use crate::nn::Module;
use crate::nn::layers::activations::{sigmoid, tanh};

// --- Helper for Weight Initialization ---
// Uniform initialization bounded by 1 / sqrt(hidden_size)
fn rnn_uniform(shape: &[usize], hidden_size: usize, key: &Array) -> Result<Array> {
    let bound = (1.0 / hidden_size as f32).sqrt();
    Array::random_uniform(shape, -bound, bound, Dtype::Float32, key)
}

// =========================================================================
// LSTM Layer
// =========================================================================

pub struct LSTM {
    pub weight_ih: Array,
    pub weight_hh: Array,
    pub bias_ih: Option<Array>,
    pub bias_hh: Option<Array>,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, key: &Array) -> Result<Self> {
        let (w_key, _b_key) = key.split()?;
        let four_h = 4 * hidden_size;

        // random_uniform expects &[usize]
        let weight_ih = rnn_uniform(&[four_h, input_size], hidden_size, &w_key)?;
        let weight_hh = rnn_uniform(&[four_h, hidden_size], hidden_size, &w_key)?;

        // full expects &[i32]
        let (bias_ih, bias_hh) = if bias {
            (
                Some(Array::full(&[four_h as i32], 0.0, Dtype::Float32)?),
                Some(Array::full(&[four_h as i32], 0.0, Dtype::Float32)?)
            )
        } else {
            (None, None)
        };

        Ok(Self { weight_ih, weight_hh, bias_ih, bias_hh })
    }

    pub fn step(&self, x: &Array, state: (Array, Array)) -> Result<(Array, Array)> {
        let (h_prev, c_prev) = state;
        
        // h_prev.shape() returns usize, cast for math
        let h_dim = h_prev.shape()?[1] as i32;
        let batch_size = h_prev.shape()?[0] as i32;

        let mut x_proj = x.matmul(&self.weight_ih.transpose(&[1, 0])?)?;
        let mut h_proj = h_prev.matmul(&self.weight_hh.transpose(&[1, 0])?)?;

        if let Some(ref b) = self.bias_ih { x_proj = x_proj.add(b)?; }
        if let Some(ref b) = self.bias_hh { h_proj = h_proj.add(b)?; }

        let gates = x_proj.add(&h_proj)?;

        // slice expects &[i32]
        let i = gates.slice(&[0, 0], &[batch_size, h_dim], &[1,1])?;
        let f = gates.slice(&[0, h_dim], &[batch_size, 2 * h_dim], &[1,1])?;
        let g = gates.slice(&[0, 2 * h_dim], &[batch_size, 3 * h_dim], &[1,1])?;
        let o = gates.slice(&[0, 3 * h_dim], &[batch_size, 4 * h_dim], &[1,1])?;

        let i_gate = sigmoid(&i)?;
        let f_gate = sigmoid(&f)?;
        let g_gate = tanh(&g)?;
        let o_gate = sigmoid(&o)?;

        let c_next = f_gate.multiply(&c_prev)?.add(&i_gate.multiply(&g_gate)?)?;
        let h_next = o_gate.multiply(&tanh(&c_next)?)?;

        Ok((h_next, c_next))
    }
}

impl Module for LSTM {
    fn forward(&self, x: &Array) -> Result<Array> {
        let batch_size = x.shape()?[0] as i32;
        let seq_len = x.shape()?[1] as usize; // Loop variable, keep as usize
        let hidden_dim = self.weight_hh.shape()?[1] as i32;

        // zeros expects &[i32]
        let mut h = Array::zeros(&[batch_size, hidden_dim], x.dtype())?;
        let mut c = Array::zeros(&[batch_size, hidden_dim], x.dtype())?;

        for t in 0..seq_len {
            // slice_axis expects i32
            let x_t = x.slice_axis(1, t as i32, (t + 1) as i32)?.squeeze(Some(&[1]))?;
            let (h_next, c_next) = self.step(&x_t, (h, c))?;
            h = h_next;
            c = c_next;
        }

        Ok(h)
    }

    fn parameters(&self) -> Vec<&Array> {
        let mut p = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih { p.push(b); }
        if let Some(ref b) = self.bias_hh { p.push(b); }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        let mut p = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut b) = self.bias_ih { p.push(b); }
        if let Some(ref mut b) = self.bias_hh { p.push(b); }
        p
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        if new_params.len() >= 2 {
            self.weight_ih = new_params[0].clone();
            self.weight_hh = new_params[1].clone();
            let mut offset = 2;
            if self.bias_ih.is_some() && new_params.len() > offset {
                self.bias_ih = Some(new_params[offset].clone());
                offset += 1;
            }
            if self.bias_hh.is_some() && new_params.len() > offset {
                self.bias_hh = Some(new_params[offset].clone());
            }
        }
    }
}

// =========================================================================
// GRU Layer
// =========================================================================

pub struct GRU {
    pub weight_ih: Array,
    pub weight_hh: Array,
    pub bias_ih: Option<Array>,
    pub bias_hh: Option<Array>,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, key: &Array) -> Result<Self> {
        let (w_key, _b_key) = key.split()?;
        let three_h = 3 * hidden_size;

        let weight_ih = rnn_uniform(&[three_h, input_size], hidden_size, &w_key)?;
        let weight_hh = rnn_uniform(&[three_h, hidden_size], hidden_size, &w_key)?;

        let (bias_ih, bias_hh) = if bias {
            (
                Some(Array::full(&[three_h as i32], 0.0, Dtype::Float32)?),
                Some(Array::full(&[three_h as i32], 0.0, Dtype::Float32)?)
            )
        } else {
            (None, None)
        };

        Ok(Self { weight_ih, weight_hh, bias_ih, bias_hh })
    }

    pub fn step(&self, x: &Array, h_prev: &Array) -> Result<Array> {
        let h_dim = h_prev.shape()?[1] as i32;
        let batch_size = h_prev.shape()?[0] as i32;

        let mut x_proj = x.matmul(&self.weight_ih.transpose(&[1, 0])?)?;
        let mut h_proj = h_prev.matmul(&self.weight_hh.transpose(&[1, 0])?)?;

        if let Some(ref b) = self.bias_ih { x_proj = x_proj.add(b)?; }
        if let Some(ref b) = self.bias_hh { h_proj = h_proj.add(b)?; }

        // Slices for Reset (r) and Update (z) gates
        let x_r = x_proj.slice(&[0, 0], &[batch_size, h_dim],&[1,1])?;
        let h_r = h_proj.slice(&[0, 0], &[batch_size, h_dim], &[1,1])?;
        
        let x_z = x_proj.slice(&[0, h_dim], &[batch_size, 2 * h_dim], &[1,1])?;
        let h_z = h_proj.slice(&[0, h_dim], &[batch_size, 2 * h_dim], &[1,1])?;

        // Slices for New candidate (n)
        let x_n = x_proj.slice(&[0, 2 * h_dim], &[batch_size, 3 * h_dim], &[1,1])?;
        let h_n = h_proj.slice(&[0, 2 * h_dim], &[batch_size, 3 * h_dim], &[1,1])?;

        let r = sigmoid(&x_r.add(&h_r)?)?;
        let z = sigmoid(&x_z.add(&h_z)?)?;
        let n = tanh(&x_n.add(&r.multiply(&h_n)?)?)?;

        let one = Array::full(&[], 1.0, z.dtype())?;
        let h_next = one.subtract(&z)?.multiply(&n)?.add(&z.multiply(h_prev)?)?;

        Ok(h_next)
    }
}

impl Module for GRU {
    fn forward(&self, x: &Array) -> Result<Array> {
        let batch_size = x.shape()?[0] as i32;
        let seq_len = x.shape()?[1] as usize;
        let hidden_dim = self.weight_hh.shape()?[1] as i32;

        let mut h = Array::zeros(&[batch_size, hidden_dim], x.dtype())?;

        for t in 0..seq_len {
            
            let x_t = x.slice_axis(1, t as i32, (t + 1) as i32)?.squeeze(Some(&[1]))?;
            h = self.step(&x_t, &h)?;
        }

        Ok(h)
    }

    fn parameters(&self) -> Vec<&Array> {
        let mut p = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih { p.push(b); }
        if let Some(ref b) = self.bias_hh { p.push(b); }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        let mut p = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut b) = self.bias_ih { p.push(b); }
        if let Some(ref mut b) = self.bias_hh { p.push(b); }
        p
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        if new_params.len() >= 2 {
            self.weight_ih = new_params[0].clone();
            self.weight_hh = new_params[1].clone();
            let mut offset = 2;
            if self.bias_ih.is_some() && new_params.len() > offset {
                self.bias_ih = Some(new_params[offset].clone());
                offset += 1;
            }
            if self.bias_hh.is_some() && new_params.len() > offset {
                self.bias_hh = Some(new_params[offset].clone());
            }
        }
    }
}

// =========================================================================
// Vanilla RNN Layer
// =========================================================================

pub struct RNN {
    pub weight_ih: Array,
    pub weight_hh: Array,
    pub bias_ih: Option<Array>,
    pub bias_hh: Option<Array>,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, key: &Array) -> Result<Self> {
        let (w_key, _b_key) = key.split()?;

        let weight_ih = rnn_uniform(&[hidden_size, input_size], hidden_size, &w_key)?;
        let weight_hh = rnn_uniform(&[hidden_size, hidden_size], hidden_size, &w_key)?;

        let (bias_ih, bias_hh) = if bias {
            (
                Some(Array::full(&[hidden_size as i32], 0.0, Dtype::Float32)?),
                Some(Array::full(&[hidden_size as i32], 0.0, Dtype::Float32)?)
            )
        } else {
            (None, None)
        };

        Ok(Self { weight_ih, weight_hh, bias_ih, bias_hh })
    }

    pub fn step(&self, x: &Array, h_prev: &Array) -> Result<Array> {
        let mut x_proj = x.matmul(&self.weight_ih.transpose(&[1, 0])?)?;
        let mut h_proj = h_prev.matmul(&self.weight_hh.transpose(&[1, 0])?)?;

        if let Some(ref b) = self.bias_ih { x_proj = x_proj.add(b)?; }
        if let Some(ref b) = self.bias_hh { h_proj = h_proj.add(b)?; }

        tanh(&x_proj.add(&h_proj)?)
    }
}

impl Module for RNN {
    fn forward(&self, x: &Array) -> Result<Array> {
        let batch_size = x.shape()?[0] as i32;
        let seq_len = x.shape()?[1] as usize;
        let hidden_dim = self.weight_hh.shape()?[1] as i32;

        let mut h = Array::zeros(&[batch_size, hidden_dim], x.dtype())?;

        for t in 0..seq_len {
            let x_t = x.slice_axis(1, t as i32, (t + 1) as i32)?.squeeze(Some(&[1]))?;
            h = self.step(&x_t, &h)?;
        }

        Ok(h)
    }

    fn parameters(&self) -> Vec<&Array> {
        let mut p = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih { p.push(b); }
        if let Some(ref b) = self.bias_hh { p.push(b); }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        let mut p = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut b) = self.bias_ih { p.push(b); }
        if let Some(ref mut b) = self.bias_hh { p.push(b); }
        p
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        if new_params.len() >= 2 {
            self.weight_ih = new_params[0].clone();
            self.weight_hh = new_params[1].clone();
            let mut offset = 2;
            if self.bias_ih.is_some() && new_params.len() > offset {
                self.bias_ih = Some(new_params[offset].clone());
                offset += 1;
            }
            if self.bias_hh.is_some() && new_params.len() > offset {
                self.bias_hh = Some(new_params[offset].clone());
            }
        }
    }
}
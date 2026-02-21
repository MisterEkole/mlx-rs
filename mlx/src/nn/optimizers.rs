use crate::{Array, Result};
use crate::tree::TreeFlatten; 

pub trait Optimizer {
    /// Update the parameters using the provided gradients.
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()>;

    /// Returns internal state handles (m, v, velocity, etc.) for batch evaluation.
    /// Defaults to empty for state-less optimizers like basic SGD.
    fn get_state_handles(&self) -> Vec<Array> {
        Vec::new()
    }
}

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}


impl TreeFlatten for SGD {
    fn flatten_state(&self) -> Vec<Array> { Vec::new() }
    fn unflatten_state(&mut self, _flat_arrays: &mut std::slice::Iter<'_, Array>) {}
}

impl Optimizer for SGD {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        for (p, g) in params.into_iter().zip(grads.into_iter()) {
            // New Value = Old Value - (Learning Rate * Gradient)
            let step = g.multiply_scalar(self.learning_rate)?;
            let new_p = p.subtract(&step)?;
            *p = new_p; // Update the actual array handle in the layer
        }
        Ok(())
    }
}


/// SGD with momentum
pub struct SGDMomentum {
    pub lr: f32,
    pub momentum: f32,
    velocity: std::cell::RefCell<Vec<Array>>,
}

impl SGDMomentum {
    pub fn new(lr: f32, momentum: f32, params: &[Array]) -> Result<Self> {
        let mut v = Vec::with_capacity(params.len());
        for p in params {
            // Initialize velocity with zeros matching param shape
            v.push(Array::zeros_like(p)?);
        }
        Ok(Self {
            lr,
            momentum,
            velocity: std::cell::RefCell::new(v),
        })
    }
}


impl TreeFlatten for SGDMomentum {
    fn flatten_state(&self) -> Vec<Array> {
        self.velocity.borrow().clone()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut velocities = self.velocity.borrow_mut();
        for v in velocities.iter_mut() {
            *v = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for SGDMomentum {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let mut velocities = self.velocity.borrow_mut();
        
        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            // v = momentum * v + grad
            let v_prev = velocities[i].multiply_scalar(self.momentum)?;
            let v_new = v_prev.add(&g)?;
            
            // θ = θ - lr * v
            let update = v_new.multiply_scalar(self.lr)?;
            *p = p.subtract(&update)?;
            
            // Store new velocity
            velocities[i] = v_new;
        }
        Ok(())
    }
}



// Adam Optimizer
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: std::cell::RefCell<Vec<Array>>,
    v: std::cell::RefCell<Vec<Array>>,
    count: std::cell::Cell<i32>,
}

impl Adam {
    pub fn new(lr: f32, params: &[Array]) -> Result<Self> {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in params {
            m.push(Array::zeros_like(p)?);
            v.push(Array::zeros_like(p)?);
        }
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: std::cell::RefCell::new(m),
            v: std::cell::RefCell::new(v),
            count: std::cell::Cell::new(0),
        })
    }

}


impl TreeFlatten for Adam {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = Vec::new();
        let ms = self.m.borrow();
        let vs = self.v.borrow();
        for i in 0..ms.len() {
            flat.push(ms[i].clone());
            flat.push(vs[i].clone());
        }
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut ms = self.m.borrow_mut();
        let mut vs = self.v.borrow_mut();
        for i in 0..ms.len() {
            ms[i] = flat_arrays.next().unwrap().clone();
            vs[i] = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let t = self.count.get() + 1;
        self.count.set(t);
        
        let mut ms = self.m.borrow_mut();
        let mut vs = self.v.borrow_mut();

        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            // 1. m = beta1 * m + (1 - beta1) * g
            let m_term1 = ms[i].multiply_scalar(self.beta1)?;
            let m_term2 = g.multiply_scalar(1.0 - self.beta1)?;
            let m_new = m_term1.add(&m_term2)?;

            // 2. v = beta2 * v + (1 - beta2) * g^2
            let v_term1 = vs[i].multiply_scalar(self.beta2)?;
            let g2 = g.multiply(&g)?; // Element-wise square
            let v_term2 = g2.multiply_scalar(1.0 - self.beta2)?;
            let v_new = v_term1.add(&v_term2)?;

            // 3. Bias correction
            let alpha_t = self.lr * (1.0 - self.beta2.powi(t)).sqrt() / (1.0 - self.beta1.powi(t));

            // 4. p = p - alpha_t * m_new / (sqrt(v_new) + eps)
            let denom = v_new.sqrt()?.add_scalar(self.eps)?;
            let step = m_new.divide(&denom)?.multiply_scalar(alpha_t)?;
            
            *p = p.subtract(&step)?;

            // Update state
            ms[i] = m_new;
            vs[i] = v_new;
        }
        Ok(())
    }
 
}


/// AdamW Optimizer (Adam with decoupled Weight Decay)
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: std::cell::RefCell<Vec<Array>>,
    v: std::cell::RefCell<Vec<Array>>,
    count: std::cell::Cell<i32>,
}

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32, params: &[Array]) -> Result<Self> {
        let m = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        let v = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            m: std::cell::RefCell::new(m),
            v: std::cell::RefCell::new(v),
            count: std::cell::Cell::new(0),
        })
    }
}


impl TreeFlatten for AdamW {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = Vec::new();
        let ms = self.m.borrow();
        let vs = self.v.borrow();
        for i in 0..ms.len() {
            flat.push(ms[i].clone());
            flat.push(vs[i].clone());
        }
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut ms = self.m.borrow_mut();
        let mut vs = self.v.borrow_mut();
        for i in 0..ms.len() {
            ms[i] = flat_arrays.next().unwrap().clone();
            vs[i] = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for AdamW {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let t = self.count.get() + 1;
        self.count.set(t);
        let mut ms = self.m.borrow_mut();
        let mut vs = self.v.borrow_mut();

        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            // 1. Decoupled Weight Decay: p = p * (1 - lr * wd)
            let decay = 1.0 - (self.lr * self.weight_decay);
            let p_decayed = p.multiply_scalar(decay)?;

            // 2. Standard Adam updates on ms[i] and vs[i]
            let m_new = ms[i].multiply_scalar(self.beta1)?.add(&g.multiply_scalar(1.0 - self.beta1)?)?;
            let v_new = vs[i].multiply_scalar(self.beta2)?.add(&g.multiply(&g)?.multiply_scalar(1.0 - self.beta2)?)?;

            // 3. Bias correction and Step
            let alpha_t = self.lr * (1.0 - self.beta2.powi(t)).sqrt() / (1.0 - self.beta1.powi(t));
            let denom = v_new.sqrt()?.add_scalar(self.eps)?;
            let step = m_new.divide(&denom)?.multiply_scalar(alpha_t)?;
            
            *p = p_decayed.subtract(&step)?;
            ms[i] = m_new;
            vs[i] = v_new;
        }
        Ok(())
    }
}

/// AdaGrad Optimizer
pub struct AdaGrad {
    pub lr: f32,
    pub eps: f32,
    sum_sq_grad: std::cell::RefCell<Vec<Array>>,
}

impl AdaGrad {
    pub fn new(lr: f32, params: &[Array]) -> Result<Self> {
        let s = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        Ok(Self { lr, eps: 1e-10, sum_sq_grad: std::cell::RefCell::new(s) })
    }
}


impl TreeFlatten for AdaGrad {
    fn flatten_state(&self) -> Vec<Array> {
        self.sum_sq_grad.borrow().clone()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut ss = self.sum_sq_grad.borrow_mut();
        for arr in ss.iter_mut() {
            *arr = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let mut ss = self.sum_sq_grad.borrow_mut();
        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            let s_new = ss[i].add(&g.multiply(&g)?)?;
            let denom = s_new.sqrt()?.add_scalar(self.eps)?;
            *p = p.subtract(&g.multiply_scalar(self.lr)?.divide(&denom)?)?;
            ss[i] = s_new;
        }
        Ok(())
    }
}

/// RMSprop Optimizer
pub struct RMSprop {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    v: std::cell::RefCell<Vec<Array>>,
}

impl RMSprop {
    pub fn new(lr: f32, params: &[Array]) -> Result<Self> {
        let v = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        Ok(Self { lr, alpha: 0.99, eps: 1e-8, v: std::cell::RefCell::new(v) })
    }
}


impl TreeFlatten for RMSprop {
    fn flatten_state(&self) -> Vec<Array> {
        self.v.borrow().clone()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut vs = self.v.borrow_mut();
        for arr in vs.iter_mut() {
            *arr = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let mut vs = self.v.borrow_mut();
        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            let v_new = vs[i].multiply_scalar(self.alpha)?.add(&g.multiply(&g)?.multiply_scalar(1.0 - self.alpha)?)?;
            let denom = v_new.sqrt()?.add_scalar(self.eps)?;
            *p = p.subtract(&g.multiply_scalar(self.lr)?.divide(&denom)?)?;
            vs[i] = v_new;
        }
        Ok(())
    }
}

/// Lion Optimizer (EvoLved Sign Momentum)
pub struct Lion {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    exp_avg: std::cell::RefCell<Vec<Array>>,
}

impl Lion {
    pub fn new(lr: f32, params: &[Array]) -> Result<Self> {
        let m = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        Ok(Self { lr, beta1: 0.9, beta2: 0.99, weight_decay: 0.0, exp_avg: std::cell::RefCell::new(m) })
    }
}


impl TreeFlatten for Lion {
    fn flatten_state(&self) -> Vec<Array> {
        self.exp_avg.borrow().clone()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut ms = self.exp_avg.borrow_mut();
        for arr in ms.iter_mut() {
            *arr = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for Lion {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let mut ms = self.exp_avg.borrow_mut();
        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            // Decoupled Weight Decay
            if self.weight_decay > 0.0 {
                *p = p.multiply_scalar(1.0 - self.lr * self.weight_decay)?;
            }

            // c = beta1 * m + (1 - beta1) * g
            let c = ms[i].multiply_scalar(self.beta1)?.add(&g.multiply_scalar(1.0 - self.beta1)?)?;
            
            // update = lr * sign(c)
            // (Assuming you have a .sign() method in ops.rs, or use where_op to simulate sign)
            let update = c.sign()?.multiply_scalar(self.lr)?;
            *p = p.subtract(&update)?;

            // m = beta2 * m + (1 - beta2) * g
            ms[i] = ms[i].multiply_scalar(self.beta2)?.add(&g.multiply_scalar(1.0 - self.beta2)?)?;
        }
        Ok(())
    }
}

/// Adafactor
pub struct Adafactor {
    pub lr: f32,
    pub eps1: f32,
    pub eps2: f32,
    v: std::cell::RefCell<Vec<Array>>,
}

impl Adafactor {
    pub fn new(lr: f32, params: &[Array]) -> Result<Self> {
        let v = params.iter().map(|p| Array::zeros_like(p)).collect::<Result<Vec<_>>>()?;
        Ok(Self { lr, eps1: 1e-30, eps2: 1e-3, v: std::cell::RefCell::new(v) })
    }
}


impl TreeFlatten for Adafactor {
    fn flatten_state(&self) -> Vec<Array> {
        self.v.borrow().clone()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        let mut vs = self.v.borrow_mut();
        for arr in vs.iter_mut() {
            *arr = flat_arrays.next().unwrap().clone();
        }
    }
}

impl Optimizer for Adafactor {
    fn update(&mut self, params: Vec<&mut Array>, grads: Vec<Array>) -> Result<()> {
        let mut vs = self.v.borrow_mut();
        for (i, (p, g)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            // Simplified update: v = v + g^2
            let v_new = vs[i].add(&g.multiply(&g)?)?;
            let r = g.divide(&v_new.sqrt()?.add_scalar(self.eps2)?)?;
            *p = p.subtract(&r.multiply_scalar(self.lr)?)?;
            vs[i] = v_new;
        }
        Ok(())
    }
}
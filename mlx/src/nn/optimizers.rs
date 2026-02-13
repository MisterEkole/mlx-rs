use crate::{Array, Result};

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
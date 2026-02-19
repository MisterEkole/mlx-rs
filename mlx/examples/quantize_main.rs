use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Module, Optimizer, Linear,ModuleParams,Adam, cross_entropy};
use mlx::nn::layers::normalization::LayerNorm;
use mlx::nn::layers::embedding::Embedding;
use mlx::nn::transformers::TransformerEncoder;
use mlx::io;
use mlx::quantization;
use std::cell::RefCell;
use std::collections::HashMap;

struct TransformerClassifier {
    token_embed: Embedding,
    pos_embed: Array,
    encoder: TransformerEncoder,
    norm: LayerNorm,
    classifier: Linear,
}

impl TransformerClassifier {
    fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        n_classes: usize,
        dropout: f32,
        key: &Array,
    ) -> Result<Self> {
        let (k1, rest) = key.split()?;
        let (k2, rest) = rest.split()?;
        let (k3, k4) = rest.split()?;

        let pos_embed = Array::random_uniform(
            &[max_seq_len, d_model], -0.02, 0.02, Dtype::Float32, &k2,
        )?;

        Ok(Self {
            token_embed: Embedding::new(vocab_size, d_model, None, &k1)?,
            pos_embed,
            encoder: TransformerEncoder::new(n_layers, d_model, n_heads, d_ff, dropout, &k3)?,
            norm: LayerNorm::new(d_model, 1e-5)?,
            classifier: Linear::new(d_model, n_classes, true, &k4)?,
        })
    }

    fn forward(&self, token_ids: &Array) -> Result<Array> {
        let x = self.token_embed.forward(token_ids)?;
        let x = x.add(&self.pos_embed)?;
        let x = self.encoder.forward(&x)?;
        let x = self.norm.forward(&x)?;
        let x = x.mean_axis(1, false)?;
        self.classifier.forward(&x)
    }

    fn parameters_owned(&self) -> Vec<Array> {
        let mut params = Vec::new();
        params.extend(self.token_embed.parameters().into_iter().cloned());
        params.push(self.pos_embed.clone());
        params.extend(self.encoder.parameters().into_iter().cloned());
        params.extend(self.norm.parameters().into_iter().cloned());
        params.extend(self.classifier.parameters().into_iter().cloned());
        params
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        let mut offset = 0;
        let n_emb = self.token_embed.parameters().len();
        self.token_embed.update_parameters(&new_params[offset..offset + n_emb]);
        offset += n_emb;

        self.pos_embed = new_params[offset].clone();
        offset += 1;

        let n_enc = self.encoder.parameters().len();
        self.encoder.update_parameters(&new_params[offset..offset + n_enc]);
        offset += n_enc;

        let n_norm = self.norm.parameters().len();
        self.norm.update_parameters(&new_params[offset..offset + n_norm]);
        offset += n_norm;

        let n_cls = self.classifier.parameters().len();
        self.classifier.update_parameters(&new_params[offset..offset + n_cls]);
    }

    fn get_parameter_map(&self) -> HashMap<String, Array> {
        self.parameters_owned().into_iter().enumerate()
            .map(|(i, p)| (format!("layer_param_{}", i), p))
            .collect()
    }

    fn load_from_map(&mut self, map: HashMap<String, Array>) {
        let mut params = Vec::new();
        for i in 0..map.len() {
            params.push(map.get(&format!("layer_param_{}", i)).unwrap().clone());
        }
        self.update_parameters(&params);
    }
}

fn main() -> Result<()> {
    Device::new(DeviceType::Gpu).set_default()?;

    let (vocab_size, seq_len, d_model, n_layers, n_classes) = (256, 32, 128, 6, 6);
    let (batch_size, lr, steps) = (16, 1e-3, 500);

    let key = Array::key(42)?;
    let model = RefCell::new(TransformerClassifier::new(
        vocab_size, seq_len, d_model, 4, 256, n_layers, n_classes, 0.0, &key,
    )?);

    let mut optimizer = Adam::new(lr, &model.borrow().parameters_owned())?;
    let (data_key, label_key) = key.split()?;

    let token_ids = Array::random_uniform(&[batch_size, seq_len], 0.0, vocab_size as f32, Dtype::Float32, &data_key)?.cast(Dtype::Int32)?;
    
    let raw_labels = Array::random_uniform(
        &[batch_size], 0.0, n_classes as f32, Dtype::Float32, &label_key,
    )?.cast(Dtype::Int32)?;
    let class_range = Array::arange(0.0, n_classes as f64, 1.0, Dtype::Float32)?;
    let targets = raw_labels.reshape(&[batch_size as i32, 1])?
        .equal(&class_range.reshape(&[1, n_classes as i32])?)?
        .cast(Dtype::Float32)?;

    println!("Training {} layer transformer...", n_layers);
    for step in 0..steps {
        let mut params = model.borrow().parameters_owned();
        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            let logits = {
                let mut m = model.borrow_mut();
                m.update_parameters(p);
                m.forward(&token_ids)?
            };
            cross_entropy(&logits, &targets)
        }, &params)?;

        optimizer.update(params.iter_mut().collect(), grads)?;
        model.borrow_mut().update_parameters(&params);
        if step % 100 == 0 { println!("Step {}: Loss {:.4}", step, loss.item::<f32>()?); }
    }

    // Save FP32 Model
    let trained_weights = model.borrow().get_parameter_map();
    io::save_safetensors("weights_fp32.safetensors", &trained_weights)?;
    io::save_npz("weights_fp32.npz", &trained_weights)?;

    println!("mlx_optional_int size: {}", std::mem::size_of::<mlx::sys::mlx_optional_int>());
    println!("mlx_optional_int align: {}", std::mem::align_of::<mlx::sys::mlx_optional_int>());

    println!("sizeof mlx_optional_int: {}", std::mem::size_of::<mlx::sys::mlx_optional_int>());
    println!("sizeof mlx_array: {}", std::mem::size_of::<mlx::sys::mlx_array>());
    println!("sizeof mlx_vector_array: {}", std::mem::size_of::<mlx::sys::mlx_vector_array>());
    println!("sizeof mlx_stream: {}", std::mem::size_of::<mlx::sys::mlx_stream>());

   
// ── Quantize and Dequantize with Debugging ────────────────────────────
    println!("\n--- Starting Quantization Process ---");
    let mut dequantized_map = HashMap::new();
    let bits = 4;
    let group_size = 64;

    for (name, weight) in trained_weights {
        let shape = weight.shape()?;
        let last_dim = shape[shape.len() - 1] as usize;
        

        // Check if the layer is actually quantizable
        if shape.len() >= 2 && last_dim % group_size as usize == 0 {
            //println!("Quantizing {} | Shape: {:?} | Last Dim: {} (Valid)", name, shape, last_dim);
            
            // Attempt the FFI call
            match quantization::quantize(&weight, bits, group_size) {
                Ok((q, s, b)) => {
                    // Attempt Dequantization
                    match quantization::dequantize(&q, &s, &b, bits, group_size) {
                        Ok(dq) => {
                            dequantized_map.insert(name, dq);
                        }
                        Err(e) => {
                            println!("  [!] Dequantization failed for {}: {:?}", name, e);
                            dequantized_map.insert(name, weight.clone());
                        }
                    }
                }
                Err(e) => {
                    println!("  [!] Quantization FFI error for {}: {:?}", name, e);
                    dequantized_map.insert(name, weight.clone());
                }
            }
        } else {
            
            //println!("Skipping {}   | Shape: {:?} | Last Dim: {} (Invalid for group_size {})",  name, shape, last_dim, group_size);
            dequantized_map.insert(name, weight.clone());
        }
    }
    println!("--- Quantization Process Finished ---\n");

    // Save Dequantized Model
    io::save_safetensors("weights_dequant.safetensors", &dequantized_map)?;
    io::save_npz("weights_dequant.npz", &dequantized_map)?;

    model.borrow_mut().load_from_map(dequantized_map);
    println!("Dequantized model inference: {:?}", model.borrow().forward(&token_ids)?.shape()?);

    Ok(())
}
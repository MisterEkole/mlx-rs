// examples/transformer_main.rs
//
// Transformer Encoder for sequence classification.
//
// Architecture:
//   Token Embedding → Positional Embedding → Transformer Encoder (N layers) →
//   LayerNorm → Mean Pool → Linear Classifier
//
// Task: classify random token sequences into N classes (synthetic data).
//
// Schedule: Linear warmup (200 steps) → Cosine decay — the standard
// recipe for transformer pretraining (Chinchilla, LLaMA, etc.).

use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Module, ModuleParams, Optimizer, Linear, cross_entropy, Adam};
use mlx::nn::{LRScheduler, WarmupCosineSchedule};
use mlx::nn::layers::normalization::LayerNorm;
use mlx::nn::layers::embedding::Embedding;
use mlx::nn::transformers::TransformerEncoder;
use std::cell::RefCell;

// ═══════════════════════════════════════════════════════════════════════════
// Model Definition
// ═══════════════════════════════════════════════════════════════════════════
//
// Can't use #[derive(ModuleParams)] from an example file because the macro
// emits `crate::nn::ModuleParams` which only resolves inside the mlx crate.
// So we implement parameter traversal manually — all inner components have
// derive-generated impls, we just delegate to them.

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

        let n = self.token_embed.parameters().len();
        self.token_embed.update_parameters(&new_params[offset..offset + n]);
        offset += n;

        self.pos_embed = new_params[offset].clone();
        offset += 1;

        let n = self.encoder.parameters().len();
        self.encoder.update_parameters(&new_params[offset..offset + n]);
        offset += n;

        let n = self.norm.parameters().len();
        self.norm.update_parameters(&new_params[offset..offset + n]);
        offset += n;

        let n = self.classifier.parameters().len();
        self.classifier.update_parameters(&new_params[offset..offset + n]);
    }

    fn train(&mut self, training: bool) {
        self.encoder.train(training);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Training
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;
    println!("--- Using Device: GPU ---");

    // ── Hyperparameters ───────────────────────────────────────────────────

    let vocab_size: usize = 256;
    let seq_len: usize = 32;
    let d_model: usize = 64;
    let n_heads: usize = 4;
    let d_ff: usize = 128;
    let n_layers: usize = 2;
    let n_classes: usize = 5;
    let batch_size: usize = 16;
    let peak_lr: f32 = 1e-3;
    let warmup_steps: u32 = 200;
    let steps: usize = 5000;

    let key = Array::key(42)?;

    // ── Build Model ───────────────────────────────────────────────────────

    let model = RefCell::new(TransformerClassifier::new(
        vocab_size, seq_len, d_model, n_heads, d_ff, n_layers, n_classes,
        0.0, &key,
    )?);

    let param_count = model.borrow().parameters_owned().len();
    println!("Model parameters: {} Arrays", param_count);
    println!(
        "Architecture: Embedding({}, {}) → {}× EncoderLayer(d={}, h={}, ff={}) → Linear({}, {})",
        vocab_size, d_model, n_layers, d_model, n_heads, d_ff, d_model, n_classes
    );

    // ── Optimizer + Scheduler ─────────────────────────────────────────────
    //
    // WarmupCosineSchedule: linear ramp 0 → peak_lr over `warmup_steps`,
    // then cosine anneal peak_lr → 0 over the remaining steps.
  

    let mut optimizer = Adam::new(peak_lr, &model.borrow().parameters_owned())?;
    let mut scheduler = WarmupCosineSchedule::new(
        peak_lr,
        warmup_steps,
        steps as u32,
        0.0, 
    );

    println!(
        "Schedule: WarmupCosine(peak={}, warmup={}, total={}, eta_min=0)",
        peak_lr, warmup_steps, steps
    );

    // ── Synthetic Data ────────────────────────────────────────────────────

    let (data_key, label_key) = key.split()?;

    let token_ids = Array::random_uniform(
        &[batch_size, seq_len], 0.0, vocab_size as f32, Dtype::Float32, &data_key,
    )?.cast(Dtype::Int32)?;

    let raw_labels = Array::random_uniform(
        &[batch_size], 0.0, n_classes as f32, Dtype::Float32, &label_key,
    )?.cast(Dtype::Int32)?;

    let class_range = Array::arange(0.0, n_classes as f64, 1.0, Dtype::Float32)?;
    let targets = raw_labels.reshape(&[batch_size as i32, 1])?
        .equal(&class_range.reshape(&[1, n_classes as i32])?)?
        .cast(Dtype::Float32)?;

    // ── Training Loop ─────────────────────────────────────────────────────

    println!("--- Transformer Training Started ---");
    println!("    {} steps, batch_size={}, seq_len={}", steps, batch_size, seq_len);

    model.borrow_mut().train(true);

    for step in 0..steps {
        let mut params = model.borrow().parameters_owned();

        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            let logits = {
                let mut model_mut = model.borrow_mut();
                model_mut.update_parameters(p);
                model_mut.forward(&token_ids)?
            };
            cross_entropy(&logits, &targets)
        }, &params)?;

        optimizer.update(params.iter_mut().collect(), grads)?;

        model.borrow_mut().update_parameters(&params);

        // Advance the scheduler and update the optimizer's learning rate
        optimizer.lr = scheduler.step();

        let mut to_eval = params.clone();
        to_eval.push(loss.clone());
        Array::eval_all(&to_eval[..])?;

        if step % 50 == 0 {
            let loss_val: f32 = loss.item()?;
            println!(
                "Step {:>5}: Loss = {:.6}  lr = {:.6}",
                step, loss_val, scheduler.get_lr()
            );
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────

    model.borrow_mut().train(false);

    let logits = model.borrow().forward(&token_ids)?;
    logits.eval()?;

    println!("\n--- Training Complete ---");
    println!("Final logits shape: {:?}", logits.shape()?);

    Ok(())
}
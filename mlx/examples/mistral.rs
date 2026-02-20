// =============================================================================
// mlx/examples/mistral.rs
//
// Mistral-7B inference in pure Rust on Apple Silicon via mlx-rs.
//
// Architecture: Mistral-7B-v0.1
//   - 32 transformer layers, 4096 hidden dim, 14336 MLP dim
//   - 32 query heads, 8 KV heads (GQA), 128 head dim
//   - RMSNorm, RoPE, SiLU gate activation
//   - Vocab: 32000
//
// Usage:
//   cargo run --example mistral -- /path/to/mistral-7b-v0.1
// =============================================================================

use mlx::{Array, Result, Dtype, Error};
use mlx::nn::layers::activations::{silu, softmax};
use mlx::nn::layers::linear::Linear;
use mlx::nn::layers::normalization::RMSNorm;
use mlx::nn::Module;
use mlx::nn::transformers::kv_cache::{KVCache, KVCacheCollection};
use mlx::io::safetensors::load_safetensors;
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl MistralConfig {
    pub fn mistral_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

// =============================================================================
// RoPE (Rotary Positional Embeddings)
// =============================================================================

/// Returns (cos, sin) each of shape [seq_len, head_dim/2].
fn rope_frequencies(
    head_dim: usize,
    seq_len: usize,
    offset: usize,
    theta: f32,
) -> Result<(Array, Array)> {
    let half_dim = head_dim / 2;

    // inv_freq[i] = 1.0 / (theta^(2i/d))
    let mut inv_freq_vals = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq_vals[i] = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
    }
    // [1, half_dim]
    let inv_freq = Array::from_slice(&inv_freq_vals, &[1, half_dim], Dtype::Float32)?;

    // positions: [seq_len, 1]
    let positions: Vec<f32> = (offset..offset + seq_len).map(|p| p as f32).collect();
    let pos = Array::from_slice(&positions, &[seq_len, 1], Dtype::Float32)?;

    // freqs = pos @ inv_freq → [seq_len, half_dim]
    let freqs = pos.matmul(&inv_freq)?;

    Ok((freqs.cos()?, freqs.sin()?))
}

/// Apply RoPE to q, k tensors of shape [batch, n_heads, seq, head_dim].
/// cos_vals, sin_vals shape: [seq_len, half_dim]
fn apply_rope(
    q: &Array,
    k: &Array,
    cos_vals: &Array,
    sin_vals: &Array,
) -> Result<(Array, Array)> {
    let shape = q.shape()?;
    let seq_len = shape[2] as i32;
    let half = (shape[3] / 2) as i32;

    // [seq, half] → [1, 1, seq, half]
    let cos_e = cos_vals.reshape(&[1, 1, seq_len, half])?;
    let sin_e = sin_vals.reshape(&[1, 1, seq_len, half])?;

    // [1,1,seq,half] → [1,1,seq,head_dim] by concatenating [cos,cos]
    let cos_full = Array::concatenate(&[&cos_e, &cos_e], -1)?;
    let sin_full = Array::concatenate(&[&sin_e, &sin_e], -1)?;

    let q_out = rope_rotate(q, &cos_full, &sin_full)?;
    let k_out = rope_rotate(k, &cos_full, &sin_full)?;

    Ok((q_out, k_out))
}

/// x * cos + rotate_half(x) * sin
fn rope_rotate(x: &Array, cos_full: &Array, sin_full: &Array) -> Result<Array> {
    let s = x.shape()?;
    let half = (s[3] / 2) as i32;
    let full = s[3] as i32;

    // x1 = x[..., :half], x2 = x[..., half:]
    let x1 = x.slice_axis(-1, 0, half)?;
    let x2 = x.slice_axis(-1, half, full)?;

    // rotate_half = [-x2, x1]
    let neg_x2 = x2.negative()?;
    let rotated = Array::concatenate(&[&neg_x2, &x1], -1)?;

    // x * cos + rotated * sin
    let a = x.multiply(cos_full)?;
    let b = rotated.multiply(sin_full)?;
    a.add(&b)
}

// =============================================================================
// MLP
// =============================================================================

pub struct MistralMLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl MistralMLP {
    /// output = down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up)?)
    }
}

// =============================================================================
// Grouped Query Attention
// =============================================================================

pub struct MistralAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl MistralAttention {
    pub fn forward(
        &self,
        x: &Array,
        cos_vals: &Array,
        sin_vals: &Array,
        cache: &mut KVCache,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = x.shape()?;
        let batch = shape[0] as i32;
        let seq_len = shape[1] as i32;
        let n_heads = self.num_heads as i32;
        let n_kv = self.num_kv_heads as i32;
        let h_dim = self.head_dim as i32;

        // Project → reshape → [batch, heads, seq, head_dim]
        let q = self.q_proj.forward(x)?
            .reshape(&[batch, seq_len, n_heads, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k = self.k_proj.forward(x)?
            .reshape(&[batch, seq_len, n_kv, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let v = self.v_proj.forward(x)?
            .reshape(&[batch, seq_len, n_kv, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE
        let (q, k) = apply_rope(&q, &k, cos_vals, sin_vals)?;

        // KV cache update
        let (k, v) = cache.update(&k, &v)?;

        // Expand KV heads for GQA: [B, kv, S, D] → [B, kv*rep, S, D]
        let n_rep = self.num_heads / self.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            (repeat_kv(&k, n_rep)?, repeat_kv(&v, n_rep)?)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?.divide_scalar(scale)?;

        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        let weights = softmax(&scores, -1)?;
        let attn_out = weights.matmul(&v)?;

        // [batch, heads, seq, dim] → [batch, seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

/// [B, kv_heads, S, D] → [B, kv_heads * n_rep, S, D]
fn repeat_kv(x: &Array, n_rep: usize) -> Result<Array> {
    if n_rep == 1 { return Ok(x.clone()); }
    let s = x.shape()?;
    let (b, kv, seq, d) = (s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32);

    x.reshape(&[b, kv, 1, seq, d])?
        .broadcast_to(&[b, kv, n_rep as i32, seq, d])?
        .reshape(&[b, kv * n_rep as i32, seq, d])
}

// =============================================================================
// Decoder Layer
// =============================================================================

pub struct MistralDecoderLayer {
    pub self_attn: MistralAttention,
    pub mlp: MistralMLP,
    pub input_layernorm: RMSNorm,
    pub post_attn_layernorm: RMSNorm,
}

impl MistralDecoderLayer {
    pub fn forward(
        &self,
        x: &Array,
        cos_vals: &Array,
        sin_vals: &Array,
        cache: &mut KVCache,
        mask: Option<&Array>,
    ) -> Result<Array> {
        // Attention block with pre-norm and residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, cos_vals, sin_vals, cache, mask)?;
        let x = x.add(&attn_out)?;

        // MLP block with pre-norm and residual
        let normed = self.post_attn_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }
}

// =============================================================================
// Full Model
// =============================================================================

pub struct MistralModel {
    pub config: MistralConfig,
    pub embed_weight: Array,
    pub layers: Vec<MistralDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
}

impl MistralModel {
    
    pub fn forward(
        &self,
        input_ids: &Array,
        cache: &mut KVCacheCollection,
    ) -> Result<Array> {
        let offset = cache.offset();
        let seq_len = input_ids.shape()?[1];

        let mut hidden = self.embed_weight.take(input_ids, 0)?;

      
        let (cos_vals, sin_vals) = rope_frequencies(
            self.config.head_dim, seq_len, offset, self.config.rope_theta,
        )?;

       
        let mask = if seq_len > 1 {
            Some(make_causal_mask(seq_len, offset)?)
        } else {
            None
        };

   
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden, &cos_vals, &sin_vals, cache.get_mut(i), mask.as_ref(),
            )?;
        }

       
        hidden = self.norm.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    /// Load from HuggingFace safetensors checkpoint.
    pub fn from_safetensors(
        weights: &HashMap<String, Array>,
        config: MistralConfig,
    ) -> Result<Self> {
        let get = |name: &str| -> Result<Array> {
            weights.get(name).cloned().ok_or_else(|| {
                Error::OperationFailed(format!("Missing weight: {}", name))
            })
        };

        let embed_weight = get("model.embed_tokens.weight")?;

        let lm_head_weight = weights.get("lm_head.weight")
            .cloned()
            .unwrap_or_else(|| embed_weight.clone());
        let lm_head = Linear::from_weights(lm_head_weight, None)?;

        let norm = RMSNorm {
            weight: get("model.norm.weight")?,
            eps: config.rms_norm_eps,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{}", i);
            layers.push(MistralDecoderLayer {
                self_attn: MistralAttention {
                    q_proj: Linear::from_weights(get(&format!("{p}.self_attn.q_proj.weight"))?, None)?,
                    k_proj: Linear::from_weights(get(&format!("{p}.self_attn.k_proj.weight"))?, None)?,
                    v_proj: Linear::from_weights(get(&format!("{p}.self_attn.v_proj.weight"))?, None)?,
                    o_proj: Linear::from_weights(get(&format!("{p}.self_attn.o_proj.weight"))?, None)?,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                },
                mlp: MistralMLP {
                    gate_proj: Linear::from_weights(get(&format!("{p}.mlp.gate_proj.weight"))?, None)?,
                    up_proj: Linear::from_weights(get(&format!("{p}.mlp.up_proj.weight"))?, None)?,
                    down_proj: Linear::from_weights(get(&format!("{p}.mlp.down_proj.weight"))?, None)?,
                },
                input_layernorm: RMSNorm {
                    weight: get(&format!("{p}.input_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                },
                post_attn_layernorm: RMSNorm {
                    weight: get(&format!("{p}.post_attention_layernorm.weight"))?,
                    eps: config.rms_norm_eps,
                },
            });
        }

        Ok(Self { config, embed_weight, layers, norm, lm_head })
    }
}

// =============================================================================
// Causal Mask
// =============================================================================

fn make_causal_mask(seq_len: usize, offset: usize) -> Result<Array> {
    let total = seq_len + offset;
    let mut data = vec![0.0f32; seq_len * total];
    for i in 0..seq_len {
        for j in (i + offset + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Array::from_slice(&data, &[1, 1, seq_len, total], Dtype::Float32)
}

// =============================================================================
// Token Sampling
// =============================================================================


fn sample_greedy(logits: &Array) -> Result<u32> {
    let shape = logits.shape()?;
    let seq = shape[1];
    let vocab = shape[2];


    let last = logits
        .slice_axis(1, (seq - 1) as i32, seq as i32)?
        .reshape(&[vocab as i32])?;

    let token_arr = last.argmax_axis(0, false)?.cast(Dtype::Float32)?;
    token_arr.eval()?;
    let val: f32 = token_arr.item()?;
    Ok(val as u32)
}

/// Temperature sampling via Gumbel-max trick.
fn sample_temperature(logits: &Array, temp: f32, seed: u64) -> Result<u32> {
    let shape = logits.shape()?;
    let seq = shape[1];
    let vocab = shape[2];

    let last = logits
        .slice_axis(1, (seq - 1) as i32, seq as i32)?
        .reshape(&[vocab as i32])?;

    let scaled = last.divide_scalar(temp)?;

    // Numerical stability: subtract max
    let max_val = scaled.max_axis(0, true)?;
    let stabilized = scaled.subtract(&max_val)?;

    // Gumbel noise: -log(-log(uniform))
    let key = Array::key(seed)?;
    let uniform = Array::random_uniform(&[vocab], 0.0001, 0.9999, Dtype::Float32, &key)?;
    let gumbel = uniform.log()?.negative()?.log()?.negative()?;

    let perturbed = stabilized.add(&gumbel)?;
    let token_arr = perturbed.argmax_axis(0, false)?.cast(Dtype::Float32)?;
    token_arr.eval()?;
    let val: f32 = token_arr.item()?;
    Ok(val as u32)
}

// =============================================================================
// Generation Loop
// =============================================================================

pub fn generate(
    model: &MistralModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
) -> Result<Vec<u32>> {
    let mut cache = KVCacheCollection::new(model.config.num_hidden_layers);
    let mut generated = Vec::with_capacity(max_tokens);
    let mut seed: u64 = 42;

    // Phase 1: Prefill — process entire prompt
    let prompt = Array::from_slice(prompt_tokens, &[1, prompt_tokens.len()], Dtype::UInt32)?;
    let logits = model.forward(&prompt, &mut cache)?;

    let mut token = if temperature <= 1e-9 {
        sample_greedy(&logits)?
    } else {
        seed += 1;
        sample_temperature(&logits, temperature, seed)?
    };
    generated.push(token);

    // Phase 2: Decode — one token at a time with KV cache
    for _ in 1..max_tokens {
        let input = Array::from_slice(&[token], &[1, 1], Dtype::UInt32)?;
        let logits = model.forward(&input, &mut cache)?;

        token = if temperature <= 1e-9 {
            sample_greedy(&logits)?
        } else {
            seed += 1;
            sample_temperature(&logits, temperature, seed)?
        };

        // EOS = token 2 for Mistral
        if token == 2 { break; }
        generated.push(token);
    }

    Ok(generated)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    // --- Parse CLI args ---
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: mistral <model-dir> [prompt]");
        eprintln!("");
        eprintln!("  model-dir   Directory containing:");
        eprintln!("                - model-*.safetensors (weight shards)");
        eprintln!("                - tokenizer.json (HuggingFace tokenizer)");
        eprintln!("  prompt      Optional text prompt (default: \"Hello, how are you?\")");
        eprintln!("");
        eprintln!("Example:");
        eprintln!("  cargo run --example mistral -- ./mistral-7b \"What is Rust?\"");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt_text = if args.len() >= 3 {
        args[2..].join(" ")
    } else {
        "Hello, how are you?".to_string()
    };

    // --- Load tokenizer ---
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    println!("Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::OperationFailed(format!("Failed to load tokenizer: {}", e)))?;

    // --- Load model weights ---
    println!("Loading Mistral-7B from {}...", model_dir);
    let config = MistralConfig::mistral_7b();

    let mut all_weights = HashMap::new();
    let mut shard_paths: Vec<_> = std::fs::read_dir(model_dir)
        .map_err(|e| Error::OperationFailed(e.to_string()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_paths.sort(); // Load in order: model-00001, model-00002, ...

    for path in &shard_paths {
        println!("  Loading {:?}", path.file_name().unwrap());
        let shard = load_safetensors(path)?;
        all_weights.extend(shard);
    }
    println!("  {} weight tensors loaded", all_weights.len());

    // --- Build model ---
    let model = MistralModel::from_safetensors(&all_weights, config)?;
    println!("Model ready: {} layers", model.layers.len());

    // --- Tokenize prompt ---
    let encoding = tokenizer.encode(prompt_text.as_str(), false)
        .map_err(|e| Error::OperationFailed(format!("Tokenization failed: {}", e)))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Ensure BOS token (1) is prepended
    if prompt_tokens.first() != Some(&1) {
        prompt_tokens.insert(0, 1);
    }
    println!("Prompt: \"{}\" → {} tokens", prompt_text, prompt_tokens.len());

    // --- Generate ---
    let max_tokens = 128;
    let temperature = 0.7;

    println!("Generating (max {} tokens, temp {})...", max_tokens, temperature);
    let output_tokens = generate(&model, &prompt_tokens, max_tokens, temperature)?;

    // --- Decode output ---
    let output_text = tokenizer.decode(&output_tokens, true)
        .map_err(|e| Error::OperationFailed(format!("Decode failed: {}", e)))?;

    println!("\n--- Output ---");
    println!("{}{}", prompt_text, output_text);
    println!("--- ({} tokens generated) ---", output_tokens.len());

    Ok(())
}
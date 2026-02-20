// =============================================================================
// mlx/tests/test_nn_layers.rs
//
// Tests for neural network layers: Linear, RMSNorm, LayerNorm, Embedding,
// and all activation functions.
//
// Run: cargo test --test test_nn_layers
// =============================================================================

use mlx::{Array, Dtype};
use mlx::nn::Module;
use mlx::nn::layers::linear::Linear;
use mlx::nn::layers::normalization::{RMSNorm, LayerNorm};
use mlx::nn::layers::embedding::Embedding;
use mlx::nn::layers::activations::{
    relu, gelu, silu, sigmoid, tanh, softmax, leaky_relu, elu,
    ReLU, GELU, Softmax,
};

// =============================================================================
// Linear Layer
// =============================================================================

#[test]
fn test_linear_new() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(4, 3, true, &key).unwrap();

    assert_eq!(linear.in_features, 4);
    assert_eq!(linear.out_features, 3);
    assert_eq!(linear.weight.shape().unwrap(), vec![3, 4]); // [out, in]
    assert!(linear.bias.is_some());
    assert_eq!(linear.bias.as_ref().unwrap().shape().unwrap(), vec![3]);
}

#[test]
fn test_linear_no_bias() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(4, 3, false, &key).unwrap();

    assert!(linear.bias.is_none());
}

#[test]
fn test_linear_forward_shape() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(4, 3, true, &key).unwrap();

    // Single sample: [1, 4] → [1, 3]
    let x = Array::full(&[1, 4], 1.0, Dtype::Float32).unwrap();
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 3]);
}

#[test]
fn test_linear_forward_batch() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(8, 5, true, &key).unwrap();

    // Batch: [16, 8] → [16, 5]
    let x = Array::full(&[16, 8], 0.5, Dtype::Float32).unwrap();
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.shape().unwrap(), vec![16, 5]);
}

#[test]
fn test_linear_forward_3d() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(64, 32, false, &key).unwrap();

    // [batch, seq, features]: [2, 10, 64] → [2, 10, 32]
    let x = Array::full(&[2, 10, 64], 0.1, Dtype::Float32).unwrap();
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.shape().unwrap(), vec![2, 10, 32]);
}

#[test]
fn test_linear_from_weights() {
    // Manual weights: identity-like
    let weight = Array::from_slice(
        &[1.0f32, 0.0, 0.0, 1.0],
        &[2, 2], Dtype::Float32,
    ).unwrap();
    let bias = Array::from_slice(&[0.5f32, -0.5], &[2], Dtype::Float32).unwrap();
    let linear = Linear::from_weights(weight, Some(bias)).unwrap();

    assert_eq!(linear.in_features, 2);
    assert_eq!(linear.out_features, 2);

    let x = Array::from_slice(&[3.0f32, 7.0], &[1, 2], Dtype::Float32).unwrap();
    let out = linear.forward(&x).unwrap();
    let data = out.to_vec_f32().unwrap();
    // y = x @ W^T + b = [3, 7] @ [[1, 0], [0, 1]]^T + [0.5, -0.5] = [3.5, 6.5]
    assert!((data[0] - 3.5).abs() < 1e-4);
    assert!((data[1] - 6.5).abs() < 1e-4);
}

#[test]
fn test_linear_from_weights_no_bias() {
    let weight = Array::from_slice(
        &[2.0f32, 0.0, 0.0, 3.0],
        &[2, 2], Dtype::Float32,
    ).unwrap();
    let linear = Linear::from_weights(weight, None).unwrap();

    let x = Array::from_slice(&[1.0f32, 1.0], &[1, 2], Dtype::Float32).unwrap();
    let out = linear.forward(&x).unwrap();
    let data = out.to_vec_f32().unwrap();
    // y = [1, 1] @ [[2, 0], [0, 3]]^T = [2, 3]
    assert!((data[0] - 2.0).abs() < 1e-4);
    assert!((data[1] - 3.0).abs() < 1e-4);
}

// =============================================================================
// RMSNorm
// =============================================================================

#[test]
fn test_rmsnorm_new() {
    let norm = RMSNorm::new(64, 1e-5).unwrap();

    assert_eq!(norm.weight.shape().unwrap(), vec![64]);
    assert_eq!(norm.eps, 1e-5);
    // Weight should be initialized to 1.0
    let w = norm.weight.to_vec_f32().unwrap();
    assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn test_rmsnorm_forward_shape() {
    let norm = RMSNorm::new(8, 1e-5).unwrap();

    let x = Array::full(&[2, 4, 8], 1.0, Dtype::Float32).unwrap();
    let out = norm.forward(&x).unwrap();
    assert_eq!(out.shape().unwrap(), vec![2, 4, 8]); // Shape preserved
}

#[test]
fn test_rmsnorm_normalizes() {
    let norm = RMSNorm::new(4, 1e-5).unwrap();

    let x = Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &[1, 4], Dtype::Float32,
    ).unwrap();
    let out = norm.forward(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
    // Normalized: each element / rms
    let rms = (7.5f32).sqrt();
    for i in 0..4 {
        let expected = (i as f32 + 1.0) / rms;
        assert!((data[i] - expected).abs() < 1e-3, "idx {}: got {} expected {}", i, data[i], expected);
    }
}

#[test]
fn test_rmsnorm_uniform_input() {
    // Uniform input → output equals weight (since normalized = 1.0 * weight)
    let norm = RMSNorm::new(4, 1e-5).unwrap();

    let x = Array::full(&[1, 4], 5.0, Dtype::Float32).unwrap();
    let out = norm.forward(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    // RMS of [5,5,5,5] = 5.0, so normalized = [1,1,1,1], times weight [1,1,1,1] = [1,1,1,1]
    for val in &data {
        assert!((*val - 1.0).abs() < 1e-3);
    }
}

// =============================================================================
// LayerNorm
// =============================================================================

#[test]
fn test_layernorm_new() {
    let norm = LayerNorm::new(32, 1e-5).unwrap();

    assert_eq!(norm.weight.shape().unwrap(), vec![32]);
    assert_eq!(norm.bias.shape().unwrap(), vec![32]);
}

#[test]
fn test_layernorm_forward_shape() {
    let norm = LayerNorm::new(8, 1e-5).unwrap();

    let x = Array::full(&[2, 4, 8], 1.0, Dtype::Float32).unwrap();
    let out = norm.forward(&x).unwrap();
    assert_eq!(out.shape().unwrap(), vec![2, 4, 8]);
}

#[test]
fn test_layernorm_zero_mean() {
    let norm = LayerNorm::new(4, 1e-5).unwrap();

    let x = Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &[1, 4], Dtype::Float32,
    ).unwrap();
    let out = norm.forward(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    // With default weight=1, bias=0, output should have ~zero mean
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-4, "LayerNorm output mean should be ~0, got {}", mean);
}

// =============================================================================
// Embedding
// =============================================================================

#[test]
fn test_embedding_new() {
    let key = Array::key(42).unwrap();
    let emb = Embedding::new(1000, 64, None, &key).unwrap();

    assert_eq!(emb.num_embeddings, 1000);
    assert_eq!(emb.embedding_dim, 64);
    assert_eq!(emb.weight.shape().unwrap(), vec![1000, 64]);
}

#[test]
fn test_embedding_forward() {
    let key = Array::key(42).unwrap();
    let emb = Embedding::new(100, 32, None, &key).unwrap();

    // Look up tokens [0, 5, 10]
    let indices = Array::from_slice(&[0u32, 5, 10], &[3], Dtype::UInt32).unwrap();
    let out = emb.forward(&indices).unwrap();

    assert_eq!(out.shape().unwrap(), vec![3, 32]);
}

#[test]
fn test_embedding_forward_2d() {
    let key = Array::key(42).unwrap();
    let emb = Embedding::new(100, 16, None, &key).unwrap();

    // Batch of sequences: [2, 4] → [2, 4, 16]
    let indices = Array::from_slice(
        &[1u32, 2, 3, 4, 10, 20, 30, 40],
        &[2, 4], Dtype::UInt32,
    ).unwrap();
    let out = emb.forward(&indices).unwrap();

    assert_eq!(out.shape().unwrap(), vec![2, 4, 16]);
}

#[test]
fn test_embedding_deterministic() {
    let key = Array::key(42).unwrap();
    let emb = Embedding::new(50, 8, None, &key).unwrap();

    let idx = Array::from_slice(&[3u32], &[1], Dtype::UInt32).unwrap();
    let out1 = emb.forward(&idx).unwrap().to_vec_f32().unwrap();
    let out2 = emb.forward(&idx).unwrap().to_vec_f32().unwrap();

    // Same index should always return the same embedding
    assert_eq!(out1, out2);
}

// =============================================================================
// Activation Functions — Functional API
// =============================================================================

#[test]
fn test_relu() {
    let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], Dtype::Float32).unwrap();
    let out = relu(&x).unwrap();

    assert_eq!(out.to_vec_f32().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid() {
    let x = Array::from_slice(&[0.0f32], &[1], Dtype::Float32).unwrap();
    let out = sigmoid(&x).unwrap();
    let val: f32 = out.item().unwrap();
    assert!((val - 0.5).abs() < 1e-5); // sigmoid(0) = 0.5
}

#[test]
fn test_sigmoid_bounds() {
    let x = Array::from_slice(&[-100.0f32, 0.0, 100.0], &[3], Dtype::Float32).unwrap();
    let out = sigmoid(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    assert!(data[0] < 0.01);   // sigmoid(-100) ≈ 0
    assert!((data[1] - 0.5).abs() < 1e-5);
    assert!(data[2] > 0.99);   // sigmoid(100) ≈ 1
}

#[test]
fn test_tanh_fn() {
    let x = Array::from_slice(&[0.0f32], &[1], Dtype::Float32).unwrap();
    let out = tanh(&x).unwrap();
    let val: f32 = out.item().unwrap();
    assert!((val - 0.0).abs() < 1e-5); // tanh(0) = 0
}

#[test]
fn test_tanh_bounds() {
    let x = Array::from_slice(&[-100.0f32, 0.0, 100.0], &[3], Dtype::Float32).unwrap();
    let out = tanh(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    assert!((data[0] - (-1.0)).abs() < 1e-4);
    assert!((data[1] - 0.0).abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-4);
}

#[test]
fn test_silu() {
    // silu(x) = x * sigmoid(x)
    let x = Array::from_slice(&[0.0f32, 1.0, -1.0], &[3], Dtype::Float32).unwrap();
    let out = silu(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    // silu(0) = 0 * 0.5 = 0
    assert!((data[0] - 0.0).abs() < 1e-5);
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    assert!((data[1] - 0.7311).abs() < 1e-3);
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
    assert!((data[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_gelu() {
    let x = Array::from_slice(&[0.0f32, 1.0, -1.0], &[3], Dtype::Float32).unwrap();
    let out = gelu(&x).unwrap();
    let data = out.to_vec_f32().unwrap();

    // gelu(0) ≈ 0
    assert!((data[0]).abs() < 1e-4);
    // gelu(1) ≈ 0.841
    assert!((data[1] - 0.841).abs() < 0.02);
    // gelu(-1) ≈ -0.159
    assert!((data[2] - (-0.159)).abs() < 0.02);
}

#[test]
fn test_softmax() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], Dtype::Float32).unwrap();
    let out = softmax(&x, -1).unwrap();
    let data = out.to_vec_f32().unwrap();

    // Softmax should sum to 1
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    // Values should be ordered: p(3) > p(2) > p(1)
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);

    // All positive
    assert!(data.iter().all(|&v| v > 0.0));
}

#[test]
fn test_softmax_uniform() {
    // Equal inputs → uniform distribution
    let x = Array::full(&[1, 4], 1.0, Dtype::Float32).unwrap();
    let out = softmax(&x, -1).unwrap();
    let data = out.to_vec_f32().unwrap();

    for val in &data {
        assert!((*val - 0.25).abs() < 1e-4);
    }
}

#[test]
fn test_leaky_relu() {
    let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], Dtype::Float32).unwrap();
    let out = leaky_relu(&x, 0.1).unwrap();
    let data = out.to_vec_f32().unwrap();

    assert!((data[0] - (-0.2)).abs() < 1e-5);  // -2 * 0.1
    assert!((data[1] - (-0.1)).abs() < 1e-5);  // -1 * 0.1
    assert!((data[2] - 0.0).abs() < 1e-5);
    assert!((data[3] - 1.0).abs() < 1e-5);
    assert!((data[4] - 2.0).abs() < 1e-5);
}

#[test]
fn test_elu() {
    let x = Array::from_slice(&[-1.0f32, 0.0, 1.0], &[3], Dtype::Float32).unwrap();
    let out = elu(&x, 1.0).unwrap();
    let data = out.to_vec_f32().unwrap();

    // elu(-1, 1.0) = 1.0 * (e^(-1) - 1) ≈ -0.6321
    assert!((data[0] - (-0.6321)).abs() < 1e-3);
    assert!((data[1] - 0.0).abs() < 1e-5);
    assert!((data[2] - 1.0).abs() < 1e-5);
}

// =============================================================================
// Activation Functions — Module API
// =============================================================================

#[test]
fn test_relu_module() {
    let layer = ReLU::new();
    let x = Array::from_slice(&[-1.0f32, 0.0, 1.0], &[3], Dtype::Float32).unwrap();
    let out = layer.forward(&x).unwrap();

    assert_eq!(out.to_vec_f32().unwrap(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_gelu_module() {
    let layer = GELU::new();
    let x = Array::from_slice(&[0.0f32], &[1], Dtype::Float32).unwrap();
    let out = layer.forward(&x).unwrap();
    let val: f32 = out.item().unwrap();
    assert!(val.abs() < 1e-4); // gelu(0) ≈ 0
}

#[test]
fn test_softmax_module() {
    let layer = Softmax::new(-1);
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], Dtype::Float32).unwrap();
    let out = layer.forward(&x).unwrap();
    let sum: f32 = out.to_vec_f32().unwrap().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

// =============================================================================
// Activations on Batched Inputs
// =============================================================================

#[test]
fn test_activations_preserve_shape() {
    let x = Array::full(&[2, 4, 8], 0.5, Dtype::Float32).unwrap();
    let target = vec![2, 4, 8];

    assert_eq!(relu(&x).unwrap().shape().unwrap(), target);
    assert_eq!(gelu(&x).unwrap().shape().unwrap(), target);
    assert_eq!(silu(&x).unwrap().shape().unwrap(), target);
    assert_eq!(sigmoid(&x).unwrap().shape().unwrap(), target);
    assert_eq!(tanh(&x).unwrap().shape().unwrap(), target);
}

// =============================================================================
// Layer Composition
// =============================================================================

#[test]
fn test_linear_then_relu() {
    let key = Array::key(42).unwrap();
    let linear = Linear::new(4, 3, true, &key).unwrap();
    let activation = ReLU::new();

    let x = Array::full(&[2, 4], 1.0, Dtype::Float32).unwrap();
    let hidden = linear.forward(&x).unwrap();
    let out = activation.forward(&hidden).unwrap();

    assert_eq!(out.shape().unwrap(), vec![2, 3]);
    // All values should be >= 0 after ReLU
    let data = out.to_vec_f32().unwrap();
    assert!(data.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_embedding_then_norm() {
    let key = Array::key(42).unwrap();
    let emb = Embedding::new(100, 16, None, &key).unwrap();
    let norm = RMSNorm::new(16, 1e-5).unwrap();

    let indices = Array::from_slice(&[1u32, 5, 10], &[1, 3], Dtype::UInt32).unwrap();
    let embedded = emb.forward(&indices).unwrap();
    let normalized = norm.forward(&embedded).unwrap();

    assert_eq!(normalized.shape().unwrap(), vec![1, 3, 16]);
}
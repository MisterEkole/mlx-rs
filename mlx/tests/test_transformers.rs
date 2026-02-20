// =============================================================================
// mlx/tests/test_transformers.rs
//
// Tests for transformer components: scaled dot-product attention,
// multi-head attention, and KV cache.
//
// Run: cargo test --test test_transformers
// =============================================================================

use mlx::{Array, Dtype};
use mlx::nn::Module;
use mlx::nn::transformers::scaled_dot_product::scaled_dot_product_attention;
use mlx::nn::transformers::multi_head_attention::MultiHeadAttention;
use mlx::nn::transformers::kv_cache::{KVCache, KVCacheCollection};

// =============================================================================
// Scaled Dot-Product Attention
// =============================================================================

#[test]
fn test_sdpa_output_shape() {
    // q, k, v: [batch, heads, seq, head_dim]
    let q = Array::full(&[1, 4, 8, 16], 0.1, Dtype::Float32).unwrap();
    let k = Array::full(&[1, 4, 8, 16], 0.1, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 4, 8, 16], 0.1, Dtype::Float32).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 4, 8, 16]);
}

#[test]
fn test_sdpa_different_kv_length() {
    // Query seq=1, KV seq=10 (decoding with cache)
    let q = Array::full(&[1, 4, 1, 16], 0.1, Dtype::Float32).unwrap();
    let k = Array::full(&[1, 4, 10, 16], 0.1, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 4, 10, 16], 0.1, Dtype::Float32).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 4, 1, 16]);
}

#[test]
fn test_sdpa_with_mask() {
    // Causal mask: 0 for valid, -inf for masked
    let q = Array::full(&[1, 2, 3, 8], 0.1, Dtype::Float32).unwrap();
    let k = Array::full(&[1, 2, 3, 8], 0.1, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 2, 3, 8], 0.1, Dtype::Float32).unwrap();

    // Simple causal mask for seq=3
    let mask_data = [
        0.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY,
        0.0, 0.0, f32::NEG_INFINITY,
        0.0, 0.0, 0.0,
    ];
    let mask = Array::from_slice(&mask_data, &[1, 1, 3, 3], Dtype::Float32).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask)).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 2, 3, 8]);
}

#[test]
fn test_sdpa_identity_attention() {
    // If q == k, attention is uniform (all equal scores)
    // With uniform values, output should equal input values
    let v = Array::from_slice(
        &[1.0f32, 0.0, 0.0, 1.0],
        &[1, 1, 2, 2], Dtype::Float32,
    ).unwrap();
    let q = Array::full(&[1, 1, 2, 2], 1.0, Dtype::Float32).unwrap();
    let k = q.clone();

    let out = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
    let data = out.to_vec_f32().unwrap();

    // Uniform attention → output is mean of value rows: [0.5, 0.5] for each query
    assert!((data[0] - 0.5).abs() < 1e-4);
    assert!((data[1] - 0.5).abs() < 1e-4);
}

// =============================================================================
// Multi-Head Attention
// =============================================================================

#[test]
fn test_mha_creation() {
    let key = Array::key(42).unwrap();
    let mha = MultiHeadAttention::new(64, 4, true, &key).unwrap();

    assert_eq!(mha.embed_dim, 64);
    assert_eq!(mha.num_heads, 4);
    assert_eq!(mha.head_dim, 16); // 64 / 4
}

#[test]
fn test_mha_self_attention_shape() {
    let key = Array::key(42).unwrap();
    let mha = MultiHeadAttention::new(32, 4, true, &key).unwrap();

    // Self-attention: q = k = v = x
    let x = Array::full(&[2, 10, 32], 0.1, Dtype::Float32).unwrap();
    let out = mha.forward(&x).unwrap();

    assert_eq!(out.shape().unwrap(), vec![2, 10, 32]); // Same shape as input
}

#[test]
fn test_mha_cross_attention_shape() {
    let key = Array::key(42).unwrap();
    let mha = MultiHeadAttention::new(64, 8, false, &key).unwrap();

    let query = Array::full(&[1, 5, 64], 0.1, Dtype::Float32).unwrap();
    let kv = Array::full(&[1, 20, 64], 0.1, Dtype::Float32).unwrap();

    let out = mha.forward_qkv(&query, &kv, &kv, None).unwrap();

    // Output seq length matches query seq length
    assert_eq!(out.shape().unwrap(), vec![1, 5, 64]);
}

#[test]
fn test_mha_with_mask() {
    let key = Array::key(42).unwrap();
    let mha = MultiHeadAttention::new(16, 2, true, &key).unwrap();

    let x = Array::full(&[1, 4, 16], 0.1, Dtype::Float32).unwrap();

    // Causal mask
    let mask_data = [
        0.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
        0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY,
        0.0, 0.0, 0.0, f32::NEG_INFINITY,
        0.0, 0.0, 0.0, 0.0,
    ];
    let mask = Array::from_slice(&mask_data, &[1, 1, 4, 4], Dtype::Float32).unwrap();

    let out = mha.forward_qkv(&x, &x, &x, Some(&mask)).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 4, 16]);
}

// =============================================================================
// KV Cache
// =============================================================================

#[test]
fn test_kv_cache_new() {
    let cache = KVCache::new();

    assert!(cache.is_empty());
    assert_eq!(cache.offset(), 0);
}

#[test]
fn test_kv_cache_first_update() {
    let mut cache = KVCache::new();

    // First token: [batch=1, heads=4, seq=1, dim=8]
    let k = Array::full(&[1, 4, 1, 8], 1.0, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 4, 1, 8], 2.0, Dtype::Float32).unwrap();

    let (k_out, v_out) = cache.update(&k, &v).unwrap();

    assert!(!cache.is_empty());
    assert_eq!(cache.offset(), 1);
    assert_eq!(k_out.shape().unwrap(), vec![1, 4, 1, 8]);
    assert_eq!(v_out.shape().unwrap(), vec![1, 4, 1, 8]);
}

#[test]
fn test_kv_cache_accumulation() {
    let mut cache = KVCache::new();

    // Prefill: 5 tokens
    let k1 = Array::full(&[1, 4, 5, 8], 1.0, Dtype::Float32).unwrap();
    let v1 = Array::full(&[1, 4, 5, 8], 2.0, Dtype::Float32).unwrap();
    let (k_out, _v_out) = cache.update(&k1, &v1).unwrap();

    assert_eq!(cache.offset(), 5);
    assert_eq!(k_out.shape().unwrap(), vec![1, 4, 5, 8]);

    // Decode: 1 token at a time
    let k2 = Array::full(&[1, 4, 1, 8], 3.0, Dtype::Float32).unwrap();
    let v2 = Array::full(&[1, 4, 1, 8], 4.0, Dtype::Float32).unwrap();
    let (k_out, v_out) = cache.update(&k2, &v2).unwrap();

    assert_eq!(cache.offset(), 6);
    assert_eq!(k_out.shape().unwrap(), vec![1, 4, 6, 8]); // 5 + 1 = 6
    assert_eq!(v_out.shape().unwrap(), vec![1, 4, 6, 8]);

    // Another decode step
    let k3 = Array::full(&[1, 4, 1, 8], 5.0, Dtype::Float32).unwrap();
    let v3 = Array::full(&[1, 4, 1, 8], 6.0, Dtype::Float32).unwrap();
    let (k_out, _) = cache.update(&k3, &v3).unwrap();

    assert_eq!(cache.offset(), 7);
    assert_eq!(k_out.shape().unwrap(), vec![1, 4, 7, 8]); // 6 + 1 = 7
}

#[test]
fn test_kv_cache_reset() {
    let mut cache = KVCache::new();

    let k = Array::full(&[1, 4, 5, 8], 1.0, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 4, 5, 8], 1.0, Dtype::Float32).unwrap();
    cache.update(&k, &v).unwrap();

    assert_eq!(cache.offset(), 5);
    assert!(!cache.is_empty());

    cache.reset();

    assert_eq!(cache.offset(), 0);
    assert!(cache.is_empty());
}

// =============================================================================
// KV Cache Collection
// =============================================================================

#[test]
fn test_kv_cache_collection() {
    let mut caches = KVCacheCollection::new(32); // 32 layers

    assert_eq!(caches.len(), 32);
    assert_eq!(caches.offset(), 0);

    // Update layer 0
    let k = Array::full(&[1, 4, 3, 8], 1.0, Dtype::Float32).unwrap();
    let v = Array::full(&[1, 4, 3, 8], 1.0, Dtype::Float32).unwrap();
    caches.get_mut(0).update(&k, &v).unwrap();

    assert_eq!(caches.get(0).offset(), 3);
    assert_eq!(caches.get(1).offset(), 0); // Other layers unaffected
}

#[test]
fn test_kv_cache_collection_reset() {
    let mut caches = KVCacheCollection::new(4);

    // Update all layers
    for i in 0..4 {
        let k = Array::full(&[1, 2, 5, 4], 1.0, Dtype::Float32).unwrap();
        let v = Array::full(&[1, 2, 5, 4], 1.0, Dtype::Float32).unwrap();
        caches.get_mut(i).update(&k, &v).unwrap();
    }

    assert_eq!(caches.offset(), 5); // First layer's offset

    caches.reset();

    assert_eq!(caches.offset(), 0);
    for i in 0..4 {
        assert!(caches.get(i).is_empty());
    }
}

#[test]
fn test_kv_cache_collection_simulated_generation() {
    // Simulate a 4-layer transformer generating tokens
    let n_layers = 4;
    let batch = 1;
    let n_heads = 2;
    let head_dim = 8;
    let mut caches = KVCacheCollection::new(n_layers);

    // Phase 1: Prefill with 10 tokens
    for layer in 0..n_layers {
        let k = Array::full(&[batch, n_heads, 10, head_dim], 0.5, Dtype::Float32).unwrap();
        let v = Array::full(&[batch, n_heads, 10, head_dim], 0.5, Dtype::Float32).unwrap();
        let (k_out, v_out) = caches.get_mut(layer).update(&k, &v).unwrap();
        assert_eq!(k_out.shape().unwrap()[2], 10);
        assert_eq!(v_out.shape().unwrap()[2], 10);
    }
    assert_eq!(caches.offset(), 10);

    // Phase 2: Decode 5 tokens one at a time
    for step in 0..5 {
        for layer in 0..n_layers {
            let k = Array::full(&[batch, n_heads, 1, head_dim], 0.1, Dtype::Float32).unwrap();
            let v = Array::full(&[batch, n_heads, 1, head_dim], 0.1, Dtype::Float32).unwrap();
            let (k_out, v_out) = caches.get_mut(layer).update(&k, &v).unwrap();
            assert_eq!(k_out.shape().unwrap()[2], 10 + step + 1);
            assert_eq!(v_out.shape().unwrap()[2], 10 + step + 1);
        }
    }
    assert_eq!(caches.offset(), 15);
}
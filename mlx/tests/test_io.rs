// =============================================================================
// mlx/tests/test_io.rs
//
// Tests for I/O operations: safetensors save/load roundtrip.
//
// Run: cargo test --test test_io
// =============================================================================

use mlx::{Array, Dtype};
use mlx::io::safetensors::{save_safetensors, load_safetensors};
use std::collections::HashMap;

/// Helper to create a temp file path
fn temp_path(name: &str) -> String {
    format!("/tmp/mlx_test_{}.safetensors", name)
}

/// Helper to compare float arrays with tolerance
fn assert_arrays_close(a: &Array, b: &Array, tol: f32) {
    let va = a.to_vec_f32().unwrap();
    let vb = b.to_vec_f32().unwrap();
    assert_eq!(va.len(), vb.len(), "Array lengths differ: {} vs {}", va.len(), vb.len());
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "Mismatch at index {}: {} vs {} (tol={})", i, x, y, tol,
        );
    }
}

// =============================================================================
// Basic Save/Load Roundtrip
// =============================================================================

#[test]
fn test_save_load_single_tensor() {
    let path = temp_path("single");
    let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("tensor_a".to_string(), arr);

    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert!(loaded.contains_key("tensor_a"));
    assert_eq!(loaded["tensor_a"].shape().unwrap(), vec![2, 2]);
    assert_arrays_close(&loaded["tensor_a"], &weights["tensor_a"], 1e-6);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_save_load_multiple_tensors() {
    let path = temp_path("multi");

    let mut weights = HashMap::new();
    weights.insert(
        "weight".to_string(),
        Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], Dtype::Float32).unwrap(),
    );
    weights.insert(
        "bias".to_string(),
        Array::from_slice(&[0.5f32, -0.5], &[2], Dtype::Float32).unwrap(),
    );
    weights.insert(
        "embedding".to_string(),
        Array::full(&[100, 16], 0.01, Dtype::Float32).unwrap(),
    );

    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded.len(), 3);
    assert!(loaded.contains_key("weight"));
    assert!(loaded.contains_key("bias"));
    assert!(loaded.contains_key("embedding"));

    assert_eq!(loaded["weight"].shape().unwrap(), vec![2, 2]);
    assert_eq!(loaded["bias"].shape().unwrap(), vec![2]);
    assert_eq!(loaded["embedding"].shape().unwrap(), vec![100, 16]);

    assert_arrays_close(&loaded["weight"], &weights["weight"], 1e-6);
    assert_arrays_close(&loaded["bias"], &weights["bias"], 1e-6);

    std::fs::remove_file(&path).ok();
}

// =============================================================================
// Shape & Value Preservation
// =============================================================================

#[test]
fn test_roundtrip_1d() {
    let path = temp_path("1d");
    let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("vec".to_string(), arr);
    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded["vec"].shape().unwrap(), vec![5]);
    assert_eq!(loaded["vec"].to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_roundtrip_3d() {
    let path = temp_path("3d");
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let arr = Array::from_slice(&data, &[2, 3, 4], Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("cube".to_string(), arr);
    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded["cube"].shape().unwrap(), vec![2, 3, 4]);
    assert_arrays_close(&loaded["cube"], &weights["cube"], 1e-6);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_roundtrip_large_tensor() {
    let path = temp_path("large");
    // Simulate a weight matrix: [4096, 4096]
    let arr = Array::full(&[256, 256], 0.02, Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("large_weight".to_string(), arr);
    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded["large_weight"].shape().unwrap(), vec![256, 256]);
    let data = loaded["large_weight"].to_vec_f32().unwrap();
    assert_eq!(data.len(), 256 * 256);
    assert!(data.iter().all(|&v| (v - 0.02).abs() < 1e-5));

    std::fs::remove_file(&path).ok();
}

// =============================================================================
// Special Values
// =============================================================================

#[test]
fn test_roundtrip_zeros() {
    let path = temp_path("zeros");
    let arr = Array::zeros(&[10, 10], Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("zeros".to_string(), arr);
    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    let data = loaded["zeros"].to_vec_f32().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_roundtrip_negative_values() {
    let path = temp_path("neg");
    let arr = Array::from_slice(&[-1.5f32, -0.5, 0.0, 0.5, 1.5], &[5], Dtype::Float32).unwrap();

    let mut weights = HashMap::new();
    weights.insert("mixed".to_string(), arr);
    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded["mixed"].to_vec_f32().unwrap(), vec![-1.5, -0.5, 0.0, 0.5, 1.5]);

    std::fs::remove_file(&path).ok();
}

// =============================================================================
// Key Names (HuggingFace-style)
// =============================================================================

#[test]
fn test_roundtrip_dotted_keys() {
    let path = temp_path("dotted");

    let mut weights = HashMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        Array::full(&[32, 32], 0.1, Dtype::Float32).unwrap(),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        Array::full(&[8, 32], 0.1, Dtype::Float32).unwrap(),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        Array::full(&[100, 32], 0.01, Dtype::Float32).unwrap(),
    );

    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded.len(), 3);
    assert!(loaded.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(loaded.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(loaded.contains_key("model.embed_tokens.weight"));

    assert_eq!(
        loaded["model.layers.0.self_attn.q_proj.weight"].shape().unwrap(),
        vec![32, 32],
    );
    assert_eq!(
        loaded["model.layers.0.self_attn.k_proj.weight"].shape().unwrap(),
        vec![8, 32],
    );

    std::fs::remove_file(&path).ok();
}

// =============================================================================
// Error Cases
// =============================================================================

#[test]
fn test_load_nonexistent_file() {
    let result = load_safetensors("/tmp/mlx_test_nonexistent_file_abc123.safetensors");
    assert!(result.is_err());
}

// =============================================================================
// Simulated Model Save/Load
// =============================================================================

#[test]
fn test_roundtrip_mini_model() {
    let path = temp_path("mini_model");
    let key = Array::key(42).unwrap();

    // Build a mini "model" with typical weight shapes
    let mut weights = HashMap::new();

    // Embedding: [vocab, dim]
    weights.insert(
        "embed.weight".to_string(),
        Array::random_uniform(&[50, 16], -0.1, 0.1, Dtype::Float32, &key).unwrap(),
    );

    // Linear layers
    let (k1, k2) = key.split().unwrap();
    weights.insert(
        "linear1.weight".to_string(),
        Array::random_uniform(&[32, 16], -0.1, 0.1, Dtype::Float32, &k1).unwrap(),
    );
    weights.insert(
        "linear1.bias".to_string(),
        Array::zeros(&[32], Dtype::Float32).unwrap(),
    );
    weights.insert(
        "linear2.weight".to_string(),
        Array::random_uniform(&[16, 32], -0.1, 0.1, Dtype::Float32, &k2).unwrap(),
    );

    // Norm
    weights.insert(
        "norm.weight".to_string(),
        Array::full(&[16], 1.0, Dtype::Float32).unwrap(),
    );

    save_safetensors(&path, &weights).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_eq!(loaded.len(), 5);

    // Verify shapes match
    for (name, original) in &weights {
        let reloaded = loaded.get(name).expect(&format!("Missing key: {}", name));
        assert_eq!(
            original.shape().unwrap(),
            reloaded.shape().unwrap(),
            "Shape mismatch for {}", name,
        );
        assert_arrays_close(original, reloaded, 1e-6);
    }

    std::fs::remove_file(&path).ok();
}
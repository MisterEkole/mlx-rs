// =============================================================================
// mlx/tests/test_array_core.rs
//
// Tests for core Array operations: creation, arithmetic, shapes,
// slicing, indexing, reductions, unary ops, and data retrieval.
//
// Run: cargo test --test test_array_core
// =============================================================================

use mlx::{Array, Dtype};

// =============================================================================
// Array Creation
// =============================================================================

#[test]
fn test_from_slice_1d() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let arr = Array::from_slice(&data, &[4], Dtype::Float32).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![4]);
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.dtype(), Dtype::Float32);

    let out = arr.to_vec_f32().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_from_slice_2d() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_slice(&data, &[2, 3], Dtype::Float32).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![2, 3]);
    assert_eq!(arr.ndim(), 2);
}

#[test]
fn test_from_slice_3d() {
    let data = vec![0.0f32; 24]; // 2 x 3 x 4
    let arr = Array::from_slice(&data, &[2, 3, 4], Dtype::Float32).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![2, 3, 4]);
    assert_eq!(arr.ndim(), 3);
}

#[test]
fn test_from_slice_scalar() {
    let data = [42.0f32];
    let arr = Array::from_slice(&data, &[], Dtype::Float32).unwrap();

    // Scalar: 0-dimensional
    assert_eq!(arr.ndim(), 0);
    let val: f32 = arr.item().unwrap();
    assert_eq!(val, 42.0);
}

#[test]
fn test_full() {
    let arr = Array::full(&[3, 4], 7.5, Dtype::Float32).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![3, 4]);
    let data = arr.to_vec_f32().unwrap();
    assert_eq!(data.len(), 12);
    for val in &data {
        assert_eq!(*val, 7.5);
    }
}

#[test]
fn test_zeros() {
    let arr = Array::zeros(&[5], Dtype::Float32).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![5]);
    let data = arr.to_vec_f32().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_arange() {
    let arr = Array::arange(0.0, 5.0, 1.0, Dtype::Float32).unwrap();

    let data = arr.to_vec_f32().unwrap();
    assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_arange_fractional_step() {
    let arr = Array::arange(0.0, 1.0, 0.25, Dtype::Float32).unwrap();

    let data = arr.to_vec_f32().unwrap();
    assert_eq!(data.len(), 4); // 0.0, 0.25, 0.5, 0.75
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[3] - 0.75).abs() < 1e-6);
}

#[test]
fn test_random_uniform_shape() {
    let key = Array::key(42).unwrap();
    let arr = Array::random_uniform(&[3, 4], 0.0, 1.0, Dtype::Float32, &key).unwrap();

    assert_eq!(arr.shape().unwrap(), vec![3, 4]);
    let data = arr.to_vec_f32().unwrap();
    assert_eq!(data.len(), 12);
    for val in &data {
        assert!(*val >= 0.0 && *val <= 1.0);
    }
}

#[test]
fn test_random_key_split() {
    let key = Array::key(123).unwrap();
    let (k1, k2) = key.split().unwrap();

    // Both should be valid arrays
    k1.eval().unwrap();
    k2.eval().unwrap();
}

#[test]
fn test_zeros_like() {
    let arr = Array::full(&[3, 4], 5.0, Dtype::Float32).unwrap();
    let z = arr.zeros_like().unwrap();

    assert_eq!(z.shape().unwrap(), vec![3, 4]);
    assert!(z.to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_like() {
    let arr = Array::full(&[2, 3], 0.0, Dtype::Float32).unwrap();
    let o = arr.ones_like().unwrap();

    assert_eq!(o.shape().unwrap(), vec![2, 3]);
    assert!(o.to_vec_f32().unwrap().iter().all(|&v| v == 1.0));
}

// =============================================================================
// Arithmetic Operations
// =============================================================================

#[test]
fn test_add() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3], Dtype::Float32).unwrap();
    let c = a.add(&b).unwrap();

    assert_eq!(c.to_vec_f32().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_subtract() {
    let a = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();
    let c = a.subtract(&b).unwrap();

    assert_eq!(c.to_vec_f32().unwrap(), vec![9.0, 18.0, 27.0]);
}

#[test]
fn test_multiply() {
    let a = Array::from_slice(&[2.0f32, 3.0, 4.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0], &[3], Dtype::Float32).unwrap();
    let c = a.multiply(&b).unwrap();

    assert_eq!(c.to_vec_f32().unwrap(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_divide() {
    let a = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[2.0f32, 5.0, 10.0], &[3], Dtype::Float32).unwrap();
    let c = a.divide(&b).unwrap();

    assert_eq!(c.to_vec_f32().unwrap(), vec![5.0, 4.0, 3.0]);
}

#[test]
fn test_scalar_ops() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();

    let b = a.add_scalar(10.0).unwrap();
    assert_eq!(b.to_vec_f32().unwrap(), vec![11.0, 12.0, 13.0]);

    let c = a.multiply_scalar(3.0).unwrap();
    assert_eq!(c.to_vec_f32().unwrap(), vec![3.0, 6.0, 9.0]);

    let d = a.subtract_scalar(1.0).unwrap();
    assert_eq!(d.to_vec_f32().unwrap(), vec![0.0, 1.0, 2.0]);

    let e = a.divide_scalar(2.0).unwrap();
    let out = e.to_vec_f32().unwrap();
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_matmul() {
    // [2, 3] @ [3, 2] = [2, 2]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], Dtype::Float32).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.shape().unwrap(), vec![2, 2]);
    let data = c.to_vec_f32().unwrap();
    // Row 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // Row 0: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert!((data[0] - 58.0).abs() < 1e-4);
    assert!((data[1] - 64.0).abs() < 1e-4);
}

#[test]
fn test_matmul_batched() {
    // [2, 2, 3] @ [2, 3, 1] = [2, 2, 1]
    let a = Array::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[2, 2, 3], Dtype::Float32,
    ).unwrap();
    let b = Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3, 1], Dtype::Float32,
    ).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.shape().unwrap(), vec![2, 2, 1]);
}

#[test]
fn test_equal() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[1.0f32, 0.0, 3.0], &[3], Dtype::Float32).unwrap();
    // equal() returns Bool dtype — must cast to Float32 before reading
    let eq = a.equal(&b).unwrap().cast(Dtype::Float32).unwrap();

    let data = eq.to_vec_f32().unwrap();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 1.0);
}

// =============================================================================
// Unary Operations
// =============================================================================

#[test]
fn test_exp() {
    let a = Array::from_slice(&[0.0f32, 1.0], &[2], Dtype::Float32).unwrap();
    let b = a.exp().unwrap();
    let data = b.to_vec_f32().unwrap();

    assert!((data[0] - 1.0).abs() < 1e-5);           // e^0 = 1
    assert!((data[1] - std::f32::consts::E).abs() < 1e-4); // e^1 ≈ 2.718
}

#[test]
fn test_log() {
    let a = Array::from_slice(&[1.0f32, std::f32::consts::E], &[2], Dtype::Float32).unwrap();
    let b = a.log().unwrap();
    let data = b.to_vec_f32().unwrap();

    assert!((data[0] - 0.0).abs() < 1e-5);  // ln(1) = 0
    assert!((data[1] - 1.0).abs() < 1e-4);  // ln(e) = 1
}

#[test]
fn test_sqrt() {
    let a = Array::from_slice(&[4.0f32, 9.0, 16.0], &[3], Dtype::Float32).unwrap();
    let b = a.sqrt().unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_abs() {
    let a = Array::from_slice(&[-3.0f32, 0.0, 5.0], &[3], Dtype::Float32).unwrap();
    let b = a.abs().unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![3.0, 0.0, 5.0]);
}

#[test]
fn test_sign() {
    let a = Array::from_slice(&[-5.0f32, 0.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = a.sign().unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn test_square() {
    let a = Array::from_slice(&[2.0f32, 3.0, -4.0], &[3], Dtype::Float32).unwrap();
    let b = a.square().unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![4.0, 9.0, 16.0]);
}

// =============================================================================
// Comparison & Conditional Operations
// =============================================================================

#[test]
fn test_less_than() {
    let a = Array::from_slice(&[1.0f32, 5.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[2.0f32, 4.0, 3.0], &[3], Dtype::Float32).unwrap();
    // less_than() returns Bool dtype — cast before reading
    let c = a.less_than(&b).unwrap().cast(Dtype::Float32).unwrap();

    let data = c.to_vec_f32().unwrap();
    assert_eq!(data[0], 1.0); // 1 < 2 → true
    assert_eq!(data[1], 0.0); // 5 < 4 → false
    assert_eq!(data[2], 0.0); // 3 < 3 → false
}

#[test]
fn test_greater_than_scalar() {
    let a = Array::from_slice(&[-1.0f32, 0.0, 1.0, 5.0], &[4], Dtype::Float32).unwrap();
    // greater_than_scalar() returns Bool dtype — cast before reading
    let mask = a.greater_than_scalar(0.0).unwrap().cast(Dtype::Float32).unwrap();

    let data = mask.to_vec_f32().unwrap();
    assert_eq!(data, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_maximum() {
    let a = Array::from_slice(&[1.0f32, 5.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[4.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();
    let c = a.maximum(&b).unwrap();

    assert_eq!(c.to_vec_f32().unwrap(), vec![4.0, 5.0, 3.0]);
}

#[test]
fn test_where_op() {
    let cond = Array::from_slice(&[1.0f32, 0.0, 1.0], &[3], Dtype::Float32).unwrap();
    let on_true = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3], Dtype::Float32).unwrap();
    let on_false = Array::from_slice(&[100.0f32, 200.0, 300.0], &[3], Dtype::Float32).unwrap();
    let result = cond.where_op(&on_true, &on_false).unwrap();

    assert_eq!(result.to_vec_f32().unwrap(), vec![10.0, 200.0, 30.0]);
}

// =============================================================================
// Reduction Operations
// =============================================================================

#[test]
fn test_mean_axis() {
    // [[1, 2, 3], [4, 5, 6]]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    // Mean along axis 0: [2.5, 3.5, 4.5]
    let m0 = a.mean_axis(0, false).unwrap();
    let data = m0.to_vec_f32().unwrap();
    assert_eq!(data.len(), 3);
    assert!((data[0] - 2.5).abs() < 1e-5);
    assert!((data[1] - 3.5).abs() < 1e-5);

    // Mean along axis 1: [2.0, 5.0]
    let m1 = a.mean_axis(1, false).unwrap();
    let data = m1.to_vec_f32().unwrap();
    assert_eq!(data.len(), 2);
    assert!((data[0] - 2.0).abs() < 1e-5);
    assert!((data[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_mean_axis_keepdims() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    let m = a.mean_axis(1, true).unwrap();
    assert_eq!(m.shape().unwrap(), vec![2, 1]); // keepdims preserves rank
}

#[test]
fn test_sum_axis() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    let s = a.sum_axis(1, false).unwrap();
    let data = s.to_vec_f32().unwrap();
    assert!((data[0] - 6.0).abs() < 1e-5);   // 1+2+3
    assert!((data[1] - 15.0).abs() < 1e-5);  // 4+5+6
}

#[test]
fn test_max_axis() {
    let a = Array::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    let m = a.max_axis(1, false).unwrap();
    let data = m.to_vec_f32().unwrap();
    assert_eq!(data, vec![5.0, 6.0]);
}

#[test]
fn test_var_and_std() {
    let a = Array::from_slice(&[2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8], Dtype::Float32).unwrap();

    let v = a.var(0, false).unwrap();
    v.eval().unwrap();
    let var_val: f32 = v.item().unwrap();
    assert!(var_val > 0.0); // Variance should be positive

    let s = a.std(0, false).unwrap();
    s.eval().unwrap();
    let std_val: f32 = s.item().unwrap();
    assert!((std_val - var_val.sqrt()).abs() < 1e-4);
}

// =============================================================================
// Shape & Transformation Operations
// =============================================================================

#[test]
fn test_reshape() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], Dtype::Float32).unwrap();
    let b = a.reshape(&[2, 3]).unwrap();

    assert_eq!(b.shape().unwrap(), vec![2, 3]);
    assert_eq!(b.to_vec_f32().unwrap(), a.to_vec_f32().unwrap()); // Data unchanged
}

#[test]
fn test_reshape_inferred() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], Dtype::Float32).unwrap();
    let b = a.reshape(&[2, -1]).unwrap(); // -1 infers dimension

    assert_eq!(b.shape().unwrap(), vec![2, 3]);
}

#[test]
fn test_transpose() {
    // [2, 3] → transpose → [3, 2]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();
    let b = a.transpose(&[]).unwrap();

    assert_eq!(b.shape().unwrap(), vec![3, 2]);
}

#[test]
fn test_transpose_axes() {
    // [2, 3, 4] → axes [0, 2, 1] → [2, 4, 3]
    let a = Array::from_slice(&vec![0.0f32; 24], &[2, 3, 4], Dtype::Float32).unwrap();
    let b = a.transpose_axes(&[0, 2, 1]).unwrap();

    assert_eq!(b.shape().unwrap(), vec![2, 4, 3]);
}

#[test]
fn test_expand_dims() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();

    let b = a.expand_dims(0).unwrap();
    assert_eq!(b.shape().unwrap(), vec![1, 3]);

    let c = a.expand_dims(1).unwrap();
    assert_eq!(c.shape().unwrap(), vec![3, 1]);

    let d = a.expand_dims(-1).unwrap();
    assert_eq!(d.shape().unwrap(), vec![3, 1]);
}

#[test]
fn test_squeeze() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1], Dtype::Float32).unwrap();

    let b = a.squeeze(None).unwrap();
    assert_eq!(b.shape().unwrap(), vec![3]); // Remove all size-1 dims

    let c = a.squeeze(Some(&[0])).unwrap();
    assert_eq!(c.shape().unwrap(), vec![3, 1]); // Only remove axis 0
}

#[test]
fn test_flatten() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();
    let b = a.flatten().unwrap();

    assert_eq!(b.shape().unwrap(), vec![6]);
    assert_eq!(b.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_broadcast_to() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], Dtype::Float32).unwrap();
    // broadcast_to creates a strided view — multiply_scalar(1.0) forces contiguous copy
    let b = a.broadcast_to(&[4, 3]).unwrap().multiply_scalar(1.0).unwrap();

    assert_eq!(b.shape().unwrap(), vec![4, 3]);
    let data = b.to_vec_f32().unwrap();
    // All 4 rows should be [1, 2, 3]
    for row in 0..4 {
        assert_eq!(data[row * 3], 1.0);
        assert_eq!(data[row * 3 + 1], 2.0);
        assert_eq!(data[row * 3 + 2], 3.0);
    }
}

// =============================================================================
// Slicing & Indexing
// =============================================================================

#[test]
fn test_slice_axis() {
    // [1, 2, 3, 4, 5]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], Dtype::Float32).unwrap();
    let b = a.slice_axis(0, 1, 4).unwrap(); // elements [1:4] → [2, 3, 4]

    assert_eq!(b.to_vec_f32().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_slice_axis_2d() {
    // [[1, 2, 3], [4, 5, 6]]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    // Slice columns 0:2 → [[1, 2], [4, 5]]
    // slice creates a strided view — add_scalar(0.0) forces contiguous copy
    let b = a.slice_axis(1, 0, 2).unwrap().add_scalar(0.0).unwrap();
    assert_eq!(b.shape().unwrap(), vec![2, 2]);
    assert_eq!(b.to_vec_f32().unwrap(), vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_index() {
    // [[1, 2, 3], [4, 5, 6]]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Dtype::Float32).unwrap();

    let row0 = a.index(0).unwrap();
    assert_eq!(row0.shape().unwrap(), vec![3]);
    assert_eq!(row0.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0]);

    let row1 = a.index(1).unwrap();
    assert_eq!(row1.to_vec_f32().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_take() {
    let a = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &[5], Dtype::Float32).unwrap();
    let indices = Array::from_slice(&[0u32, 2, 4], &[3], Dtype::UInt32).unwrap();
    let b = a.take(&indices, 0).unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![10.0, 30.0, 50.0]);
}

#[test]
fn test_concatenate() {
    let a = Array::from_slice(&[1.0f32, 2.0], &[2], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[3.0f32, 4.0, 5.0], &[3], Dtype::Float32).unwrap();
    let c = Array::concatenate(&[&a, &b], 0).unwrap();

    assert_eq!(c.shape().unwrap(), vec![5]);
    assert_eq!(c.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_concatenate_2d() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], Dtype::Float32).unwrap();
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], Dtype::Float32).unwrap();

    // Concat along axis 0: [4, 2]
    let c0 = Array::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(c0.shape().unwrap(), vec![4, 2]);

    // Concat along axis 1: [2, 4]
    let c1 = Array::concatenate(&[&a, &b], 1).unwrap();
    assert_eq!(c1.shape().unwrap(), vec![2, 4]);
}

// =============================================================================
// Dtype & Casting
// =============================================================================

#[test]
fn test_cast() {
    let a = Array::from_slice(&[1.5f32, 2.7, 3.9], &[3], Dtype::Float32).unwrap();
    assert_eq!(a.dtype(), Dtype::Float32);

    let b = a.cast(Dtype::Int32).unwrap();
    assert_eq!(b.dtype(), Dtype::Int32);
}

// =============================================================================
// Clone & Memory
// =============================================================================

#[test]
fn test_clone() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();
    let b = a.clone();

    assert_eq!(a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    assert_eq!(a.shape().unwrap(), b.shape().unwrap());
}

#[test]
fn test_eval() {
    // MLX is lazy — eval forces computation
    let a = Array::from_slice(&[1.0f32, 2.0], &[2], Dtype::Float32).unwrap();
    let b = a.add_scalar(10.0).unwrap();
    b.eval().unwrap();

    assert_eq!(b.to_vec_f32().unwrap(), vec![11.0, 12.0]);
}

#[test]
fn test_debug_display() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();

    // Just verify Debug and Display don't panic
    let _ = format!("{:?}", a);
    let _ = format!("{}", a);
}

// =============================================================================
// Chained Operations
// =============================================================================

#[test]
fn test_chained_arithmetic() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3], Dtype::Float32).unwrap();

    // (a + 1) * 2 - 3
    let result = a.add_scalar(1.0).unwrap()
        .multiply_scalar(2.0).unwrap()
        .subtract_scalar(3.0).unwrap();

    assert_eq!(result.to_vec_f32().unwrap(), vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_normalize_pattern() {
    // Common pattern: (x - mean) / std
    let x = Array::from_slice(&[2.0f32, 4.0, 6.0], &[3], Dtype::Float32).unwrap();
    let mean = x.mean_axis(0, false).unwrap(); // 4.0
    let centered = x.subtract(&mean).unwrap();
    let std_val = centered.square().unwrap().mean_axis(0, false).unwrap().sqrt().unwrap();
    let normalized = centered.divide(&std_val).unwrap();

    let data = normalized.to_vec_f32().unwrap();
    // Should be approximately [-1.22, 0.0, 1.22]
    assert!((data[1]).abs() < 1e-5); // Center should be ~0
    assert!(data[0] < 0.0);          // Below mean → negative
    assert!(data[2] > 0.0);          // Above mean → positive
}
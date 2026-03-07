// mlx/src/ane/mil.rs
//
// MIL (Model Intermediate Language) text generation for ANE programs.
//
// Matrix multiplication is expressed as 1×1 convolution — the ANE's primary
// compute primitive. This yields higher throughput than the native matmul path.
//
// ANE tensor layout is NCHW: [N, Channels, Height, Width].
// Mapping for a linear layer [batch, in_f] → [batch, out_f]:
//   input  → [1, in_f,  1, batch]
//   weight → [out_f, in_f, 1, 1]  (1×1 conv kernel)
//   output → [1, out_f, 1, batch]
//
// Weights and biases are BLOBFILE references; the byte data is passed
// separately to the compiler and baked into the compiled binary.

/// Describes what blobs the generated MIL program needs.
/// The caller must supply these in the same order when passing data to bridge.m.
pub struct LinearMil {
    pub text:     String,
    pub has_bias: bool,
}

/// Generate a MIL program for a linear forward pass (y = x @ W^T + b).
///
/// `in_f`    — number of input features
/// `out_f`   — number of output features
/// `batch`   — batch size (ANE programs are shape-specific)
/// `has_bias` — whether a bias vector is included
pub fn linear_forward_mil(in_f: usize, out_f: usize, batch: usize, has_bias: bool) -> LinearMil {
    let mut m = String::with_capacity(512);

    m.push_str("program 1.0\n");
    m.push_str(&format!(
        "func main(%input: tensor<fp16, [1, {in_f}, 1, {batch}]>) -> \
         (tensor<fp16, [1, {out_f}, 1, {batch}]>) {{\n"
    ));
    m.push_str("  block0():\n");

    // Weight constant — BLOBFILE "weight" maps to the weight byte data
    // Shape [out_f, in_f, 1, 1] is a 1×1 conv kernel computing the dot product
    m.push_str(&format!(
        "    %w: tensor<fp16, [{out_f}, {in_f}, 1, 1]> = \
         const()[val = BLOBFILE(path = \"weight\", offset = 0)]\n"
    ));

    // 1×1 conv: input [1, in_f, 1, batch] × kernel [out_f, in_f, 1, 1]
    //            → output [1, out_f, 1, batch]
    // pad_type="valid" means no padding
    m.push_str(&format!(
        "    %out: tensor<fp16, [1, {out_f}, 1, {batch}]> = conv(\
         x = %input, weight = %w, strides = [1, 1], \
         pad_type = \"valid\", pad = [0, 0, 0, 0], \
         dilations = [1, 1], groups = 1)[]\n"
    ));

    if has_bias {
        // Bias [1, out_f, 1, 1] broadcasts across the batch (spatial) dimension
        m.push_str(&format!(
            "    %b: tensor<fp16, [1, {out_f}, 1, 1]> = \
             const()[val = BLOBFILE(path = \"bias\", offset = 0)]\n"
        ));
        m.push_str(&format!(
            "    %biased: tensor<fp16, [1, {out_f}, 1, {batch}]> = \
             add(x = %out, y = %b)[]\n"
        ));
        m.push_str("    return (%biased)\n");
    } else {
        m.push_str("    return (%out)\n");
    }

    m.push_str("}\n");

    LinearMil { text: m, has_bias }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mil_no_bias_contains_key_ops() {
        let mil = linear_forward_mil(64, 128, 4, false);
        assert!(mil.text.contains("conv("));
        assert!(mil.text.contains("BLOBFILE(path = \"weight\""));
        assert!(!mil.text.contains("bias"));
        assert!(mil.text.contains("[128, 64, 1, 1]")); // kernel shape
        assert!(mil.text.contains("[1, 64, 1, 4]"));   // input shape
        assert!(mil.text.contains("[1, 128, 1, 4]"));  // output shape
    }

    #[test]
    fn test_mil_with_bias_contains_add() {
        let mil = linear_forward_mil(64, 128, 4, true);
        assert!(mil.text.contains("BLOBFILE(path = \"bias\""));
        assert!(mil.text.contains("add("));
        assert!(mil.has_bias);
    }
}

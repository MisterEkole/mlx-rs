// mlx/src/ane/convert.rs
//
// CPU-side fp32 ↔ fp16 conversion for the ANE data path.
//
// IEEE 754 fp32 → fp16 with round-to-nearest-even.
// Handles normals, denormals, ±0, ±Inf, and NaN.

/// Convert a single f32 to its IEEE 754 fp16 bit-pattern.
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits: u32 = v.to_bits();
    let sign: u32 = (bits >> 16) & 0x8000;
    let exp: i32  = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant: u32 = bits & 0x007F_FFFF;

    if exp <= 0 {
        // Underflow: flush to signed zero (denormal fp16 range not handled).
        return sign as u16;
    }

    if exp >= 31 {
        if exp == 128 && mant != 0 {
            return (sign | 0x7E00) as u16; // NaN — quiet NaN
        }
        return (sign | 0x7C00) as u16; // Inf
    }

    // Normal number — shift mantissa down 13 bits, round-to-nearest-even
    let h_mant: u32   = mant >> 13;
    let round_bit: u32 = (mant >> 12) & 1;
    let sticky: u32    = mant & 0x0000_0FFF;
    // Round up if round_bit is set AND (result is odd OR sticky bits are set)
    let h_mant = h_mant + (round_bit & (h_mant & 1 | if sticky != 0 { 1 } else { 0 }));

    (sign | ((exp as u32) << 10) | h_mant) as u16
}

/// Convert a single fp16 bit-pattern back to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign: u32  = (bits as u32 & 0x8000) << 16;
    let exp: u32   = (bits as u32 & 0x7C00) >> 10;
    let mant: u32  = bits as u32 & 0x03FF;

    let result = if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // Denormal fp16 → normalised f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x0400) == 0 { m <<= 1; e += 1; }
            m &= !0x0400;
            sign | ((127u32 - 14 - e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13) // ±Inf or NaN
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13) // Normal
    };

    f32::from_bits(result)
}

/// Convert a slice of fp32 values to packed fp16 bit-patterns.
pub fn f32_slice_to_f16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_f16(v)).collect()
}

/// Convert a slice of packed fp16 bit-patterns back to fp32.
pub fn f16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&v| f16_to_f32(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_f16_roundtrip_common_values() {
        let vals: &[f32] = &[0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001];
        for &v in vals {
            let rt = f16_to_f32(f32_to_f16(v));
            // fp16 has ~3 decimal digits of precision; allow 0.2% relative error
            let rel_err = if v == 0.0 { rt.abs() } else { (rt - v).abs() / v.abs() };
            assert!(rel_err < 0.002, "roundtrip failed for {v}: got {rt}");
        }
    }

    #[test]
    fn test_f32_f16_inf_nan() {
        assert_eq!(f16_to_f32(f32_to_f16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(f32::NEG_INFINITY)), f32::NEG_INFINITY);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_zero() {
        assert_eq!(f32_to_f16(0.0_f32), 0u16);
        assert_eq!(f16_to_f32(0u16), 0.0_f32);
    }
}

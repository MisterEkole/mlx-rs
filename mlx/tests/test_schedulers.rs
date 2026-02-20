// =========================================================================
// Tests
// =========================================================================
use mlx::nn::schedulers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr() {
        let mut s = StepLR::new(0.1, 10, 0.5);
        // First 10 steps: lr = 0.1
        for _ in 0..10 {
            s.step();
        }
        assert!((s.get_lr() - 0.1).abs() < 1e-6);
        // After step 10: lr = 0.05
        s.step();
        assert!((s.get_lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let mut s = ExponentialLR::new(1.0, 0.9);
        s.step();
        assert!((s.get_lr() - 0.9).abs() < 1e-6);
        s.step();
        assert!((s.get_lr() - 0.81).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut s = CosineAnnealingLR::new(0.1, 100, 0.0);
        // At halfway point, lr should be ~0.05
        for _ in 0..50 {
            s.step();
        }
        assert!((s.get_lr() - 0.05).abs() < 0.005);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut s = ReduceLROnPlateau::new(0.1, 0.5, 2, 1e-6, 0.0, PlateauMode::Min);
        // 3 bad epochs (no improvement) should trigger reduction
        s.step_with_metric(1.0); // best = 1.0
        s.step_with_metric(1.0); // bad 1
        s.step_with_metric(1.0); // bad 2
        s.step_with_metric(1.0); // bad 3 -> triggers reduction
        assert!((s.get_lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_one_cycle() {
        let mut s = OneCycleLR::default(0.01, 100);
        // Should start low
        s.step();
        assert!(s.get_lr() < 0.01);
        // At 30% (warmup end), should be near max
        for _ in 1..30 {
            s.step();
        }
        assert!((s.get_lr() - 0.01).abs() < 0.002);
    }

    #[test]
    fn test_warmup_linear() {
        let mut s = WarmupSchedule::new(0.001, 10, 100, 0.0);
        // After 5 warmup steps: lr should be ~0.0005
        for _ in 0..5 {
            s.step();
        }
        assert!((s.get_lr() - 0.0005).abs() < 1e-6);
        // After all 10 warmup steps: lr = 0.001
        for _ in 0..5 {
            s.step();
        }
        assert!((s.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine() {
        let mut s = WarmupCosineSchedule::new(0.001, 10, 110, 0.0);
        // Complete warmup
        for _ in 0..10 {
            s.step();
        }
        assert!((s.get_lr() - 0.001).abs() < 1e-6);
        // At end of cosine phase, should be near eta_min (0)
        for _ in 0..100 {
            s.step();
        }
        assert!(s.get_lr() < 1e-5);
    }

    #[test]
    fn test_reset() {
        let mut s = ExponentialLR::new(1.0, 0.5);
        s.step();
        s.step();
        assert!((s.get_lr() - 0.25).abs() < 1e-6);
        s.reset();
        assert!((s.get_lr() - 1.0).abs() < 1e-6);
    }
}
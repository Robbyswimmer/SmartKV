#!/usr/bin/env python3
"""
Quick test script for baseline validation.

Tests memory measurement and basic functionality without requiring large models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smartkv.experiments.validate_baselines import test_memory_measurement
from smartkv.baselines.evaluator import BaselineEvaluator
from smartkv.utils.logger import create_logger


def test_baseline_memory():
    """Test baseline memory measurements."""
    logger = create_logger(name="test_validation")
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 10: BASELINE VALIDATION TEST")
    logger.info("="*70)
    
    # Test 1: Memory measurement
    logger.info("\nTest 1: Memory Measurement")
    logger.info("-" * 70)
    
    results = test_memory_measurement()
    
    # Verify results
    baseline_names = [r.baseline_name for r in results]
    assert 'FP16' in baseline_names, "FP16 baseline missing"
    assert 'Uniform-INT8' in baseline_names, "INT8 baseline missing"
    assert 'Uniform-INT4' in baseline_names, "INT4 baseline missing"
    assert 'KIVI' in baseline_names, "KIVI baseline missing"
    
    # Test 2: Verify memory ordering
    logger.info("\nTest 2: Memory Ordering Verification")
    logger.info("-" * 70)
    
    memory_map = {r.baseline_name: r.memory_ratio for r in results}
    
    # FP16 should be 1.0
    assert memory_map['FP16'] == 1.0, "FP16 should have ratio 1.0"
    logger.info("✓ FP16 baseline: 1.00x memory (baseline)")
    
    # INT8 should be ~0.5
    assert 0.4 <= memory_map['Uniform-INT8'] <= 0.6, "INT8 should be ~0.5x"
    logger.info(f"✓ INT8 baseline: {memory_map['Uniform-INT8']:.2f}x memory (~50% of FP16)")
    
    # INT4 should be ~0.25
    assert 0.2 <= memory_map['Uniform-INT4'] <= 0.3, "INT4 should be ~0.25x"
    logger.info(f"✓ INT4 baseline: {memory_map['Uniform-INT4']:.2f}x memory (~25% of FP16)")
    
    # KIVI should be less than INT8
    assert memory_map['KIVI'] < memory_map['Uniform-INT8'], "KIVI should use less than INT8"
    logger.info(f"✓ KIVI baseline: {memory_map['KIVI']:.2f}x memory (K=2bit, V=4bit)")
    
    # Test 3: Reconstruction error
    logger.info("\nTest 3: Reconstruction Error Test")
    logger.info("-" * 70)
    
    evaluator = BaselineEvaluator(
        num_layers=4,
        num_heads=8,
        head_dim=64
    )
    
    fp16_error = evaluator.evaluate_reconstruction_error('FP16', num_samples=20)
    int8_error = evaluator.evaluate_reconstruction_error('Uniform-INT8', num_samples=20)
    
    logger.info(f"FP16 reconstruction error: {fp16_error:.6f}")
    logger.info(f"INT8 reconstruction error: {int8_error:.6f}")
    
    assert fp16_error < int8_error, "FP16 should have lower error than INT8"
    logger.info("✓ Reconstruction error ordering correct")
    
    # Test 4: Latency benchmarking
    logger.info("\nTest 4: Latency Benchmarking")
    logger.info("-" * 70)
    
    latencies = {}
    for baseline in ['FP16', 'Uniform-INT8', 'KIVI']:
        latency = evaluator.benchmark_latency(baseline, num_tokens=50, num_iterations=5)
        latencies[baseline] = latency
        logger.info(f"{baseline:<20} Latency: {latency:.2f} ms")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info("✓ All baselines initialized correctly")
    logger.info("✓ Memory ratios match expected values")
    logger.info("✓ Reconstruction errors follow expected ordering")
    logger.info("✓ Latency benchmarking works")
    logger.info("\nPhase 10 validation: PASSED ✅")
    logger.info("="*70)
    
    return True


def main():
    """Run validation tests."""
    try:
        test_baseline_memory()
        print("\n✅ All validation tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

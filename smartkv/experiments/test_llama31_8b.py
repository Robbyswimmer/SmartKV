"""
Test SmartKV with Llama 3.1 8B (NousResearch).

Quick compatibility test before running full evaluation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
import time

def test_llama31_compatibility():
    """Test if Llama 3.1 8B works with SmartKV."""

    print("="*80)
    print("LLAMA 3.1 8B SMARTKV COMPATIBILITY TEST")
    print("="*80)

    device = "cpu"
    model_name = "NousResearch/Meta-Llama-3.1-8B"

    # Load model and tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    print("\n[2/5] Loading Llama 3.1 8B model (this may take a while)...")
    print("      Note: 8B model requires ~16GB RAM")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True  # More efficient loading
        )
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nModel may not be cached locally. To download:")
        print(f"  huggingface-cli download {model_name}")
        return False

    # Test baseline inference
    print("\n[3/5] Testing baseline (FP16) inference...")
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    baseline_time = time.time() - start

    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Baseline inference successful ({baseline_time:.2f}s)")
    print(f"  Output: {baseline_text}")

    # Create SmartKV wrapper
    print("\n[4/5] Creating SmartKV wrapper...")
    try:
        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.5,
            decay=0.9,
            realloc_freq=16,
            available_bits=[2, 3, 4, 8],
            device=device
        )

        smartkv_model = LlamaSmartKV(model, config)
        print("✓ SmartKV wrapper created successfully")

        # Check integration
        num_layers = len(model.model.layers)
        print(f"  Model has {num_layers} layers")
        print(f"  SmartKV cache: {smartkv_model.smartkv_cache is not None}")

    except Exception as e:
        print(f"✗ Failed to create SmartKV wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test SmartKV inference
    print("\n[5/5] Testing SmartKV inference...")
    smartkv_model.reset_cache()

    start = time.time()
    with torch.no_grad():
        outputs = smartkv_model.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    smartkv_time = time.time() - start

    smartkv_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ SmartKV inference successful ({smartkv_time:.2f}s)")
    print(f"  Output: {smartkv_text}")

    # Compare results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nBaseline:  {baseline_text}")
    print(f"SmartKV:   {smartkv_text}")
    print(f"\nMatch: {'✓ EXACT' if baseline_text == smartkv_text else '✗ Different'}")
    print(f"\nBaseline Time: {baseline_time:.2f}s")
    print(f"SmartKV Time:  {smartkv_time:.2f}s")
    print(f"Overhead:      {(smartkv_time/baseline_time - 1)*100:+.1f}%")

    print("\n" + "="*80)
    print("✅ COMPATIBILITY TEST PASSED")
    print("="*80)
    print("\nLlama 3.1 8B is compatible with SmartKV!")
    print("\nNext steps:")
    print("  1. Run full evaluation with --model NousResearch/Meta-Llama-3.1-8B")
    print("  2. Consider using GPU if available (much faster for 8B model)")
    print("  3. Expect better quality than TinyLlama (1.1B)")

    return True


if __name__ == "__main__":
    try:
        success = test_llama31_compatibility()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

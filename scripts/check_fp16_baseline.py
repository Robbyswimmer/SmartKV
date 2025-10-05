"""Quick sanity script to verify FP16 baseline outputs for Llama-like models."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In machine learning, attention mechanisms are used to",
    "Write a Python function that returns the n-th Fibonacci number",
    "Complete the quote: Four score and seven years ago",
]


def format_prompts(tokenizer, prompts):
    """Apply chat template when available."""
    formatted = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(chat)
                continue
            except Exception:
                pass
        formatted.append(prompt)
    return formatted


def generate_responses(model, tokenizer, prompts, max_new_tokens, device, do_sample, temperature, top_p):
    results = []
    formatted_prompts = format_prompts(tokenizer, prompts)

    for raw_prompt, formatted in zip(prompts, formatted_prompts):
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"prompt": raw_prompt, "formatted_prompt": formatted, "output": text})

    return results


def main():
    parser = argparse.ArgumentParser(description="Sanity-check FP16 baseline outputs.")
    parser.add_argument("model_name", help="HuggingFace repo ID or local path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to load the model with (default: float16).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation length (default: 64).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="Optional JSON file with a list of prompts.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (only used when --sample).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus threshold (only used when --sample).",
    )

    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"Loading tokenizer for {args.model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({args.dtype})…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto" if args.device != "cpu" else None,
    )
    model.to(args.device)
    model.eval()

    print("Generating responses…")
    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Baseline Outputs ===")
    for item in responses:
        print(f"\nPrompt: {item['prompt']}")
        if item['formatted_prompt'] != item['prompt']:
            print(f"Formatted Prompt: {item['formatted_prompt']}")
        print("Response:\n" + item['output'])


if __name__ == "__main__":
    main()

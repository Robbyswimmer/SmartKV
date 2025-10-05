"""
Long context evaluation for SmartKV using Romeo and Juliet.

Tests SmartKV's ability to handle long contexts (30K+ tokens) by:
1. Loading full Romeo and Juliet text as context
2. Asking specific questions about the text
3. Comparing FP16 vs SmartKV across multiple memory budgets
4. Measuring latency, memory usage, and answer quality
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import time

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
from smartkv.utils.logger import create_logger


# Questions about Romeo and Juliet to test retrieval
TEST_QUESTIONS = [
    {
        "question": "In which city does Romeo and Juliet take place?",
        "expected_answer": "Verona"
    },
    {
        "question": "Who kills Tybalt?",
        "expected_answer": "Romeo"
    },
    {
        "question": "What is the name of Juliet's cousin who is killed?",
        "expected_answer": "Tybalt"
    },
    {
        "question": "Who marries Romeo and Juliet?",
        "expected_answer": "Friar Lawrence"
    },
    {
        "question": "How does Juliet die?",
        "expected_answer": "stabs herself"
    },
    {
        "question": "What families are Romeo and Juliet from?",
        "expected_answer": "Montague and Capulet"
    },
    {
        "question": "Who is Romeo's best friend?",
        "expected_answer": "Mercutio"
    },
    {
        "question": "What poison does Romeo drink?",
        "expected_answer": "apothecary's poison"
    },
]


class LongContextTester:
    """Test SmartKV on long context tasks."""

    def __init__(
        self,
        model_name: str,
        context_file: str,
        device: str = "cpu",
        output_dir: str = "experiments/long_context",
        use_fused_cpu: bool = False
    ):
        self.model_name = model_name
        self.context_file = Path(context_file)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_fused_cpu = use_fused_cpu

        self.logger = create_logger(
            name="long_context_test",
            log_file=str(self.output_dir / "test.log")
        )

        self.tokenizer = None
        self.base_model = None
        self.context_text = None
        self.context_tokens = None

    def load_context(self, max_context_tokens: int = None) -> bool:
        """Load the long context file."""
        try:
            self.logger.info(f"Loading context from {self.context_file}")
            self.context_text = self.context_file.read_text(encoding='utf-8')

            # Tokenize to get length
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            tokens = self.tokenizer.encode(self.context_text)
            self.context_tokens = len(tokens)

            # Truncate if needed
            if max_context_tokens and self.context_tokens > max_context_tokens:
                self.logger.info(f"Truncating context from {self.context_tokens} to {max_context_tokens} tokens")
                tokens = tokens[:max_context_tokens]
                self.context_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                self.context_tokens = len(tokens)

            self.logger.info(f"Context loaded: {len(self.context_text)} chars, {self.context_tokens} tokens")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load context: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self) -> bool:
        """Load the base model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("transformers not available")
            return False

        try:
            self.logger.info(f"Loading model: {self.model_name}")

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            self.base_model = self.base_model.to(self.device)
            self.base_model.eval()

            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_smartkv_model(
        self,
        memory_budget: float
    ) -> LlamaSmartKV:
        """Create SmartKV model with given budget."""
        config = SmartKVConfig(
            enabled=True,
            memory_budget=memory_budget,
            decay=0.9,
            realloc_freq=16,
            available_bits=[2, 3, 4, 8],
            device=self.device
        )
        model = LlamaSmartKV(self.base_model, config)
        model.set_use_fused_cpu(self.use_fused_cpu)
        return model

    def ask_question(
        self,
        question: str,
        max_new_tokens: int = 50,
        use_smartkv: bool = False,
        smartkv_model: LlamaSmartKV = None
    ) -> Dict[str, Any]:
        """Ask a question with the full context."""

        # Build prompt: context + question
        prompt_body = (
            f"{self.context_text}\n\n"
            "Based on the text above, respond factually in a single short sentence.\n"
            f"Question: {question}\nAnswer:"
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "Provide concise answers grounded in the preceding document."},
                        {"role": "user", "content": prompt_body},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = prompt_body
        else:
            prompt = prompt_body

        # Select model
        model = smartkv_model.model if use_smartkv and smartkv_model else self.base_model

        # Reset cache if using SmartKV
        if use_smartkv and smartkv_model:
            smartkv_model.reset_cache()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32768  # Support long context
        ).to(self.device)

        input_length = len(inputs['input_ids'][0])

        # Measure latency
        start_time = time.perf_counter()

        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id,
            'use_cache': True,
        }
        if getattr(self.tokenizer, 'eos_token_id', None) is not None:
            generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode answer
        generated_ids = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if answer:
            answer = answer.split('\n')[0]

        result = {
            'question': question,
            'answer': answer.strip(),
            'latency_ms': latency_ms,
            'input_tokens': input_length,
            'output_tokens': len(outputs[0]) - input_length
        }

        # Get precision distribution if using SmartKV
        if use_smartkv and smartkv_model and hasattr(smartkv_model, 'smartkv_cache'):
            cache = smartkv_model.smartkv_cache
            if cache is not None and hasattr(cache, 'precision_map'):
                # Get precision distribution from cache
                precision_dist = {}
                for bits in cache.precision_map.values():
                    precision_dist[bits] = precision_dist.get(bits, 0) + 1

                if precision_dist:
                    result['precision_distribution'] = precision_dist
                    # Log distribution for debugging
                    total = sum(precision_dist.values())
                    dist_str = ', '.join([f"{bits}bit: {count} ({100*count/total:.1f}%)"
                                         for bits, count in sorted(precision_dist.items(), reverse=True)])
                    if hasattr(self, 'logger'):
                        self.logger.info(f"  Precision dist: {dist_str}")

        return result

    def run_test(
        self,
        memory_budgets: List[float] = [0.5, 0.35, 0.25],
        max_new_tokens: int = 50,
        num_questions: int = None,
        max_context_tokens: int = None
    ) -> Dict[str, Any]:
        """Run long context test across multiple budgets."""

        from smartkv.experiments._thread_utils import configure_threads
        threads = configure_threads()
        self.logger.info(f"Configured inference threads: {threads}")
        if self.use_fused_cpu:
            self.logger.info("Using fused CPU streaming attention simulation")
        self.logger.info("="*80)
        self.logger.info("LONG CONTEXT TEST: Romeo and Juliet")
        self.logger.info("="*80)

        # Load context
        if not self.load_context(max_context_tokens=max_context_tokens):
            return {'error': 'Failed to load context'}

        # Load model
        if not self.load_model():
            return {'error': 'Failed to load model'}

        # Select questions to test
        questions = TEST_QUESTIONS[:num_questions] if num_questions else TEST_QUESTIONS

        results = {
            'model': self.model_name,
            'context_file': str(self.context_file),
            'context_chars': len(self.context_text),
            'context_tokens': self.context_tokens,
            'num_questions': len(questions),
            'budgets_tested': memory_budgets,
            'fp16_baseline': {},
            'smartkv_results': {}
        }

        # Test FP16 baseline
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING FP16 BASELINE")
        self.logger.info("="*80)

        fp16_answers = []
        for i, q_data in enumerate(questions):
            question = q_data['question']
            self.logger.info(f"\n[{i+1}/{len(questions)}] {question}")

            result = self.ask_question(
                question,
                max_new_tokens=max_new_tokens,
                use_smartkv=False
            )
            fp16_answers.append(result)

            self.logger.info(f"  Answer: {result['answer'][:100]}...")
            self.logger.info(f"  Latency: {result['latency_ms']:.2f} ms")

        results['fp16_baseline'] = {
            'answers': fp16_answers,
            'avg_latency_ms': sum(r['latency_ms'] for r in fp16_answers) / len(fp16_answers)
        }

        # Test SmartKV with different budgets
        for budget in memory_budgets:
            self.logger.info("\n" + "="*80)
            self.logger.info(f"TESTING SMARTKV (budget={budget:.0%})")
            self.logger.info("="*80)

            smartkv_model = self.create_smartkv_model(budget)
            smartkv_answers = []

            for i, q_data in enumerate(questions):
                question = q_data['question']
                self.logger.info(f"\n[{i+1}/{len(questions)}] {question}")

                result = self.ask_question(
                    question,
                    max_new_tokens=max_new_tokens,
                    use_smartkv=True,
                    smartkv_model=smartkv_model
                )
                smartkv_answers.append(result)

                self.logger.info(f"  Answer: {result['answer'][:100]}...")
                self.logger.info(f"  Latency: {result['latency_ms']:.2f} ms")
                if 'precision_distribution' in result:
                    self.logger.info(f"  Precision: {result['precision_distribution']}")

            avg_latency = sum(r['latency_ms'] for r in smartkv_answers) / len(smartkv_answers)
            overhead_pct = (avg_latency / results['fp16_baseline']['avg_latency_ms'] - 1) * 100

            results['smartkv_results'][f'budget_{int(budget*100)}'] = {
                'budget': budget,
                'answers': smartkv_answers,
                'avg_latency_ms': avg_latency,
                'latency_overhead_pct': overhead_pct
            }

        # Save results
        results_file = self.output_dir / "long_context_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nResults saved to {results_file}")

        # Print summary
        self.print_summary(results, questions)

        return results

    def print_summary(self, results: Dict[str, Any], questions: List[Dict[str, str]]):
        """Print test summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("SUMMARY")
        self.logger.info("="*80)

        self.logger.info(f"\nContext: {results['context_tokens']} tokens")
        self.logger.info(f"Questions: {results['num_questions']}")

        self.logger.info(f"\n--- FP16 Baseline ---")
        self.logger.info(f"Avg Latency: {results['fp16_baseline']['avg_latency_ms']:.2f} ms")

        self.logger.info(f"\n--- SmartKV Results ---")
        for budget_key, budget_results in results['smartkv_results'].items():
            budget = budget_results['budget']
            self.logger.info(f"\nBudget {budget:.0%}:")
            self.logger.info(f"  Avg Latency:  {budget_results['avg_latency_ms']:.2f} ms")
            self.logger.info(f"  Overhead:     {budget_results['latency_overhead_pct']:+.1f}%")

        # Compare answers
        self.logger.info(f"\n--- Answer Comparison ---")
        for i, q_data in enumerate(questions):
            self.logger.info(f"\n{q_data['question']}")
            self.logger.info(f"  Expected: {q_data['expected_answer']}")
            self.logger.info(f"  FP16:     {results['fp16_baseline']['answers'][i]['answer'][:80]}")

            for budget_key, budget_results in results['smartkv_results'].items():
                budget = budget_results['budget']
                answer = budget_results['answers'][i]['answer'][:80]
                self.logger.info(f"  {budget:.0%}:      {answer}")

        self.logger.info("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Long context SmartKV test")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default="data/romeo_juliet.txt",
        help="Path to context file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.5, 0.35, 0.25],
        help="Memory budgets to test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens for answers"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to test (default: all)"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum context tokens (truncate if longer)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/long_context",
        help="Output directory"
    )
    parser.add_argument(
        "--use-fused-cpu",
        action="store_true",
        help="Use streaming CPU fused attention (simulated)"
    )

    args = parser.parse_args()

    # Run test
    tester = LongContextTester(
        model_name=args.model,
        context_file=args.context_file,
        device=args.device,
        output_dir=args.output_dir,
        use_fused_cpu=args.use_fused_cpu
    )

    results = tester.run_test(
        memory_budgets=args.budgets,
        max_new_tokens=args.max_tokens,
        num_questions=args.num_questions,
        max_context_tokens=args.max_context_tokens
    )

    if 'error' not in results:
        print("\n" + "="*80)
        print("LONG CONTEXT TEST COMPLETED")
        print("="*80)
        print(f"Context: {results['context_tokens']} tokens")
        print(f"Questions: {results['num_questions']}")
        print(f"\nResults saved to: {args.output_dir}/long_context_results.json")

        print(f"\nFP16 Baseline: {results['fp16_baseline']['avg_latency_ms']:.2f} ms")
        for budget_key, budget_results in results['smartkv_results'].items():
            budget = budget_results['budget']
            print(f"SmartKV {budget:.0%}:   {budget_results['avg_latency_ms']:.2f} ms ({budget_results['latency_overhead_pct']:+.1f}%)")
    else:
        print(f"\nTest failed: {results['error']}")


if __name__ == "__main__":
    main()

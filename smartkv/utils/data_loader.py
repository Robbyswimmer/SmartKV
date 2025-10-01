"""
Data loaders for evaluation datasets.

Implements loaders for:
- LongBench: Multi-task long-context benchmark
- RULER: Retrieval Understanding over Long contexts
- Needle-in-Haystack: Classic long-context retrieval test
"""

import torch
import random
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class DataSample:
    """Single data sample for evaluation."""
    
    text: str
    question: Optional[str] = None
    answer: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __len__(self) -> int:
        """Return text length in characters."""
        return len(self.text)


class LongBenchLoader:
    """
    Loader for LongBench dataset.
    
    LongBench is a multi-task benchmark for long-context understanding,
    covering tasks like summarization, QA, and code completion.
    """
    
    def __init__(
        self,
        task: str = "narrativeqa",
        split: str = "test",
        max_samples: Optional[int] = None
    ):
        """
        Initialize LongBench loader.
        
        Args:
            task: Task name (e.g., 'narrativeqa', 'qasper', 'multifieldqa_en')
            split: Dataset split ('test' or 'validation')
            max_samples: Maximum number of samples to load
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required for LongBench")
        
        self.task = task
        self.split = split
        self.max_samples = max_samples
        self.samples: List[DataSample] = []
        
    def load(self) -> List[DataSample]:
        """
        Load LongBench dataset.
        
        Returns:
            List of DataSample objects
        """
        # Load from HuggingFace
        try:
            dataset = load_dataset("THUDM/LongBench", self.task, split=self.split)
        except Exception as e:
            print(f"Warning: Could not load LongBench from HuggingFace: {e}")
            print("Creating mock samples for testing...")
            return self._create_mock_samples()
        
        # Convert to DataSample format
        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            
            sample = DataSample(
                text=item.get('context', ''),
                question=item.get('input', ''),
                answer=item.get('answers', [''])[0] if isinstance(item.get('answers'), list) else item.get('answers', ''),
                metadata={
                    'task': self.task,
                    'length': item.get('length', 0),
                    'all_answers': item.get('answers', [])
                }
            )
            self.samples.append(sample)
        
        return self.samples
    
    def _create_mock_samples(self, n: int = 10) -> List[DataSample]:
        """Create mock samples for testing."""
        mock_samples = []
        for i in range(n):
            sample = DataSample(
                text=f"This is a long context document {i}. " * 100,
                question=f"What is the answer to question {i}?",
                answer=f"Answer {i}",
                metadata={'task': self.task, 'mock': True}
            )
            mock_samples.append(sample)
        return mock_samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.samples:
            self.load()
        
        lengths = [len(s.text) for s in self.samples]
        
        return {
            'num_samples': len(self.samples),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'task': self.task,
        }


class RULERLoader:
    """
    Loader for RULER dataset.
    
    RULER evaluates retrieval and reasoning over long contexts
    with synthetic tasks at various context lengths.
    """
    
    def __init__(
        self,
        context_length: int = 4096,
        num_samples: int = 100,
        task_type: str = "niah"  # needle-in-a-haystack
    ):
        """
        Initialize RULER loader.
        
        Args:
            context_length: Target context length
            num_samples: Number of samples to generate
            task_type: Type of task ('niah', 'variable_tracking', 'common_words')
        """
        self.context_length = context_length
        self.num_samples = num_samples
        self.task_type = task_type
        self.samples: List[DataSample] = []
    
    def load(self) -> List[DataSample]:
        """
        Load/generate RULER samples.
        
        Returns:
            List of DataSample objects
        """
        # For now, generate synthetic samples similar to RULER
        if self.task_type == "niah":
            return self._generate_niah_samples()
        elif self.task_type == "variable_tracking":
            return self._generate_variable_tracking()
        else:
            return self._generate_common_words()
    
    def _generate_niah_samples(self) -> List[DataSample]:
        """Generate needle-in-a-haystack samples."""
        for i in range(self.num_samples):
            # Generate haystack text
            haystack = self._generate_filler_text(self.context_length - 100)
            
            # Insert needle at random position
            needle = f"The magic number is {i * 42}."
            position = random.randint(0, len(haystack))
            text = haystack[:position] + " " + needle + " " + haystack[position:]
            
            sample = DataSample(
                text=text,
                question=f"What is the magic number?",
                answer=str(i * 42),
                metadata={
                    'task': 'ruler_niah',
                    'needle_position': position,
                    'context_length': len(text)
                }
            )
            self.samples.append(sample)
        
        return self.samples
    
    def _generate_variable_tracking(self) -> List[DataSample]:
        """Generate variable tracking samples."""
        for i in range(self.num_samples):
            # Track a variable through context
            var_name = f"var_{i}"
            var_value = random.randint(0, 1000)
            
            text = f"Initialize {var_name} = {var_value}. "
            text += self._generate_filler_text(self.context_length - 200)
            text += f" Remember that {var_name} was set to {var_value}."
            
            sample = DataSample(
                text=text,
                question=f"What is the value of {var_name}?",
                answer=str(var_value),
                metadata={'task': 'ruler_variable', 'variable': var_name}
            )
            self.samples.append(sample)
        
        return self.samples
    
    def _generate_common_words(self) -> List[DataSample]:
        """Generate common words frequency counting."""
        words = ["apple", "banana", "orange", "grape", "mango"]
        
        for i in range(self.num_samples):
            target_word = random.choice(words)
            count = random.randint(5, 15)
            
            # Generate text with specific word count
            text_parts = []
            for _ in range(count):
                text_parts.append(target_word)
                text_parts.append(self._generate_filler_text(50))
            
            text = " ".join(text_parts)
            
            sample = DataSample(
                text=text,
                question=f"How many times does the word '{target_word}' appear?",
                answer=str(count),
                metadata={'task': 'ruler_common_words', 'target_word': target_word}
            )
            self.samples.append(sample)
        
        return self.samples
    
    def _generate_filler_text(self, target_length: int) -> str:
        """Generate filler text of approximately target length."""
        filler_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "In a galaxy far, far away, there was a planet called Earth.",
            "The rain in Spain stays mainly in the plain.",
            "To be or not to be, that is the question.",
        ]
        
        text = []
        current_length = 0
        
        while current_length < target_length:
            sentence = random.choice(filler_sentences)
            text.append(sentence)
            current_length += len(sentence) + 1
        
        return " ".join(text)


class NeedleInHaystackGenerator:
    """
    Generator for Needle-in-Haystack evaluation.
    
    Classic test for long-context retrieval: hide a "needle" (fact)
    in a "haystack" (long document) and test retrieval.
    """
    
    def __init__(
        self,
        haystack_lengths: List[int] = [1000, 2000, 4000, 8000, 16000],
        needle_depths: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        num_samples_per_config: int = 10
    ):
        """
        Initialize Needle-in-Haystack generator.
        
        Args:
            haystack_lengths: List of haystack lengths to test
            needle_depths: List of needle positions (0.0 = start, 1.0 = end)
            num_samples_per_config: Samples per length/depth combination
        """
        self.haystack_lengths = haystack_lengths
        self.needle_depths = needle_depths
        self.num_samples_per_config = num_samples_per_config
        self.samples: List[DataSample] = []
    
    def generate(self) -> List[DataSample]:
        """
        Generate needle-in-haystack samples.
        
        Returns:
            List of DataSample objects
        """
        self.samples = []
        
        for length in self.haystack_lengths:
            for depth in self.needle_depths:
                for i in range(self.num_samples_per_config):
                    sample = self._generate_single_sample(length, depth, i)
                    self.samples.append(sample)
        
        return self.samples
    
    def _generate_single_sample(
        self,
        haystack_length: int,
        needle_depth: float,
        sample_id: int
    ) -> DataSample:
        """Generate a single needle-in-haystack sample."""
        # Create needle (the fact to retrieve)
        needle_fact = f"The secret passcode is: SMARTKV{sample_id:04d}"
        
        # Generate haystack
        haystack = self._generate_haystack(haystack_length)
        
        # Insert needle at specified depth
        insertion_point = int(len(haystack) * needle_depth)
        text = haystack[:insertion_point] + " " + needle_fact + " " + haystack[insertion_point:]
        
        sample = DataSample(
            text=text,
            question="What is the secret passcode?",
            answer=f"SMARTKV{sample_id:04d}",
            metadata={
                'task': 'needle_in_haystack',
                'haystack_length': haystack_length,
                'needle_depth': needle_depth,
                'actual_length': len(text),
                'needle_position': insertion_point,
            }
        )
        
        return sample
    
    def _generate_haystack(self, target_length: int) -> str:
        """Generate haystack text."""
        # Use Paul Graham essays as inspiration for filler
        essay_snippets = [
            "When I was a kid, I thought the best thing about being an adult would be that you could eat ice cream whenever you wanted.",
            "The most important thing is to keep working. Don't stop. Keep going.",
            "If you want to learn something, read about it. If you want to understand it, write about it.",
            "The way to get startup ideas is not to try to think of startup ideas.",
            "Unless you're very lucky, your first few ideas will be bad.",
            "The best writing is rewriting. The best ideas are refined ideas.",
        ]
        
        text = []
        current_length = 0
        
        while current_length < target_length:
            snippet = random.choice(essay_snippets)
            text.append(snippet)
            current_length += len(snippet) + 1
        
        return " ".join(text)
    
    def get_configurations(self) -> List[Dict[str, Any]]:
        """Get all test configurations."""
        configs = []
        for length in self.haystack_lengths:
            for depth in self.needle_depths:
                configs.append({
                    'length': length,
                    'depth': depth,
                    'num_samples': self.num_samples_per_config
                })
        return configs


def load_longbench(
    task: str = "narrativeqa",
    max_samples: int = 100
) -> List[DataSample]:
    """
    Convenience function to load LongBench.
    
    Args:
        task: Task name
        max_samples: Maximum samples to load
    
    Returns:
        List of DataSample objects
    """
    loader = LongBenchLoader(task=task, max_samples=max_samples)
    return loader.load()


def load_ruler(
    context_length: int = 4096,
    num_samples: int = 100
) -> List[DataSample]:
    """
    Convenience function to load RULER.
    
    Args:
        context_length: Target context length
        num_samples: Number of samples
    
    Returns:
        List of DataSample objects
    """
    loader = RULERLoader(context_length=context_length, num_samples=num_samples)
    return loader.load()


def generate_needle_in_haystack(
    haystack_lengths: List[int] = [1000, 4000, 8000],
    needle_depths: List[float] = [0.0, 0.5, 1.0]
) -> List[DataSample]:
    """
    Convenience function to generate Needle-in-Haystack.
    
    Args:
        haystack_lengths: List of haystack lengths
        needle_depths: List of needle positions
    
    Returns:
        List of DataSample objects
    """
    generator = NeedleInHaystackGenerator(
        haystack_lengths=haystack_lengths,
        needle_depths=needle_depths
    )
    return generator.generate()

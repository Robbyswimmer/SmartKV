"""
Evaluation metrics for SmartKV.

Implements metrics for assessing:
- Accuracy (exact match, F1)
- Perplexity
- Rouge-L
- Memory efficiency
- Latency
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import re

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sklearn.metrics import f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """
    Compute exact match score.
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        normalize: Whether to normalize text before comparison
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)
    
    return 1.0 if prediction == reference else 0.0


def f1_token_score(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Count common tokens
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    common = pred_counter & ref_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def rouge_l_score(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute Rouge-L score.
    
    Args:
        prediction: Predicted text
        reference: Reference text
    
    Returns:
        Dict with precision, recall, and F1
    """
    if not ROUGE_AVAILABLE:
        # Fallback to simple F1 if rouge_score not available
        f1 = f1_token_score(prediction, reference)
        return {'precision': f1, 'recall': f1, 'fmeasure': f1}
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'precision': scores['rougeL'].precision,
        'recall': scores['rougeL'].recall,
        'fmeasure': scores['rougeL'].fmeasure,
    }


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity from model outputs.
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Ground truth labels [batch, seq_len]
        ignore_index: Label index to ignore
    
    Returns:
        Perplexity value
    """
    # Flatten
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute accuracy metrics for a batch of predictions.
    
    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
    
    Returns:
        Dict with various accuracy metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    # Exact match
    em_scores = [exact_match(pred, ref) for pred, ref in zip(predictions, references)]
    
    # F1 scores
    f1_scores = [f1_token_score(pred, ref) for pred, ref in zip(predictions, references)]
    
    # Rouge-L
    rouge_scores = [rouge_l_score(pred, ref) for pred, ref in zip(predictions, references)]
    
    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores),
        'rouge_l_precision': np.mean([s['precision'] for s in rouge_scores]),
        'rouge_l_recall': np.mean([s['recall'] for s in rouge_scores]),
        'rouge_l_f1': np.mean([s['fmeasure'] for s in rouge_scores]),
        'num_samples': len(predictions),
    }


def compute_retrieval_accuracy(
    predictions: List[str],
    references: List[str],
    threshold: float = 0.8
) -> Dict[str, float]:
    """
    Compute retrieval accuracy (for needle-in-haystack).
    
    Args:
        predictions: List of retrieved answers
        references: List of ground truth answers
        threshold: F1 threshold for considering retrieval successful
    
    Returns:
        Dict with retrieval metrics
    """
    correct = 0
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        f1 = f1_token_score(pred, ref)
        f1_scores.append(f1)
        
        if f1 >= threshold:
            correct += 1
    
    return {
        'retrieval_accuracy': correct / len(predictions) if predictions else 0.0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
        'perfect_retrieval': sum(1 for f1 in f1_scores if f1 == 1.0) / len(f1_scores) if f1_scores else 0.0,
        'num_samples': len(predictions),
    }


def compute_memory_efficiency(
    memory_used: float,
    memory_baseline: float
) -> Dict[str, float]:
    """
    Compute memory efficiency metrics.
    
    Args:
        memory_used: Memory used by method (in bytes or ratio)
        memory_baseline: Baseline memory (FP16)
    
    Returns:
        Dict with efficiency metrics
    """
    ratio = memory_used / memory_baseline if memory_baseline > 0 else 1.0
    reduction = (1.0 - ratio) * 100  # Percentage reduction
    
    return {
        'memory_ratio': ratio,
        'memory_reduction_pct': reduction,
        'compression_rate': 1.0 / ratio if ratio > 0 else 1.0,
    }


def compute_latency_metrics(
    latencies: List[float]
) -> Dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latencies: List of latency measurements (in milliseconds)
    
    Returns:
        Dict with latency statistics
    """
    if not latencies:
        return {
            'mean_latency': 0.0,
            'median_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0,
            'min_latency': 0.0,
            'max_latency': 0.0,
        }
    
    latencies = np.array(latencies)
    
    return {
        'mean_latency': float(np.mean(latencies)),
        'median_latency': float(np.median(latencies)),
        'p95_latency': float(np.percentile(latencies, 95)),
        'p99_latency': float(np.percentile(latencies, 99)),
        'min_latency': float(np.min(latencies)),
        'max_latency': float(np.max(latencies)),
        'std_latency': float(np.std(latencies)),
    }


class MetricsTracker:
    """
    Track metrics over multiple evaluations.
    
    Accumulates metrics and computes aggregated statistics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """
        Add a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Optional metadata for this measurement
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        if metadata:
            if metric_name not in self.metadata:
                self.metadata[metric_name] = []
            self.metadata[metric_name].append(metadata)
    
    def add_batch(self, metrics_dict: Dict[str, float]):
        """
        Add a batch of metrics.
        
        Args:
            metrics_dict: Dict of metric_name -> value
        """
        for name, value in metrics_dict.items():
            self.add(name, value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dict of metric_name -> statistics
        """
        summary = {}
        
        for name, values in self.metrics.items():
            if not values:
                continue
            
            summary[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values),
            }
        
        return summary
    
    def get_metric(self, metric_name: str) -> Optional[List[float]]:
        """Get all values for a specific metric."""
        return self.metrics.get(metric_name)
    
    def get_mean(self, metric_name: str) -> Optional[float]:
        """Get mean value for a specific metric."""
        values = self.metrics.get(metric_name)
        if values:
            return float(np.mean(values))
        return None
    
    def clear(self):
        """Clear all tracked metrics."""
        self.metrics.clear()
        self.metadata.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        num_metrics = len(self.metrics)
        total_samples = sum(len(v) for v in self.metrics.values())
        return f"MetricsTracker(metrics={num_metrics}, samples={total_samples})"


def evaluate_sample(
    prediction: str,
    reference: str,
    compute_rouge: bool = True
) -> Dict[str, float]:
    """
    Evaluate a single prediction against reference.
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
        compute_rouge: Whether to compute Rouge scores
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {
        'exact_match': exact_match(prediction, reference),
        'f1': f1_token_score(prediction, reference),
    }
    
    if compute_rouge:
        rouge = rouge_l_score(prediction, reference)
        metrics['rouge_l_f1'] = rouge['fmeasure']
        metrics['rouge_l_precision'] = rouge['precision']
        metrics['rouge_l_recall'] = rouge['recall']
    
    return metrics


def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple samples.
    
    Args:
        metrics_list: List of metric dicts
    
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    # Collect all metric names
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    # Compute mean for each metric
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    aggregated['num_samples'] = len(metrics_list)
    
    return aggregated

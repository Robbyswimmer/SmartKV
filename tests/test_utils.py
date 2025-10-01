"""
Unit tests for utility modules.

Tests data loaders, metrics, and logging utilities.
"""

import torch
import pytest
import tempfile
import json
from pathlib import Path
from smartkv.utils.data_loader import (
    DataSample,
    LongBenchLoader,
    RULERLoader,
    NeedleInHaystackGenerator,
    load_longbench,
    load_ruler,
    generate_needle_in_haystack
)
from smartkv.utils.metrics import (
    normalize_text,
    exact_match,
    f1_token_score,
    rouge_l_score,
    compute_accuracy,
    compute_retrieval_accuracy,
    compute_memory_efficiency,
    compute_latency_metrics,
    MetricsTracker,
    evaluate_sample,
    aggregate_metrics
)
from smartkv.utils.logger import (
    SmartKVLogger,
    ExperimentTracker,
    MemoryProfiler,
    create_logger,
    get_logger
)


class TestDataSample:
    """Test DataSample dataclass."""
    
    def test_creation(self):
        """Test creating a data sample."""
        sample = DataSample(
            text="This is a test.",
            question="What is this?",
            answer="A test"
        )
        
        assert sample.text == "This is a test."
        assert sample.question == "What is this?"
        assert sample.answer == "A test"
    
    def test_length(self):
        """Test sample length."""
        sample = DataSample(text="Hello world")
        assert len(sample) == 11


class TestLongBenchLoader:
    """Test LongBench dataset loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = LongBenchLoader(task="narrativeqa", max_samples=10)
        
        assert loader.task == "narrativeqa"
        assert loader.max_samples == 10
    
    def test_load_mock_samples(self):
        """Test loading mock samples."""
        loader = LongBenchLoader(task="test_task", max_samples=5)
        samples = loader._create_mock_samples(n=5)
        
        assert len(samples) == 5
        assert all(isinstance(s, DataSample) for s in samples)
    
    def test_get_statistics(self):
        """Test getting dataset statistics."""
        loader = LongBenchLoader(task="test_task", max_samples=5)
        loader.samples = loader._create_mock_samples(n=5)
        
        stats = loader.get_statistics()
        
        assert 'num_samples' in stats
        assert 'avg_length' in stats
        assert stats['num_samples'] == 5


class TestRULERLoader:
    """Test RULER dataset loader."""
    
    def test_initialization(self):
        """Test RULER loader initialization."""
        loader = RULERLoader(context_length=2048, num_samples=10)
        
        assert loader.context_length == 2048
        assert loader.num_samples == 10
    
    def test_generate_niah_samples(self):
        """Test generating needle-in-haystack samples."""
        loader = RULERLoader(context_length=500, num_samples=5, task_type="niah")
        samples = loader.load()
        
        assert len(samples) == 5
        assert all(isinstance(s, DataSample) for s in samples)
        assert all('magic number' in s.question.lower() for s in samples)
    
    def test_generate_variable_tracking(self):
        """Test generating variable tracking samples."""
        loader = RULERLoader(context_length=500, num_samples=3, task_type="variable_tracking")
        samples = loader.load()
        
        assert len(samples) == 3
        assert all('var_' in s.question for s in samples)


class TestNeedleInHaystackGenerator:
    """Test Needle-in-Haystack generator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = NeedleInHaystackGenerator(
            haystack_lengths=[1000, 2000],
            needle_depths=[0.0, 0.5, 1.0]
        )
        
        assert len(generator.haystack_lengths) == 2
        assert len(generator.needle_depths) == 3
    
    def test_generate_samples(self):
        """Test generating samples."""
        generator = NeedleInHaystackGenerator(
            haystack_lengths=[500, 1000],
            needle_depths=[0.0, 1.0],
            num_samples_per_config=2
        )
        
        samples = generator.generate()
        
        # 2 lengths * 2 depths * 2 samples = 8 total
        assert len(samples) == 8
        assert all(isinstance(s, DataSample) for s in samples)
        assert all('passcode' in s.question.lower() for s in samples)
    
    def test_needle_positions(self):
        """Test that needles are placed at correct depths."""
        generator = NeedleInHaystackGenerator(
            haystack_lengths=[1000],
            needle_depths=[0.0, 0.5, 1.0],
            num_samples_per_config=1
        )
        
        samples = generator.generate()
        
        # Check that depths are approximately correct
        for sample in samples:
            depth = sample.metadata['needle_depth']
            position = sample.metadata['needle_position']
            length = sample.metadata['actual_length']
            
            assert 0 <= depth <= 1.0
            assert 0 <= position <= length


class TestMetricsBasic:
    """Test basic metric functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "Hello, World! This is a TEST."
        normalized = normalize_text(text)
        
        assert normalized == "hello world this is a test"
    
    def test_exact_match_positive(self):
        """Test exact match with matching strings."""
        score = exact_match("the answer is 42", "The Answer Is 42")
        assert score == 1.0
    
    def test_exact_match_negative(self):
        """Test exact match with different strings."""
        score = exact_match("hello", "world")
        assert score == 0.0
    
    def test_f1_token_score_perfect(self):
        """Test F1 with perfect match."""
        score = f1_token_score("hello world", "hello world")
        assert score == 1.0
    
    def test_f1_token_score_partial(self):
        """Test F1 with partial match."""
        score = f1_token_score("hello world", "hello there")
        assert 0 < score < 1.0
    
    def test_f1_token_score_no_match(self):
        """Test F1 with no match."""
        score = f1_token_score("hello", "world")
        assert score == 0.0


class TestComputeAccuracy:
    """Test accuracy computation."""
    
    def test_compute_accuracy_perfect(self):
        """Test with perfect predictions."""
        predictions = ["answer1", "answer2", "answer3"]
        references = ["answer1", "answer2", "answer3"]
        
        metrics = compute_accuracy(predictions, references)
        
        assert metrics['exact_match'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['num_samples'] == 3
    
    def test_compute_accuracy_mixed(self):
        """Test with mixed predictions."""
        predictions = ["correct", "wrong answer", "correct"]
        references = ["correct", "right answer", "correct"]
        
        metrics = compute_accuracy(predictions, references)
        
        assert 0 < metrics['exact_match'] < 1.0
        assert 0 < metrics['f1'] <= 1.0


class TestRetrievalAccuracy:
    """Test retrieval accuracy."""
    
    def test_perfect_retrieval(self):
        """Test with perfect retrieval."""
        predictions = ["needle1", "needle2", "needle3"]
        references = ["needle1", "needle2", "needle3"]
        
        metrics = compute_retrieval_accuracy(predictions, references)
        
        assert metrics['retrieval_accuracy'] == 1.0
        assert metrics['perfect_retrieval'] == 1.0
    
    def test_partial_retrieval(self):
        """Test with partial retrieval."""
        predictions = ["needle1", "wrong answer", "needle3"]
        references = ["needle1", "needle2", "needle3"]
        
        metrics = compute_retrieval_accuracy(predictions, references, threshold=0.8)
        
        # Should have 2/3 correct
        assert 0 < metrics['retrieval_accuracy'] < 1.0
        assert metrics['retrieval_accuracy'] > 0.5


class TestMemoryEfficiency:
    """Test memory efficiency metrics."""
    
    def test_memory_efficiency_half(self):
        """Test with 50% memory usage."""
        metrics = compute_memory_efficiency(
            memory_used=512,
            memory_baseline=1024
        )
        
        assert metrics['memory_ratio'] == 0.5
        assert metrics['memory_reduction_pct'] == 50.0
        assert metrics['compression_rate'] == 2.0
    
    def test_memory_efficiency_quarter(self):
        """Test with 25% memory usage."""
        metrics = compute_memory_efficiency(
            memory_used=256,
            memory_baseline=1024
        )
        
        assert metrics['memory_ratio'] == 0.25
        assert metrics['memory_reduction_pct'] == 75.0
        assert metrics['compression_rate'] == 4.0


class TestLatencyMetrics:
    """Test latency metrics."""
    
    def test_latency_statistics(self):
        """Test computing latency statistics."""
        latencies = [10.0, 12.0, 11.0, 13.0, 10.5]
        
        metrics = compute_latency_metrics(latencies)
        
        assert 'mean_latency' in metrics
        assert 'median_latency' in metrics
        assert 'p95_latency' in metrics
        assert 10.0 <= metrics['mean_latency'] <= 13.0
    
    def test_empty_latencies(self):
        """Test with empty latency list."""
        metrics = compute_latency_metrics([])
        
        assert metrics['mean_latency'] == 0.0


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert len(tracker.metrics) == 0
    
    def test_add_metric(self):
        """Test adding metrics."""
        tracker = MetricsTracker()
        
        tracker.add('accuracy', 0.95)
        tracker.add('accuracy', 0.96)
        tracker.add('loss', 0.1)
        
        assert len(tracker.metrics['accuracy']) == 2
        assert len(tracker.metrics['loss']) == 1
    
    def test_add_batch(self):
        """Test adding batch of metrics."""
        tracker = MetricsTracker()
        
        tracker.add_batch({'acc': 0.9, 'loss': 0.2})
        tracker.add_batch({'acc': 0.92, 'loss': 0.18})
        
        assert len(tracker.metrics['acc']) == 2
        assert len(tracker.metrics['loss']) == 2
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        tracker = MetricsTracker()
        
        for i in range(10):
            tracker.add('metric1', float(i))
        
        summary = tracker.get_summary()
        
        assert 'metric1' in summary
        assert 'mean' in summary['metric1']
        assert summary['metric1']['count'] == 10
    
    def test_get_mean(self):
        """Test getting mean of a metric."""
        tracker = MetricsTracker()
        
        tracker.add('test', 10.0)
        tracker.add('test', 20.0)
        
        mean = tracker.get_mean('test')
        assert mean == 15.0
    
    def test_clear(self):
        """Test clearing metrics."""
        tracker = MetricsTracker()
        tracker.add('test', 1.0)
        tracker.clear()
        
        assert len(tracker.metrics) == 0


class TestAggregateMetrics:
    """Test metric aggregation."""
    
    def test_aggregate_multiple_samples(self):
        """Test aggregating metrics from multiple samples."""
        metrics_list = [
            {'acc': 0.9, 'f1': 0.85},
            {'acc': 0.92, 'f1': 0.87},
            {'acc': 0.88, 'f1': 0.83},
        ]
        
        aggregated = aggregate_metrics(metrics_list)
        
        assert 'acc_mean' in aggregated
        assert 'f1_mean' in aggregated
        assert aggregated['num_samples'] == 3
    
    def test_aggregate_empty(self):
        """Test aggregating empty list."""
        aggregated = aggregate_metrics([])
        assert aggregated == {}


class TestSmartKVLogger:
    """Test SmartKV logger."""
    
    def test_initialization(self):
        """Test logger initialization."""
        logger = SmartKVLogger(name="test")
        assert logger.name == "test"
    
    def test_log_messages(self):
        """Test logging messages."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            logger = SmartKVLogger(name="test", log_file=log_file)
            
            logger.info("Test message")
            logger.debug("Debug message")
            logger.warning("Warning message")
            
            # Check that log file was created
            assert Path(log_file).exists()
        finally:
            Path(log_file).unlink(missing_ok=True)
    
    def test_log_with_data(self):
        """Test logging with additional data."""
        logger = SmartKVLogger(name="test")
        
        # Should not raise error
        logger.info("Test", accuracy=0.95, loss=0.1)


class TestExperimentTracker:
    """Test ExperimentTracker."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=tmpdir)
            assert tracker.log_dir.exists()
    
    def test_start_experiment(self):
        """Test starting an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=tmpdir)
            
            exp_id = tracker.start_experiment(
                name="test_exp",
                config={'lr': 0.001, 'batch_size': 32}
            )
            
            assert exp_id is not None
            assert tracker.current_experiment is not None
    
    def test_log_and_end_experiment(self):
        """Test full experiment lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=tmpdir)
            
            exp_id = tracker.start_experiment(
                name="test",
                config={'param': 1}
            )
            
            tracker.log_metrics({'acc': 0.9}, step=1)
            tracker.log_metrics({'acc': 0.95}, step=2)
            
            tracker.end_experiment({'final_acc': 0.95})
            
            # Check that file was saved
            exp_file = tracker.get_experiment_path(exp_id)
            assert exp_file.exists()
            
            # Load and verify
            loaded = tracker.load_experiment(exp_id)
            assert loaded['name'] == "test"
            assert len(loaded['metrics']) == 2
    
    def test_list_experiments(self):
        """Test listing experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=tmpdir)
            
            # Create multiple experiments
            for i in range(3):
                tracker.start_experiment(f"exp_{i}", {'param': i})
                tracker.end_experiment({})
            
            experiments = tracker.list_experiments()
            assert len(experiments) >= 3


class TestMemoryProfiler:
    """Test MemoryProfiler."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        profiler = MemoryProfiler()
        assert len(profiler.measurements) == 0
    
    def test_measure(self):
        """Test measuring memory."""
        profiler = MemoryProfiler()
        
        stats = profiler.measure(label="test")
        
        assert 'system_memory_mb' in stats
        assert 'system_memory_percent' in stats
        assert stats['label'] == "test"
    
    def test_get_summary(self):
        """Test getting summary."""
        profiler = MemoryProfiler()
        
        profiler.measure("start")
        profiler.measure("end")
        
        summary = profiler.get_summary()
        
        assert 'num_measurements' in summary
        assert summary['num_measurements'] == 2


class TestLoggerFactory:
    """Test logger factory functions."""
    
    def test_create_logger(self):
        """Test creating logger."""
        logger = create_logger(name="test", level="DEBUG")
        assert isinstance(logger, SmartKVLogger)
    
    def test_get_global_logger(self):
        """Test getting global logger."""
        logger = get_logger()
        assert isinstance(logger, SmartKVLogger)


class TestConvenienceFunctions:
    """Test convenience functions for data loading."""
    
    def test_load_longbench_function(self):
        """Test load_longbench convenience function."""
        samples = load_longbench(task="test", max_samples=5)
        assert len(samples) > 0
    
    def test_load_ruler_function(self):
        """Test load_ruler convenience function."""
        samples = load_ruler(context_length=500, num_samples=5)
        assert len(samples) == 5
    
    def test_generate_needle_function(self):
        """Test generate_needle_in_haystack convenience function."""
        samples = generate_needle_in_haystack(
            haystack_lengths=[500],
            needle_depths=[0.5]
        )
        assert len(samples) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Structured logging utilities for SmartKV.

Provides logging with colored output, JSON formatting, and experiment tracking.
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class SmartKVLogger:
    """
    Structured logger for SmartKV experiments.
    
    Supports colored console output and JSON file logging.
    """
    
    def __init__(
        self,
        name: str = "smartkv",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        use_colors: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for logs
            use_colors: Whether to use colored output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if use_colors and COLORLOG_AVAILABLE:
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            formatter = logging.Formatter(
                '%(levelname)-8s %(name)s: %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (JSON format)
        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        else:
            self.log_file = None
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message)
        if kwargs:
            self.logger.debug(f"  Data: {json.dumps(kwargs, indent=2)}")
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message)
        if kwargs:
            self.logger.info(f"  Data: {json.dumps(kwargs, indent=2)}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message)
        if kwargs:
            self.logger.warning(f"  Data: {json.dumps(kwargs, indent=2)}")
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message)
        if kwargs:
            self.logger.error(f"  Data: {json.dumps(kwargs, indent=2)}")
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message)
        if kwargs:
            self.logger.critical(f"  Data: {json.dumps(kwargs, indent=2)}")
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.info("=" * 80)
        self.info("EXPERIMENT START")
        self.info("=" * 80)
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 80)
    
    def log_experiment_end(self, results: Dict[str, Any]):
        """Log experiment end with results."""
        self.info("=" * 80)
        self.info("EXPERIMENT END")
        self.info("=" * 80)
        self.info("Results:")
        for key, value in results.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 80)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at a specific step."""
        prefix = f"[Step {step}] " if step is not None else ""
        self.info(f"{prefix}Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")


class ExperimentTracker:
    """
    Track experiment configurations and results.
    
    Saves experiments to JSON for reproducibility.
    """
    
    def __init__(self, log_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            log_dir: Directory to save experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.experiment_id: Optional[str] = None
    
    def start_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            experiment_id: Optional custom experiment ID
        
        Returns:
            Experiment ID
        """
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{name}_{timestamp}"
        
        self.experiment_id = experiment_id
        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'metrics': [],
            'results': {},
        }
        
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics for current experiment.
        
        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        
        if step is not None:
            entry['step'] = step
        
        self.current_experiment['metrics'].append(entry)
    
    def end_experiment(self, results: Dict[str, Any]):
        """
        End current experiment and save results.
        
        Args:
            results: Final experiment results
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['results'] = results
        
        # Save to file
        experiment_file = self.log_dir / f"{self.experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        # Reset
        self.current_experiment = None
        self.experiment_id = None
    
    def get_experiment_path(self, experiment_id: str) -> Path:
        """Get path to experiment file."""
        return self.log_dir / f"{experiment_id}.json"
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load a previous experiment.
        
        Args:
            experiment_id: Experiment ID to load
        
        Returns:
            Experiment data
        """
        experiment_file = self.get_experiment_path(experiment_id)
        
        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        with open(experiment_file, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> list:
        """List all saved experiments."""
        experiment_files = list(self.log_dir.glob("*.json"))
        experiments = []
        
        for exp_file in experiment_files:
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                    experiments.append({
                        'id': exp_data['id'],
                        'name': exp_data['name'],
                        'start_time': exp_data['start_time'],
                    })
            except Exception:
                continue
        
        return sorted(experiments, key=lambda x: x['start_time'], reverse=True)


class MemoryProfiler:
    """
    Profile memory usage during experiments.
    
    Tracks GPU and system memory.
    """
    
    def __init__(self):
        """Initialize memory profiler."""
        self.measurements: list = []
    
    def measure(self, label: str = "default") -> Dict[str, float]:
        """
        Measure current memory usage.
        
        Args:
            label: Label for this measurement
        
        Returns:
            Dict with memory statistics
        """
        import psutil
        
        stats = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'system_memory_mb': psutil.virtual_memory().used / (1024 ** 2),
            'system_memory_percent': psutil.virtual_memory().percent,
        }
        
        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        except Exception:
            pass
        
        self.measurements.append(stats)
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of memory measurements."""
        if not self.measurements:
            return {}
        
        system_mem = [m['system_memory_mb'] for m in self.measurements]
        
        summary = {
            'num_measurements': len(self.measurements),
            'system_memory_mean_mb': sum(system_mem) / len(system_mem),
            'system_memory_max_mb': max(system_mem),
            'system_memory_min_mb': min(system_mem),
        }
        
        # GPU stats if available
        gpu_mem = [m.get('gpu_memory_mb', 0) for m in self.measurements if 'gpu_memory_mb' in m]
        if gpu_mem:
            summary['gpu_memory_mean_mb'] = sum(gpu_mem) / len(gpu_mem)
            summary['gpu_memory_max_mb'] = max(gpu_mem)
        
        return summary
    
    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()


def create_logger(
    name: str = "smartkv",
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> SmartKVLogger:
    """
    Create a SmartKV logger.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        SmartKVLogger instance
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    return SmartKVLogger(
        name=name,
        level=level_map.get(level, logging.INFO),
        log_file=log_file
    )


# Global default logger
_default_logger: Optional[SmartKVLogger] = None


def get_logger() -> SmartKVLogger:
    """Get the default global logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = create_logger()
    return _default_logger


def set_default_logger(logger: SmartKVLogger):
    """Set the default global logger."""
    global _default_logger
    _default_logger = logger

"""
Utility Functions

Common utilities for the LLM fine-tuning pipeline including:
- Seed setting and reproducibility
- Device management
- Memory optimization
- Data formatting
- Model utilities
- Configuration helpers
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic operations (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: "auto", "cuda", "cpu", or specific device index
        
    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return torch.device(device)


def get_device_map(
    model: torch.nn.Module,
    max_memory: Optional[Dict[str, str]] = None,
    device_map: str = "auto",
) -> Dict[str, Any]:
    """
    Get device map for model parallelism.
    
    Args:
        model: The model to map
        max_memory: Maximum memory per device
        device_map: Device mapping strategy
        
    Returns:
        Device map dictionary
    """
    if device_map == "auto":
        from accelerate import infer_auto_device_map
        if max_memory is None:
            max_memory = get_balanced_memory(model)
        return infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=model._no_split_modules)
    
    return device_map


def get_balanced_memory(model: torch.nn.Module) -> Dict[str, str]:
    """
    Get balanced memory allocation for all available devices.
    
    Args:
        model: The model to calculate memory for
        
    Returns:
        Dictionary mapping device indices to memory strings
    """
    if not torch.cuda.is_available():
        return {}

    memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(i)
        # Use 85% of free memory
        memory[i] = f"{int(free_mem * 0.85 / 1e9)}GiB"
    
    memory["cpu"] = "16GiB"
    return memory


@contextmanager
def empty_cuda_cache():
    """Context manager to empty CUDA cache after block."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


def clear_memory() -> None:
    """Clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics."""
    stats = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            max_allocated = torch.cuda.max_memory_allocated(i)
            
            stats[f"gpu_{i}"] = {
                "total_gb": total / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "max_allocated_gb": max_allocated / 1e9,
                "free_gb": (total - reserved) / 1e9,
                "utilization": allocated / total * 100,
            }
    
    # CPU memory
    import psutil
    vm = psutil.virtual_memory()
    stats["cpu"] = {
        "total_gb": vm.total / 1e9,
        "available_gb": vm.available / 1e9,
        "used_gb": vm.used / 1e9,
        "percent": vm.percent,
    }
    
    return stats


def print_memory_summary(prefix: str = "") -> None:
    """Print memory summary to console."""
    stats = get_memory_stats()
    print(f"\n{prefix}Memory Summary:")
    for device, mem in stats.items():
        if device.startswith("gpu"):
            print(f"  {device}: {mem['allocated_gb']:.2f}GB allocated, "
                  f"{mem['reserved_gb']:.2f}GB reserved, "
                  f"{mem['free_gb']:.2f}GB free ({mem['utilization']:.1f}%)")
        elif device == "cpu":
            print(f"  {device}: {mem['used_gb']:.2f}GB used, "
                  f"{mem['available_gb']:.2f}GB available ({mem['percent']:.1f}%)")


def format_number(num: Union[int, float]) -> str:
    """Format number with appropriate suffix."""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f}PB"


def format_time(seconds: float) -> str:
    """Format seconds as human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def estimate_model_memory(
    num_params: int,
    dtype: torch.dtype = torch.bfloat16,
    quant_bits: Optional[int] = None,
) -> float:
    """
    Estimate model memory usage.
    
    Args:
        num_params: Number of parameters
        dtype: Data type for parameters
        quant_bits: Quantization bits (4, 8) or None for no quantization
        
    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)
    
    if quant_bits:
        bytes_per_param = quant_bits / 8
    
    # Model params + gradients + optimizer states (Adam: 2x params)
    total_bytes = num_params * bytes_per_param * 3  # params + grads + optimizer
    return total_bytes / 1e9


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """Get detailed parameter statistics."""
    total = 0
    trainable = 0
    frozen = 0
    
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": trainable / total * 100 if total > 0 else 0,
    }


def freeze_layers(model: torch.nn.Module, layer_pattern: str) -> int:
    """Freeze layers matching pattern."""
    import re
    pattern = re.compile(layer_pattern)
    frozen = 0
    
    for name, param in model.named_parameters():
        if pattern.search(name):
            param.requires_grad = False
            frozen += 1
    
    return frozen


def unfreeze_layers(model: torch.nn.Module, layer_pattern: str) -> int:
    """Unfreeze layers matching pattern."""
    import re
    pattern = re.compile(layer_pattern)
    unfrozen = 0
    
    for name, param in model.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            unfrozen += 1
    
    return unfrozen


def get_model_device_map(model: torch.nn.Module) -> Dict[str, Any]:
    """Get device map of model parameters."""
    if hasattr(model, "hf_device_map"):
        return model.hf_device_map
    
    devices = set()
    for param in model.parameters():
        devices.add(str(param.device))
    
    return {"devices": list(devices)}


def move_model_to_device(model: torch.nn.Module, device: Union[str, torch.device]) -> torch.nn.Module:
    """Move model to device."""
    if isinstance(device, str):
        device = torch.device(device)
    
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        # Model already has device map, skip manual move
        return model
    
    return model.to(device)


def cast_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """Cast model parameters to dtype."""
    for param in model.parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
    return model


def get_model_size_str(model: torch.nn.Module) -> str:
    """Get human-readable model size string."""
    stats = get_parameter_stats(model)
    return f"{stats['total']:,} parameters ({stats['trainable_percent']:.1f}% trainable)"


def print_model_summary(model: torch.nn.Module, input_shape: Optional[Tuple] = None) -> None:
    """Print comprehensive model summary."""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Architecture: {model.__class__.__name__}")
    
    stats = get_parameter_stats(model)
    print(f"Total Parameters: {stats['total']:,} ({stats['total']/1e9:.2f}B)")
    print(f"Trainable: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
    print(f"Frozen: {stats['frozen']:,}")
    
    # Memory estimate
    mem_gb = estimate_model_memory(stats['total'])
    print(f"Estimated Memory (bf16): {mem_gb:.2f}GB")
    
    if input_shape:
        print(f"Input Shape: {input_shape}")
    
    print("=" * 60)


def create_unique_id(prefix: str = "") -> str:
    """Create a unique identifier."""
    timestamp = int(time.time() * 1000)
    random_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}{timestamp}_{random_suffix}"


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data as JSON with proper encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict], path: Union[str, Path]) -> None:
    """Save data as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    separator: str = " ",
) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split(separator)
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = separator.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks


def batch_iterator(
    items: List[Any],
    batch_size: int,
    drop_last: bool = False,
) -> Generator[List[Any], None, None]:
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        yield batch


def chunked_dict(
    d: Dict[str, Any],
    chunk_size: int,
) -> Generator[Dict[str, Any], None, None]:
    """Split dictionary into chunks of specified size."""
    items = list(d.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute with default."""
    try:
        return getattr(obj, attr)
    except AttributeError:
        return default


def safe_setattr(obj: Any, attr: str, value: Any) -> bool:
    """Safely set attribute."""
    try:
        setattr(obj, attr, value)
        return True
    except (AttributeError, TypeError):
        return False


def merge_dicts(*dicts: Dict[str, Any], deep: bool = False) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if deep:
            for k, v in d.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge_dicts(result[k], v, deep=True)
                else:
                    result[k] = v
        else:
            result.update(d)
    return result


def filter_dict(d: Dict[str, Any], keys: List[str], keep: bool = True) -> Dict[str, Any]:
    """Filter dictionary by keys."""
    if keep:
        return {k: v for k, v in d.items() if k in keys}
    return {k: v for k, v in d.items() if k not in keys}


def deep_update(base: Dict, update: Dict) -> Dict:
    """Deep update dictionary."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable:
    """Decorator to retry function on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


def timed_cache(ttl: int = 300) -> Callable:
    """Decorator for time-limited caching."""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", logger: Optional[Callable] = None):
        self.name = name
        self.logger = logger or print
        self.start_time = 0
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        self.logger(f"{self.name} took {self.elapsed:.4f}s")
    
    def get_elapsed(self) -> float:
        return time.perf_counter() - self.start_time


class ProgressTracker:
    """Track progress with ETA estimation."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, n: int = 1) -> None:
        self.current += n
        self._print_progress()
    
    def _print_progress(self) -> None:
        if self.current == 0:
            return
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        percent = self.current / self.total * 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(
            f"\r{self.description}: [{bar}] {percent:.1f}% "
            f"({self.current}/{self.total}) "
            f"ETA: {format_time(remaining)}",
            end="",
            flush=True,
        )
        
        if self.current >= self.total:
            print()  # New line at completion


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    import subprocess
    
    info = {}
    
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        info["commit"] = "unknown"
    
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        info["branch"] = "unknown"
    
    try:
        info["dirty"] = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip() != ""
    except:
        info["dirty"] = False
    
    return info


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    base = Path(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = base / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def backup_file(path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """Create backup of file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if backup_dir is None:
        backup_dir = path.parent / "backups"
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{timestamp}{path.suffix}"
    backup_path = backup_dir / backup_name
    
    import shutil
    shutil.copy2(path, backup_path)
    
    return backup_path


__all__ = [
    "set_seed",
    "get_device",
    "get_device_map",
    "get_balanced_memory",
    "empty_cuda_cache",
    "clear_memory",
    "get_memory_stats",
    "print_memory_summary",
    "format_number",
    "format_bytes",
    "format_time",
    "estimate_model_memory",
    "count_parameters",
    "get_parameter_stats",
    "freeze_layers",
    "unfreeze_layers",
    "get_model_device_map",
    "move_model_to_device",
    "cast_model_dtype",
    "get_model_size_str",
    "print_model_summary",
    "create_unique_id",
    "save_json",
    "load_json",
    "save_jsonl",
    "load_jsonl",
    "truncate_text",
    "split_text_into_chunks",
    "batch_iterator",
    "chunked_dict",
    "safe_getattr",
    "safe_setattr",
    "merge_dicts",
    "filter_dict",
    "deep_update",
    "retry",
    "timed_cache",
    "Timer",
    "ProgressTracker",
    "get_git_info",
    "create_experiment_dir",
    "backup_file",
]
"""
Structured Logging Module

Provides unified logging with multiple outputs (console, file, TensorBoard, W&B, MLflow).
Supports structured JSON logging, colored console output, and experiment tracking.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StructuredFormatter(logging.Formatter):
    """JSON formatter with structured fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_fields = {
            "service": "llm-finetuning",
            "version": "1.0.0",
        }

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            **self.default_fields,
        }

        # Add GPU info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            log_data["gpu"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "name", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter with level-based colors."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = None):
        super().__init__(fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[34m{record.name}{self.RESET}"  # Blue logger name
        return super().format(record)


class TensorBoardLogger:
    """TensorBoard logging wrapper with automatic step tracking."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        comment: str = "",
        **kwargs,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or f"run_{int(time.time())}"
        self.comment = comment
        self._writer = None
        self._steps: Dict[str, int] = {}
        self._kwargs = kwargs
        self._init_writer()

    def _init_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(
                log_dir=str(self.log_dir / self.experiment_name),
                comment=self.comment,
                **self._kwargs,
            )
        except ImportError:
            logging.warning("TensorBoard not available, TensorBoardLogger disabled")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log a scalar value."""
        if self._writer is None:
            return
        step = step if step is not None else self._steps.get(tag, 0)
        self._writer.add_scalar(tag, value, step, walltime=walltime)
        self._steps[tag] = step + 1

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple scalar values under a tag."""
        if self._writer is None:
            return
        step = step if step is not None else self._steps.get(tag, 0)
        self._writer.add_scalars(tag, values, step)
        self._steps[tag] = step + 1

    def log_histogram(
        self,
        tag: str,
        values: Union[List[float], "torch.Tensor"],
        step: Optional[int] = None,
        bins: str = "tensorflow",
    ) -> None:
        """Log histogram of values."""
        if self._writer is None:
            return
        step = step if step is not None else self._steps.get(tag, 0)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self._writer.add_histogram(tag, values, step, bins=bins)
        self._steps[tag] = step + 1

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text."""
        if self._writer is None:
            return
        step = step if step is not None else self._steps.get(tag, 0)
        self._writer.add_text(tag, text, step)

    def log_model_graph(self, model: "torch.nn.Module", input_sample: "torch.Tensor") -> None:
        """Log model computational graph."""
        if self._writer is None:
            return
        self._writer.add_graph(model, input_sample)

    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log hyperparameters and final metrics."""
        if self._writer is None:
            return
        self._writer.add_hparams(hparams, metrics or {})

    def flush(self) -> None:
        """Flush the writer."""
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer."""
        if self._writer:
            self._writer.close()
            self._writer = None


class WandbLogger:
    """Weights & Biases logger with automatic config tracking."""

    def __init__(
        self,
        project: str = "llm-finetuning",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        **kwargs,
    ):
        self._run = None
        self._config = config or {}
        self._step = 0
        self._init_wandb(project, entity, name, tags, notes, mode, **kwargs)

    def _init_wandb(
        self,
        project: str,
        entity: Optional[str],
        name: Optional[str],
        tags: Optional[List[str]],
        notes: Optional[str],
        mode: str,
        **kwargs,
    ) -> None:
        try:
            import wandb
            wandb.init(
                project=project,
                entity=entity,
                name=name or f"run_{int(time.time())}",
                config=self._config,
                tags=tags or [],
                notes=notes,
                mode=mode,
                **kwargs,
            )
            self._run = wandb.run
            # Watch model gradients if model is provided later
        except ImportError:
            logging.warning("wandb not available, WandbLogger disabled")

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics and data."""
        if self._run is None:
            return
        step = step if step is not None else self._step
        import wandb
        wandb.log(data, step=step)
        self._step = step + 1

    def log_model(
        self,
        model: "torch.nn.Module",
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Log model as W&B artifact."""
        if self._run is None:
            return
        import wandb
        artifact = wandb.Artifact(
            name or "model",
            type="model",
            description=f"Model checkpoint at step {self._step}",
        )
        # Save model temporarily
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), path)
            artifact.add_file(str(path))
        wandb.log_artifact(artifact, aliases=aliases)

    def watch(
        self,
        model: "torch.nn.Module",
        log: str = "all",
        log_freq: int = 100,
    ) -> None:
        """Watch model gradients and parameters."""
        if self._run is None:
            return
        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run:
            import wandb
            wandb.finish()
            self._run = None


class MlflowLogger:
    """MLflow logger for experiment tracking."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "llm-finetuning",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tags = tags or {}
        self._active_run = None
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._client = mlflow.tracking.MlflowClient()
        except ImportError:
            logging.warning("MLflow not available, MlflowLogger disabled")

    def start_run(self) -> None:
        """Start MLflow run."""
        import mlflow
        self._active_run = mlflow.start_run(run_name=self.run_name)
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        import mlflow
        mlflow.end_run(status=status)
        self._active_run = None

    def log_param(self, key: str, value: Any) -> None:
        import mlflow
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        import mlflow
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        import mlflow
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model(self, model: "torch.nn.Module", artifact_path: str) -> None:
        import mlflow.pytorch
        mlflow.pytorch.log_model(model, artifact_path)


def setup_logging(
    log_dir: Union[str, Path] = "./logs",
    level: str = "INFO",
    format_type: str = "json",
    console: bool = True,
    file_rotation: bool = True,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 10,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        format_type: "json" or "text"
        console: Enable console output
        file_rotation: Enable file rotation
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
        
    Returns:
        Configured root logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))

    formatters = {
        "json": StructuredFormatter(),
        "text": logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        "colored": ColoredConsoleFormatter(),
    }

    formatter = formatters.get(format_type, formatters["text"])

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatters["colored"])
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # File handler
    if file_rotation:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "training.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(
            log_dir / "training.log",
            encoding="utf-8",
        )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    # Training metrics file handler
    metrics_handler = logging.handlers.RotatingFileHandler(
        log_dir / "training_metrics.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    metrics_handler.setFormatter(formatter)
    metrics_handler.setLevel(logging.INFO)
    # Add filter for metrics
    metrics_handler.addFilter(lambda r: getattr(r, "log_metrics", False))
    root_logger.addHandler(metrics_handler)

    return root_logger


class TrainingMetricsLogger:
    """Specialized logger for training metrics with structured output."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("training.metrics")

    def log_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        gpu_memory_allocated: Optional[float] = None,
        gpu_memory_reserved: Optional[float] = None,
        **extra,
    ) -> None:
        """Log training step metrics."""
        self.logger.info(
            f"Step {step} metrics",
            extra={
                "log_metrics": True,
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_sec,
                "samples_per_sec": samples_per_sec,
                "gpu_memory_allocated": gpu_memory_allocated,
                "gpu_memory_reserved": gpu_memory_reserved,
                **extra,
            },
        )

    def log_eval(
        self,
        step: int,
        eval_loss: float,
        metrics: Dict[str, float],
        **extra,
    ) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            f"Evaluation at step {step}",
            extra={
                "log_metrics": True,
                "step": step,
                "eval_loss": eval_loss,
                "metrics": metrics,
                **extra,
            },
        )

    def log_checkpoint(
        self,
        step: int,
        path: str,
        is_best: bool = False,
    ) -> None:
        """Log checkpoint save."""
        self.logger.info(
            f"Checkpoint saved",
            extra={
                "log_metrics": True,
                "step": step,
                "checkpoint_path": path,
                "is_best": is_best,
            },
        )

    def log_lr_schedule(
        self,
        step: int,
        lr: float,
        scheduler_type: str,
    ) -> None:
        """Log learning rate schedule."""
        self.logger.info(
            f"Learning rate updated",
            extra={
                "log_metrics": True,
                "step": step,
                "learning_rate": lr,
                "scheduler_type": scheduler_type,
            },
        )


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.info(
                    f"{func.__name__} completed in {elapsed:.3f}s",
                    extra={"function": func.__name__, "elapsed_seconds": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}",
                    extra={"function": func.__name__, "elapsed_seconds": elapsed},
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with proper configuration."""
    return logging.getLogger(name)


def log_gpu_memory(logger: Optional[logging.Logger] = None, prefix: str = ""):
    """Log current GPU memory usage."""
    log = logger or logging.getLogger(__name__)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        log.info(
            f"{prefix}GPU Memory: {allocated:.2f}GB allocated, "
            f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max allocated"
        )


def configure_experiment_logging(
    experiment_name: str,
    log_dir: Union[str, Path] = "./logs",
    level: str = "INFO",
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    use_mlflow: bool = False,
    wandb_config: Optional[Dict[str, Any]] = None,
    mlflow_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure logging for an experiment with multiple backends.
    
    Returns:
        Dictionary with logger instances
    """
    loggers = {}
    
    # Setup base logging
    base_logger = setup_logging(
        log_dir=Path(log_dir) / experiment_name,
        level=level,
        format_type="json",
    )
    loggers["base"] = base_logger

    # Training metrics logger
    loggers["metrics"] = TrainingMetricsLogger(
        logging.getLogger("training.metrics")
    )

    # TensorBoard
    tb_logger = None
    if use_tensorboard:
        try:
            tb_logger = TensorBoardLogger(
                log_dir=Path(log_dir) / experiment_name / "tensorboard",
                experiment_name=experiment_name,
            )
            loggers["tensorboard"] = tb_logger
        except Exception as e:
            logging.warning(f"Failed to initialize TensorBoard: {e}")

    # W&B
    wandb_logger = None
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=wandb_config.get("project", "llm-finetuning"),
                entity=wandb_config.get("entity"),
                name=experiment_name,
                tags=wandb_config.get("tags", []),
                notes=wandb_config.get("notes"),
            )
            loggers["wandb"] = wandb_logger
        except Exception as e:
            logging.warning(f"Failed to initialize W&B: {e}")

    # MLflow
    mlflow_logger = None
    if use_mlflow:
        try:
            mlflow_logger = MlflowLogger(
                tracking_uri=mlflow_config.get("tracking_uri", "http://localhost:5000"),
                experiment_name=mlflow_config.get("experiment", experiment_name),
                run_name=experiment_name,
                tags=mlflow_config.get("tags", {}),
            )
            mlflow_logger.start_run()
            loggers["mlflow"] = mlflow_logger
        except Exception as e:
            logging.warning(f"Failed to initialize MLflow: {e}")

    return loggers


__all__ = [
    "StructuredFormatter",
    "ColoredConsoleFormatter",
    "TensorBoardLogger",
    "WandbLogger",
    "MlflowLogger",
    "setup_logging",
    "TrainingMetricsLogger",
    "log_execution_time",
    "get_logger",
    "log_gpu_memory",
    "configure_experiment_logging",
]
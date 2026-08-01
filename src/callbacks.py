"""
Training Callbacks

Comprehensive callback system for training monitoring, checkpointing,
early stopping, logging, and profiling.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.integrations import (
    TensorBoardCallback,
    WandbCallback,
    MLflowCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class CallbackState:
    """Internal state for callback coordination."""
    best_metric: Optional[float] = None
    best_step: int = 0
    patience_counter: int = 0
    training_start_time: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    step_times: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0


class BaseCallback(TrainerCallback, ABC):
    """Base callback with common utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = CallbackState()

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback with configurable patience and threshold.
    
    Stops training when the monitored metric doesn't improve for `patience`
    evaluations by at least `threshold`.
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.verbose = verbose
        self.state = CallbackState()
        self._best_value: Optional[float] = None
        self._counter = 0

    @property
    def best_metric(self) -> Optional[float]:
        return self._best_value

    @property
    def counter(self) -> int:
        return self._counter

    @counter.setter
    def counter(self, val: int) -> None:
        self._counter = val

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        if metrics is None or self.metric_for_best not in metrics:
            return

        current_value = metrics[self.metric_for_best]

        if self._best_value is None:
            self._best_value = current_value
            self._counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: Initial best {self.metric_for_best} = {current_value:.4f}"
                )
            return

        improved = False
        if self.greater_is_better:
            improved = current_value > self._best_value + self.threshold
        else:
            improved = current_value < self._best_value - self.threshold

        if improved:
            self._best_value = current_value
            self._counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.metric_for_best} improved to {current_value:.4f}"
                )
        else:
            self._counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement for {self._counter}/{self.patience} "
                    f"evals. Best {self.metric_for_best} = {self._best_value:.4f}"
                )

            if self._counter >= self.patience:
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training after {self.patience} "
                        f"evaluations without improvement"
                    )
                control.should_training_stop = True


class ModelCheckpointCallback(BaseCallback):
    """
    Model checkpointing callback with best model tracking.
    
    Saves checkpoints based on strategy and keeps track of best model
    based on monitored metric.
    """

    def __init__(
        self,
        save_steps: int = 100,
        save_total_limit: int = 3,
        save_best: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        output_dir: str = "./checkpoints",
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        super().__init__()
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self._best_metric: Optional[float] = None
        self._saved_checkpoints: List[Path] = []

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step % self.save_steps == 0:
            self._save_checkpoint(args, state, kwargs.get("model"), kwargs.get("optimizer"), 
                                 kwargs.get("lr_scheduler"), is_best=False)

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        if self.save_best and metrics and self.metric_for_best in metrics:
            current = metrics[self.metric_for_best]
            
            is_best = False
            if self._best_metric is None:
                is_best = True
            elif self.greater_is_better:
                is_best = current > self._best_metric
            else:
                is_best = current < self._best_metric
            
            if is_best:
                self._best_metric = current
                self._save_checkpoint(args, state, kwargs.get("model"), kwargs.get("optimizer"),
                                    kwargs.get("lr_scheduler"), is_best=True)
                logger.info(f"Saved best model with {self.metric_for_best} = {current:.4f}")

    def _save_checkpoint(
        self,
        args: Any,
        state: TrainerState,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        is_best: bool,
    ) -> None:
        if model is None:
            return

        step = state.global_step
        if is_best:
            checkpoint_dir = self.output_dir / "best"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        
        # Save tokenizer if available
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            model.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "global_step": step,
            "epoch": state.epoch,
            "best_metric": self._best_metric,
            "optimizer_state_dict": optimizer.state_dict() if optimizer and self.save_optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler and self.save_scheduler else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

        # Track checkpoint
        if not is_best:
            self._saved_checkpoints.append(checkpoint_dir)
            self._cleanup_old_checkpoints()

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _cleanup_old_checkpoints(self) -> None:
        while len(self._saved_checkpoints) > self.save_total_limit:
            oldest = self._saved_checkpoints.pop(0)
            if oldest.exists():
                import shutil
                shutil.rmtree(oldest)
                logger.info(f"Removed old checkpoint: {oldest}")


class LoggingCallback(BaseCallback):
    """
    Comprehensive logging callback for training metrics.
    
    Logs to console, file, TensorBoard, W&B, and MLflow.
    """

    def __init__(
        self,
        log_steps: int = 10,
        log_gpu_memory: bool = True,
        log_learning_rate: bool = True,
        log_grad_norm: bool = True,
        log_file: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        use_mlflow: bool = False,
    ):
        super().__init__()
        self.log_steps = log_steps
        self.log_gpu_memory = log_gpu_memory
        self.log_learning_rate = log_learning_rate
        self.log_grad_norm = log_grad_norm
        self.log_file = log_file
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        
        self._tensorboard_writer = None
        self._wandb_run = None
        
        if log_file:
            self._setup_file_logging(log_file)

    def _setup_file_logging(self, log_file: str) -> None:
        """Setup file logging with JSON format."""
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.state.training_start_time = time.time()
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tensorboard_writer = SummaryWriter(log_dir=args.logging_dir)
            except ImportError:
                logger.warning("TensorBoard not available, skipping TensorBoard logging")

        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project="llm-finetuning",
                    config=vars(args) if hasattr(args, "__dict__") else {},
                    name=args.run_name if hasattr(args, "run_name") else None,
                )
            except ImportError:
                logger.warning("W&B not available, skipping W&B logging")

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> None:
        if state.global_step % self.log_steps != 0:
            return

        current_time = time.time()
        step_time = current_time - self.state.last_log_time
        self.state.last_log_time = current_time
        self.state.step_times.append(step_time)

        metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else 0,
            "step_time": step_time,
            "samples_per_sec": args.per_device_train_batch_size * args.gradient_accumulation_steps / step_time if step_time > 0 else 0,
        }

        if self.log_gpu_memory and torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated_mb": self._get_memory_mb(),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_peak_memory_mb": self._get_peak_memory_mb(),
            })

        if self.log_grad_norm and model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            metrics["grad_norm"] = total_norm ** 0.5

        self._log_metrics(metrics, state.global_step)

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        # Console logging
        log_parts = [f"Step {step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                log_parts.append(f"{k}={v:.4f}")
            else:
                log_parts.append(f"{k}={v}")
        logger.info(" | ".join(log_parts))

        # TensorBoard
        if self._tensorboard_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tensorboard_writer.add_scalar(k, v, step)

        # W&B
        if self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        # MLflow
        # MLflow logging would go here

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        if metrics:
            eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
            self._log_metrics(eval_metrics, state.global_step)

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        total_time = time.time() - self.state.training_start_time
        logger.info(f"Training completed in {self._format_time(total_time)}")
        
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
        if self._wandb_run:
            import wandb
            wandb.finish()


class ProfilerCallback(BaseCallback):
    """
    PyTorch profiler callback for performance analysis.
    
    Profiles CPU/GPU usage, memory, and operator-level during training.
    """

    def __init__(
        self,
        profile_steps: int = 10,
        profile_dir: str = "./logs/profiler",
        activities: Optional[List[str]] = None,
        record_shapes: bool = True,
        with_stack: bool = True,
        wait: int = 0,
        warmup: int = 1,
        active: int = 10,
        repeat: int = 1,
    ):
        super().__init__()
        self.profile_steps = profile_steps
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.activities = activities or ["cpu", "cuda"]
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        self._profiler: Optional[torch.profiler.profile] = None

    def on_train_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if torch.cuda.is_available():
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ] if "cuda" in self.activities else [torch.profiler.ProfilerActivity.CPU],
                schedule=self.schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                profile_memory=True,
                with_flops=True,
            )
            self._profiler.__enter__()

    def on_step_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._profiler and state.global_step % self.profile_steps == 0:
            self._profiler.step()

    def on_train_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._profiler:
            self._profiler.__exit__(None, None, None)
            logger.info(f"Profile saved to {self.profile_dir}")


class GradientNormCallback(BaseCallback):
    """Log gradient norms at specified intervals."""

    def __init__(self, log_steps: int = 10, max_norm: float = 1.0):
        super().__init__()
        self.log_steps = log_steps
        self.max_norm = max_norm

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> None:
        if state.global_step % self.log_steps != 0 or model is None:
            return

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        logger.info(f"Step {state.global_step}: grad_norm = {total_norm:.4f}")

        if total_norm > self.max_norm * 10:
            logger.warning(f"Gradient norm {total_norm:.4f} exceeds threshold, potential instability")


class ThroughputCallback(BaseCallback):
    """Log training throughput metrics (tokens/sec, samples/sec)."""

    def __init__(self, log_steps: int = 10):
        super().__init__()
        self.log_steps = log_steps
        self.last_log_time = time.time()
        self.last_log_step = 0

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step % self.log_steps != 0:
            return

        current_time = time.time()
        current_step = state.global_step

        elapsed = current_time - self.last_log_time
        steps = current_step - self.last_log_step

        if elapsed > 0:
            steps_per_sec = steps / elapsed
            samples_per_sec = (
                steps_per_sec * args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            tokens_per_sec = (
                samples_per_sec * args.max_seq_length if hasattr(args, "max_seq_length") else 0
            )

            logger.info(
                f"Step {current_step}: {steps_per_sec:.2f} steps/s, "
                f"{samples_per_sec:.2f} samples/s, "
                f"{tokens_per_sec:.0f} tokens/s"
            )

        self.last_log_time = current_time
        self.last_log_step = current_step


class CallbackManager:
    """
    Central manager for all training callbacks.
    
    Handles callback registration, ordering, and execution.
    """

    def __init__(self):
        self.callbacks: List[BaseCallback] = []

    def add_callback(self, callback: BaseCallback) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type) -> None:
        self.callbacks = [c for c in self.callbacks if not isinstance(c, callback_type)]

    def get_callback(self, callback_type: type) -> Optional[BaseCallback]:
        for c in self.callbacks:
            if isinstance(c, callback_type):
                return c
        return None

    def get_trainer_callbacks(self) -> List[TrainerCallback]:
        """Convert to Trainer-compatible callbacks."""
        return [c for c in self.callbacks if isinstance(c, TrainerCallback)]


def create_default_callbacks(
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "./checkpoints",
    logging_dir: str = "./logs",
) -> List[TrainerCallback]:
    """Create default callback suite for training."""
    config = config or {}
    
    callbacks = [
        EarlyStoppingCallback(
            patience=config.get("early_stopping_patience", 3),
            threshold=config.get("early_stopping_threshold", 0.001),
            metric_for_best=config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=config.get("greater_is_better", False),
        ),
        ModelCheckpointCallback(
            save_steps=config.get("save_steps", 100),
            save_total_limit=config.get("save_total_limit", 3),
            save_best=config.get("load_best_model_at_end", True),
            metric_for_best=config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=config.get("greater_is_better", False),
            output_dir=output_dir,
        ),
        LoggingCallback(
            log_steps=config.get("logging_steps", 10),
            log_gpu_memory=config.get("log_gpu_memory", True),
            log_learning_rate=config.get("log_learning_rate", True),
            log_grad_norm=config.get("log_grad_norm", True),
            log_file=f"{logging_dir}/training.log",
            use_tensorboard=config.get("use_tensorboard", True),
            use_wandb=config.get("use_wandb", True),
        ),
    ]

    # Add profiler if enabled
    if config.get("use_profiler", False):
        callbacks.append(ProfilerCallback(
            profile_steps=config.get("profile_steps", 10),
            profile_dir=f"{logging_dir}/profiler",
        ))

    # Add gradient norm callback
    if config.get("log_grad_norm", True):
        callbacks.append(GradientNormCallback(
            log_steps=config.get("logging_steps", 10),
            max_norm=config.get("max_grad_norm", 1.0),
        ))

    # Add throughput callback
    callbacks.append(ThroughputCallback(log_steps=config.get("logging_steps", 10)))

    return callbacks


__all__ = [
    "BaseCallback",
    "CallbackState",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LoggingCallback",
    "ProfilerCallback",
    "GradientNormCallback",
    "ThroughputCallback",
    "CallbackManager",
    "create_default_callbacks",
]
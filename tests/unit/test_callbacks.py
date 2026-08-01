"""Unit tests for src/callbacks.py."""

import time
from unittest.mock import MagicMock

import torch
from src.callbacks import (
    BaseCallback,
    CallbackState,
    EarlyStoppingCallback,
    GradientNormCallback,
    LoggingCallback,
    ThroughputCallback,
)


class TestCallbackState:
    """Tests for CallbackState dataclass."""

    def test_default_state(self):
        state = CallbackState()
        assert state.best_metric is None
        assert state.best_step == 0
        assert state.patience_counter == 0
        assert isinstance(state.step_times, list)
        assert state.peak_memory_mb == 0.0


class TestBaseCallback:
    """Tests for BaseCallback class."""

    def test_initialization(self):
        cb = BaseCallback(config={"key": "value"})
        assert cb.config["key"] == "value"

    def test_get_memory_mb(self):
        cb = BaseCallback()
        mem = cb._get_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0.0

    def test_get_peak_memory_mb(self):
        cb = BaseCallback()
        peak = cb._get_peak_memory_mb()
        assert isinstance(peak, float)
        assert peak >= 0.0

    def test_format_time(self):
        cb = BaseCallback()
        assert cb._format_time(30.5) == "30.5s"
        assert cb._format_time(120.0) == "2.0m"
        assert cb._format_time(7200.0) == "2.0h"


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_initialization(self):
        cb = EarlyStoppingCallback(patience=5, threshold=0.01)
        assert cb.patience == 5
        assert cb.threshold == 0.01
        assert cb.metric_for_best == "eval_loss"
        assert not cb.greater_is_better

    def test_improvement_triggers_reset(self):
        cb = EarlyStoppingCallback(patience=3, greater_is_better=False, verbose=False)
        control = MagicMock()
        state = MagicMock()

        cb.on_evaluate(None, state, control, metrics={"eval_loss": 2.5})
        assert cb._best_value == 2.5
        assert cb._counter == 0

        cb.on_evaluate(None, state, control, metrics={"eval_loss": 2.1})
        assert cb._best_value == 2.1
        assert cb._counter == 0

    def test_no_improvement_increments_counter(self):
        cb = EarlyStoppingCallback(
            patience=2, threshold=0.01, greater_is_better=False, verbose=False
        )
        control = MagicMock()
        state = MagicMock()

        cb.on_evaluate(None, state, control, metrics={"eval_loss": 2.0})
        cb.on_evaluate(None, state, control, metrics={"eval_loss": 2.005})
        assert cb._counter == 1

        cb.on_evaluate(None, state, control, metrics={"eval_loss": 2.1})
        assert cb._counter == 2
        assert control.should_training_stop is True

    def test_greater_is_better(self):
        cb = EarlyStoppingCallback(
            patience=2, threshold=0.05, greater_is_better=True, verbose=False
        )
        control = MagicMock()
        state = MagicMock()

        cb.on_evaluate(None, state, control, metrics={"eval_loss": 0.8})
        cb.on_evaluate(None, state, control, metrics={"eval_loss": 0.9})
        assert cb._best_value == 0.9
        assert cb._counter == 0


class TestGradientNormCallback:
    """Tests for GradientNormCallback."""

    def test_on_step_end(self):
        cb = GradientNormCallback(log_steps=1)
        state = MagicMock()
        state.global_step = 1
        control = MagicMock()
        model = MagicMock()

        param1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param1.grad = torch.tensor([0.1, 0.2])
        model.parameters.return_value = [param1]

        cb.on_step_end(None, state, control, model=model)
        assert cb.state is not None


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_on_step_end(self):
        cb = LoggingCallback(log_steps=1, use_tensorboard=False, use_wandb=False)
        state = MagicMock()
        state.global_step = 1
        control = MagicMock()
        args = MagicMock()
        args.per_device_train_batch_size = 2
        args.gradient_accumulation_steps = 2

        cb.on_step_end(args, state, control)


class TestThroughputCallback:
    """Tests for ThroughputCallback."""

    def test_throughput_calculation(self):
        cb = ThroughputCallback(log_steps=1)
        state = MagicMock()
        state.global_step = 10
        control = MagicMock()
        args = MagicMock()
        args.per_device_train_batch_size = 2
        args.gradient_accumulation_steps = 2
        args.max_seq_length = 512

        cb.last_log_time = time.time() - 1.0
        cb.last_log_step = 0
        cb.on_step_end(args, state, control)
        assert cb.last_log_step == 10

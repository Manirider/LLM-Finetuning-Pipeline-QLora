#!/usr/bin/env python
"""
Unit tests for Training Module
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import (
    CallbacksConfig,
    CheckpointConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    LoggingCallbackConfig,
    LoRAConfig,
    ProfilerConfig,
    QuantizationConfig,
    RuntimeConfig,
    TrainerConfig,
    TrainingConfig,
)
from src.train import (
    EarlyStoppingCallback,
    GPUMemoryCallback,
    GradientNormCallback,
    LearningRateCallback,
    ProfilerCallback,
    ThroughputCallback,
    TrainingMetrics,
    create_argument_parser,
    create_callbacks,
)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_default_values(self):
        metrics = TrainingMetrics()
        assert metrics.train_loss == 0.0
        assert metrics.eval_loss == 0.0
        assert metrics.learning_rate == 0.0
        assert metrics.grad_norm == 0.0
        assert metrics.step == 0

    def test_custom_values(self):
        metrics = TrainingMetrics(
            train_loss=1.5,
            eval_loss=1.2,
            learning_rate=2e-4,
            grad_norm=0.5,
            step=100,
        )
        assert metrics.train_loss == 1.5
        assert metrics.step == 100


class TestGradientNormCallback:
    """Test GradientNormCallback."""

    def test_initialization(self):
        callback = GradientNormCallback(log_freq=5)
        assert callback.log_freq == 5

    @patch("src.train.logger")
    def test_on_step_end_logs(self, mock_logger):
        callback = GradientNormCallback(log_freq=10)
        model = Mock()
        param = Mock()
        param.grad = Mock()
        param.grad.data.norm.return_value = 1.0
        model.parameters.return_value = [param]

        state = Mock()
        state.global_step = 10
        args = Mock()

        callback.on_step_end(args, state, None, model=model)

        mock_logger.info.assert_called()


class TestGPUMemoryCallback:
    """Test GPUMemoryCallback."""

    def test_initialization(self):
        callback = GPUMemoryCallback(log_freq=20)
        assert callback.log_freq == 20


class TestLearningRateCallback:
    """Test LearningRateCallback."""

    def test_initialization(self):
        callback = LearningRateCallback(log_freq=15)
        assert callback.log_freq == 15

    @patch("src.train.logger")
    def test_on_step_end_logs_lr(self, mock_logger):
        callback = LearningRateCallback(log_freq=10)
        optimizer = Mock()
        optimizer.param_groups = [{"lr": 2e-4}]

        state = Mock()
        state.global_step = 10
        args = Mock()

        callback.on_step_end(args, state, None, optimizer=optimizer)

        mock_logger.info.assert_called()


class TestThroughputCallback:
    """Test ThroughputCallback."""

    def test_initialization(self):
        callback = ThroughputCallback(log_freq=10)
        assert callback.log_freq == 10
        assert callback.last_step == 0


class TestProfilerCallback:
    """Test ProfilerCallback."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ProfilerCallback(
                profile_dir=tmpdir,
                profile_steps=5,
                activities=["cpu"],
            )
            assert callback.profile_steps == 5
            assert Path(tmpdir).exists()


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""

    def test_initialization(self):
        callback = EarlyStoppingCallback(
            patience=3,
            threshold=0.001,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )
        assert callback.patience == 3
        assert callback.threshold == 0.001
        assert callback._counter == 0

    def test_improvement_resets_counter(self):
        callback = EarlyStoppingCallback(patience=3, metric_for_best="eval_loss")

        metrics = {"eval_loss": 1.0}
        state = Mock()
        state.global_step = 10
        args = Mock()
        control = Mock()

        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback._best_value == 1.0
        assert callback._counter == 0

        metrics = {"eval_loss": 0.8}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback._best_value == 0.8
        assert callback._counter == 0

    def test_no_improvement_increments_counter(self):
        callback = EarlyStoppingCallback(patience=2, metric_for_best="eval_loss")

        metrics = {"eval_loss": 1.0}
        state = Mock()
        state.global_step = 10
        args = Mock()
        control = Mock()

        callback.on_evaluate(args, state, control, metrics=metrics)

        metrics = {"eval_loss": 1.1}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback._counter == 1

        metrics = {"eval_loss": 1.2}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback._counter == 2
        assert control.should_training_stop is True


class TestCreateCallbacks:
    """Test create_callbacks function."""

    def test_creates_all_callbacks(self):
        callbacks_config = CallbacksConfig(
            early_stopping=EarlyStoppingConfig(enabled=True, patience=3),
            checkpoint=CheckpointConfig(enabled=True, save_steps=100),
            logging=LoggingCallbackConfig(enabled=True, log_steps=10, log_gpu_memory=True),
            profiler=ProfilerConfig(enabled=False),
        )

        callbacks = create_callbacks(callbacks_config)

        callback_types = [type(c).__name__ for c in callbacks]
        assert "EarlyStoppingCallback" in callback_types
        assert "GradientNormCallback" in callback_types
        assert "GPUMemoryCallback" in callback_types
        assert "LearningRateCallback" in callback_types
        assert "ThroughputCallback" in callback_types

    def test_skips_disabled_callbacks(self):
        callbacks_config = CallbacksConfig(
            early_stopping=EarlyStoppingConfig(enabled=False),
            checkpoint=CheckpointConfig(enabled=False),
            logging=LoggingCallbackConfig(enabled=False),
            profiler=ProfilerConfig(enabled=False),
        )

        callbacks = create_callbacks(callbacks_config)
        assert len(callbacks) == 0


class TestArgumentParser:
    """Test CLI argument parser."""

    def test_default_config(self):
        parser = create_argument_parser()
        args = parser.parse_args([])
        assert args.config == "configs"

    def test_custom_config(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_resume_from_checkpoint(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--resume-from-checkpoint", "./checkpoints/checkpoint-100"])
        assert args.resume_from_checkpoint == "./checkpoints/checkpoint-100"

    def test_output_dir_override(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--output-dir", "./custom_output"])
        assert args.output_dir == "./custom_output"

    def test_learning_rate_override(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--learning-rate", "1e-4"])
        assert args.learning_rate == 1e-4

    def test_batch_size_override(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--batch-size", "8"])
        assert args.batch_size == 8

    def test_epochs_override(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--epochs", "5"])
        assert args.epochs == 5

    def test_wandb_disable(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--no-wandb"])
        assert args.no_wandb is True

    def test_tensorboard_disable(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--no-tensorboard"])
        assert args.no_tensorboard is True

    def test_dry_run(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True


class TestTrainingConfig:
    """Test TrainingConfig model."""

    def test_default_config(self):
        config = TrainingConfig()
        assert isinstance(config.trainer, TrainerConfig)
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.quantization, QuantizationConfig)
        assert isinstance(config.callbacks, CallbacksConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.experiment, ExperimentConfig)

    def test_trainer_config_defaults(self):
        config = TrainingConfig()
        assert config.trainer.learning_rate == 2.0e-4
        assert config.trainer.num_train_epochs == 3
        assert config.trainer.per_device_train_batch_size == 4
        assert config.trainer.gradient_accumulation_steps == 4
        assert config.trainer.bf16 is True
        assert config.trainer.fp16 is False

    def test_lora_config_defaults(self):
        config = TrainingConfig()
        assert config.lora.r == 64
        assert config.lora.lora_alpha == 16
        assert config.lora.lora_dropout == 0.05
        assert config.lora.use_rslora is True

    def test_quantization_config_defaults(self):
        config = TrainingConfig()
        assert config.quantization.load_in_4bit is True
        assert config.quantization.bnb_4bit_quant_type == "nf4"
        assert config.quantization.bnb_4bit_use_double_quant is True


class TestConfigOverride:
    """Test configuration override functionality."""

    def test_override_learning_rate(self):
        config = TrainingConfig()
        config.trainer.learning_rate = 1e-4
        assert config.trainer.learning_rate == 1e-4

    def test_override_batch_size(self):
        config = TrainingConfig()
        config.trainer.per_device_train_batch_size = 8
        assert config.trainer.per_device_train_batch_size == 8

    def test_override_lora_r(self):
        config = TrainingConfig()
        config.lora.r = 128
        assert config.lora.r == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

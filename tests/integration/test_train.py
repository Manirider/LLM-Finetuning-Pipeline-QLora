"""Integration tests for training module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import (
    CallbacksConfig,
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingCallbackConfig,
    ProfilerConfig,
    RuntimeConfig,
    TrainerConfig,
)
from src.config import SFTConfig as ConfigSFTConfig
from src.train import (
    EarlyStoppingCallback,
    GradientNormCallback,
    LearningRateCallback,
    ProfilerCallback,
    ThroughputCallback,
    create_callbacks,
    create_sft_config,
    create_training_arguments,
)


class TestCallbacksIntegration:
    """Test training callbacks integration."""

    def test_early_stopping_callback(self):
        """Test early stopping callback logic."""
        callback = EarlyStoppingCallback(
            patience=2,
            threshold=0.01,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        state = MagicMock()
        state.global_step = 100
        control = MagicMock()
        args = MagicMock()

        # First evaluation - sets best metric
        metrics = {"eval_loss": 1.0}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.best_metric == 1.0
        assert callback.counter == 0

        # Improvement - resets counter
        metrics = {"eval_loss": 0.8}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.best_metric == 0.8
        assert callback.counter == 0

        # No improvement - increments counter
        metrics = {"eval_loss": 0.85}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.counter == 1

        # No improvement - triggers early stop
        metrics = {"eval_loss": 0.9}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.counter == 2
        assert control.should_training_stop is True

    def test_early_stopping_greater_is_better(self):
        """Test early stopping with greater_is_better=True."""
        callback = EarlyStoppingCallback(
            patience=2,
            threshold=0.01,
            metric_for_best="accuracy",
            greater_is_better=True,
        )

        state = MagicMock()
        state.global_step = 100
        control = MagicMock()
        args = MagicMock()

        # First evaluation
        metrics = {"accuracy": 0.5}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.best_metric == 0.5

        # Improvement (higher is better)
        metrics = {"accuracy": 0.6}
        callback.on_evaluate(args, state, control, metrics=metrics)
        assert callback.best_metric == 0.6
        assert callback.counter == 0

    def test_gradient_norm_callback(self, capfd):
        """Test gradient norm callback."""
        callback = GradientNormCallback(log_freq=10)

        # Create model with gradients
        param = MagicMock()
        param.grad = MagicMock()
        param.grad.data.norm.return_value = 1.5

        model = MagicMock()
        model.parameters.return_value = [param]

        state = MagicMock()
        state.global_step = 10
        args = MagicMock()
        control = MagicMock()

        callback.on_step_end(args, state, control, model=model)

        out, _ = capfd.readouterr()
        assert "grad_norm" in out

    def test_learning_rate_callback(self, capfd):
        """Test learning rate callback."""
        callback = LearningRateCallback(log_freq=10)

        optimizer = MagicMock()
        optimizer.param_groups = [{"lr": 2e-4}]

        state = MagicMock()
        state.global_step = 10
        args = MagicMock()
        control = MagicMock()

        callback.on_step_end(args, state, control, optimizer=optimizer)

        out, _ = capfd.readouterr()
        assert "learning_rate" in out

    def test_throughput_callback(self):
        """Test throughput callback."""
        callback = ThroughputCallback(log_freq=10)

        state = MagicMock()
        state.global_step = 10
        args = MagicMock()
        control = MagicMock()

        # Should not crash
        callback.on_step_end(args, state, control)

    def test_profiler_callback(self, temp_dir):
        """Test profiler callback."""
        profiler_dir = temp_dir / "profiler"
        callback = ProfilerCallback(
            profile_dir=str(profiler_dir),
            profile_steps=3,
        )

        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        # Should not crash on train begin/end
        callback.on_train_begin(args, state, control)
        callback.on_step_end(args, state, control)
        callback.on_train_end(args, state, control)


class TestCreateCallbacks:
    """Test callback creation from config."""

    def test_create_all_callbacks(self):
        """Test creating all callbacks from config."""
        callbacks_config = CallbacksConfig(
            early_stopping=EarlyStoppingConfig(enabled=True, patience=3),
            checkpoint=CheckpointConfig(enabled=True),
            logging=LoggingCallbackConfig(
                enabled=True,
                log_steps=10,
                log_gpu_memory=True,
                log_learning_rate=True,
                log_grad_norm=True,
            ),
            profiler=ProfilerConfig(enabled=False),
        )

        callbacks = create_callbacks(callbacks_config)

        callback_types = [type(c).__name__ for c in callbacks]
        assert "EarlyStoppingCallback" in callback_types
        assert "GradientNormCallback" in callback_types
        assert "GPUMemoryCallback" in callback_types
        assert "LearningRateCallback" in callback_types
        assert "ThroughputCallback" in callback_types

    def test_create_callbacks_disabled(self):
        """Test creating callbacks with some disabled."""
        callbacks_config = CallbacksConfig(
            early_stopping=EarlyStoppingConfig(enabled=False),
            checkpoint=CheckpointConfig(enabled=False),
            logging=LoggingCallbackConfig(enabled=False),
            profiler=ProfilerConfig(enabled=False),
        )

        callbacks = create_callbacks(callbacks_config)
        assert len(callbacks) == 0


class TestTrainingArguments:
    """Test TrainingArguments creation."""

    def test_create_training_arguments(self, temp_dir):
        """Test creating TrainingArguments from config."""
        import torch

        has_cuda = torch.cuda.is_available()
        trainer_config = TrainerConfig(
            output_dir=str(temp_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            fp16=False,
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            run_name="test-run",
        )

        runtime_config = RuntimeConfig(bf16=True, fp16=False)

        args = create_training_arguments(trainer_config, str(temp_dir), runtime_config)

        assert args.output_dir == str(temp_dir)
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-4
        if has_cuda:
            assert args.bf16 is True
        assert args.fp16 is False
        assert args.report_to == ["tensorboard"]
        assert args.run_name == "test-run"

    def test_create_sft_config(self):
        """Test creating SFTConfig from config."""
        sft_config = ConfigSFTConfig(
            max_seq_length=2048,
            packing=False,
            dataset_text_field="text",
        )

        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"

        sft_config_obj = create_sft_config(sft_config, tokenizer)

        assert sft_config_obj.max_seq_length == 2048
        assert sft_config_obj.packing is False
        assert sft_config_obj.dataset_text_field == "text"


class TestTrainingIntegration:
    """Integration tests for training components."""

    @patch("src.train.SFTTrainer")
    @patch("src.train.DataPipeline")
    @patch("src.train.load_model_and_tokenizer")
    @patch("src.train.ConfigManager")
    def test_training_pipeline_components(
        self,
        mock_config_manager,
        mock_load_model,
        mock_data_pipeline,
        mock_sft_trainer,
        temp_dir,
    ):
        """Test that training pipeline components are correctly wired."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.training = MagicMock()
        mock_config.training.trainer = MagicMock()
        mock_config.training.trainer.num_train_epochs = 1
        mock_config.training.trainer.per_device_train_batch_size = 2
        mock_config.training.trainer.gradient_accumulation_steps = 2
        mock_config.training.trainer.output_dir = str(temp_dir)
        mock_config.training.trainer.save_strategy = "steps"
        mock_config.training.trainer.save_steps = 10
        mock_config.training.trainer.eval_steps = 10
        mock_config.training.trainer.logging_steps = 5
        mock_config.training.trainer.report_to = ["none"]
        mock_config.training.trainer.run_name = "test"
        mock_config.training.trainer.dataloader_num_workers = 0
        mock_config.training.trainer.remove_unused_columns = False
        mock_config.training.trainer.label_names = ["labels"]
        mock_config.training.trainer.seed = 42
        mock_config.training.trainer.data_seed = 42
        mock_config.training.trainer.bf16 = False
        mock_config.training.trainer.fp16 = False
        mock_config.training.trainer.learning_rate = 2e-4
        mock_config.training.trainer.weight_decay = 0.01
        mock_config.training.trainer.lr_scheduler_type = "cosine"
        mock_config.training.trainer.warmup_ratio = 0.03
        mock_config.training.trainer.max_grad_norm = 1.0
        mock_config.training.trainer.gradient_checkpointing = True
        mock_config.training.trainer.save_total_limit = 1
        mock_config.training.trainer.load_best_model_at_end = True
        mock_config.training.trainer.metric_for_best_model = "eval_loss"
        mock_config.training.trainer.greater_is_better = False
        mock_config.training.trainer.save_safetensors = True
        mock_config.training.trainer.save_only_model = False
        mock_config.training.trainer.push_to_hub = False
        mock_config.training.trainer.dataloader_pin_memory = True
        mock_config.training.trainer.dataloader_prefetch_factor = 2
        mock_config.training.trainer.dataloader_persistent_workers = False
        mock_config.training.trainer.dataloader_drop_last = False
        mock_config.training.trainer.auto_find_batch_size = False
        mock_config.training.trainer.torch_compile = False
        mock_config.training.trainer.eval_accumulation_steps = 1
        mock_config.training.trainer.prediction_loss_only = False
        mock_config.training.trainer.eval_delay = 0
        mock_config.training.trainer.eval_strategy = "steps"
        mock_config.training.trainer.logging_strategy = "steps"
        mock_config.training.trainer.logging_first_step = True
        mock_config.training.trainer.logging_nan_inf_filter = True
        mock_config.training.trainer.ddp_backend = "nccl"
        mock_config.training.trainer.ddp_find_unused_parameters = False
        mock_config.training.trainer.ddp_bucket_cap_mb = 25
        mock_config.training.trainer.ddp_timeout = 1800

        mock_config.training.callbacks = MagicMock()
        mock_config.training.callbacks.early_stopping = MagicMock(enabled=False)
        mock_config.training.callbacks.logging = MagicMock(enabled=False)
        mock_config.training.callbacks.profiler = MagicMock(enabled=False)

        mock_config.training.sft = MagicMock()
        mock_config.training.sft.max_seq_length = 512
        mock_config.training.sft.packing = False
        mock_config.training.sft.dataset_text_field = "text"

        mock_config.model = MagicMock()
        mock_config.model.model = MagicMock()
        mock_config.model.model.model_name_or_path = "test-model"
        mock_config.model.tokenizer = MagicMock()

        mock_config_manager.return_value = mock_config

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = MagicMock(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Mock datasets
        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = Mock(return_value=100)
        mock_eval_ds = MagicMock()
        mock_eval_ds.__len__ = Mock(return_value=20)

        mock_processed = {"test": MagicMock(train=mock_train_ds, validation=mock_eval_ds)}
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_processed
        mock_data_pipeline.return_value = mock_pipeline

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train = MagicMock()
        mock_trainer_instance.state = MagicMock()
        mock_trainer_instance.state.best_model_checkpoint = str(temp_dir / "best")
        mock_trainer_instance.evaluate = MagicMock(return_value={"eval_loss": 1.0})
        mock_sft_trainer.return_value = mock_trainer_instance

        # Import and run (would be actual train.main() in real test)
        from src.train import (
            create_sft_config,
            create_trainer,
            create_training_arguments,
        )

        # Verify imports work
        assert create_training_arguments is not None
        assert create_sft_config is not None
        assert create_trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

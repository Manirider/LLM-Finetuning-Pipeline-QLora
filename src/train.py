"""
Training Module for LLM Fine-tuning

Complete training pipeline with:
- SFTTrainer / Trainer integration
- QLoRA / LoRA / PEFT support
- Gradient checkpointing
- Flash Attention
- Experiment tracking (W&B, TensorBoard, MLflow)
- Callbacks (early stopping, checkpointing, logging, profiling)
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Learning rate scheduling
- Distributed training (DDP)
- Model saving and merging
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from peft import (
    PeftModel,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from src.callbacks import EarlyStoppingCallback as _EarlyStoppingCallbackBase
from src.config import (
    CallbacksConfig,
    ConfigManager,
    ExperimentConfig,
    RuntimeConfig,
    TrainerConfig,
    TrainingConfig,
)
from src.config import SFTConfig as ConfigSFTConfig
from src.data_pipeline import DataPipeline
from src.model_utils import (
    clear_gpu_cache,
    get_gpu_memory_info,
    load_model_and_tokenizer,
    log_gpu_memory,
    merge_and_unload_peft,
    print_model_summary,
    save_model_and_tokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    train_loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    epoch: float = 0.0
    step: int = 0


class GradientNormCallback(TrainerCallback):
    """Callback to log gradient norms."""

    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module = None,
        **kwargs,
    ):
        if state.global_step % self.log_freq == 0 and model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    norm_val = (
                        param_norm.item() if hasattr(param_norm, "item") else float(param_norm)
                    )
                    total_norm += norm_val**2
            total_norm = total_norm**0.5
            print(f"Step {state.global_step}: grad_norm = {total_norm:.4f}")


class GPUMemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage."""

    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.log_freq == 0 and torch.cuda.is_available():
            info = get_gpu_memory_info()
            if info["cuda_available"]:
                for dev in info["devices"]:
                    logger.info(
                        f"Step {state.global_step}: GPU {dev['index']} - "
                        f"Allocated: {dev['allocated_gb']:.2f}GB, "
                        f"Reserved: {dev['reserved_gb']:.2f}GB "
                        f"({dev['utilization_percent']:.1f}%)"
                    )


class LearningRateCallback(TrainerCallback):
    """Callback to log learning rate."""

    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        optimizer: torch.optim.Optimizer = None,
        **kwargs,
    ):
        if state.global_step % self.log_freq == 0 and optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step {state.global_step}: learning_rate = {lr:.2e}")


class ThroughputCallback(TrainerCallback):
    """Callback to log training throughput."""

    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.last_step = 0
        self.last_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.log_freq == 0:
            current_time = time.time()
            steps = state.global_step - self.last_step
            elapsed = current_time - self.last_time
            if elapsed > 0:
                steps_per_sec = steps / elapsed
                logger.info(f"Step {state.global_step}: throughput = {steps_per_sec:.2f} steps/sec")
            self.last_step = state.global_step
            self.last_time = current_time


class ProfilerCallback(TrainerCallback):
    """Callback for PyTorch profiling."""

    def __init__(
        self,
        profile_dir: str = "./logs/profiler",
        profile_steps: int = 10,
        activities: list[str] = None,
        record_shapes: bool = True,
        with_stack: bool = True,
    ):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.profile_steps = profile_steps
        self.activities = activities or ["cpu", "cuda"]
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profiler = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=(
                    [
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                    if "cuda" in self.activities
                    else [torch.profiler.ProfilerActivity.CPU]
                ),
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=1,
                    active=self.profile_steps,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.profiler.__enter__()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            logger.info(f"Profile saved to {self.profile_dir}")


# EarlyStoppingCallback is imported from src.callbacks to avoid duplication.
# Re-export for backward compatibility.
EarlyStoppingCallback = _EarlyStoppingCallbackBase  # noqa: F811


def create_callbacks(callbacks_config: CallbacksConfig) -> list[TrainerCallback]:
    """Create list of callbacks from configuration."""
    callbacks = []

    if callbacks_config.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                patience=callbacks_config.early_stopping.patience,
                threshold=callbacks_config.early_stopping.threshold,
                metric_for_best=callbacks_config.early_stopping.metric_for_best,
                greater_is_better=callbacks_config.early_stopping.greater_is_better,
            )
        )

    if callbacks_config.logging.enabled:
        callbacks.append(GradientNormCallback(log_freq=callbacks_config.logging.log_steps))
        callbacks.append(GPUMemoryCallback(log_freq=callbacks_config.logging.log_steps))
        callbacks.append(LearningRateCallback(log_freq=callbacks_config.logging.log_steps))
        callbacks.append(ThroughputCallback(log_freq=callbacks_config.logging.log_steps))

    if callbacks_config.profiler.enabled:
        callbacks.append(
            ProfilerCallback(
                profile_dir=callbacks_config.profiler.profile_dir,
                profile_steps=callbacks_config.profiler.profile_steps,
                activities=callbacks_config.profiler.activities,
                record_shapes=callbacks_config.profiler.record_shapes,
                with_stack=callbacks_config.profiler.with_stack,
            )
        )

    return callbacks


def setup_experiment_tracking(
    training_config: TrainingConfig,
    experiment_config: ExperimentConfig,
    output_dir: str,
) -> tuple[Any | None, Any | None, Any | None]:
    """Setup experiment tracking (W&B, TensorBoard, MLflow)."""
    wandb_run = None
    tensorboard_writer = None
    mlflow_client = None

    # TensorBoard
    (
        training_config.callbacks.tensorboard
        if hasattr(training_config.callbacks, "tensorboard")
        else None
    )
    if hasattr(training_config, "trainer") and training_config.trainer.report_to:
        if "tensorboard" in training_config.trainer.report_to:
            logger.info(f"TensorBoard logging to {training_config.trainer.logging_dir}")

    # Weights & Biases
    if "wandb" in training_config.trainer.report_to:
        try:
            import wandb

            wandb.init(
                project=experiment_config.project,
                entity=experiment_config.entity,
                name=experiment_config.name,
                tags=experiment_config.tags,
                notes=experiment_config.notes,
                config=(
                    training_config.model_dump() if hasattr(training_config, "model_dump") else {}
                ),
                dir=output_dir,
                resume="allow",
            )
            wandb_run = wandb.run
            logger.info(f"W&B initialized: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    # MLflow
    if hasattr(experiment_config, "mlflow") and experiment_config.mlflow.enabled:
        try:
            import mlflow

            mlflow.set_tracking_uri(experiment_config.mlflow.tracking_uri)
            mlflow.set_experiment(experiment_config.mlflow.experiment_name)
            mlflow.start_run(run_name=experiment_config.mlflow.run_name)
            mlflow_client = mlflow
            logger.info(f"MLflow initialized: {experiment_config.mlflow.tracking_uri}")
        except ImportError:
            logger.warning("mlflow not installed, skipping MLflow logging")

    return wandb_run, tensorboard_writer, mlflow_client


def cleanup_experiment_tracking(
    wandb_run: Any | None,
    mlflow_client: Any | None,
):
    """Cleanup experiment tracking."""
    if wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    if mlflow_client is not None:
        try:
            mlflow_client.end_run()
        except Exception:
            pass


def create_training_arguments(
    trainer_config: TrainerConfig,
    output_dir: str,
    runtime_config: RuntimeConfig,
) -> TrainingArguments:
    """Create TrainingArguments from configuration."""
    bf16_attr = getattr(runtime_config, "bf16", False) if runtime_config else False
    fp16_attr = getattr(runtime_config, "fp16", False) if runtime_config else False

    if bf16_attr:
        bf16 = True
        fp16 = False
    elif fp16_attr:
        fp16 = True
        bf16 = False
    else:
        fp16 = getattr(trainer_config, "fp16", False)
        bf16 = getattr(trainer_config, "bf16", False)

    if not torch.cuda.is_available():
        bf16 = False
        fp16 = False

    if hasattr(trainer_config, "model_dump"):
        config_dict = trainer_config.model_dump()
    elif hasattr(trainer_config, "dict"):
        config_dict = trainer_config.dict()
    else:
        config_dict = {}

    raw_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": getattr(trainer_config, "overwrite_output_dir", True),
        "num_train_epochs": getattr(trainer_config, "num_train_epochs", 3),
        "max_steps": getattr(trainer_config, "max_steps", -1),
        "per_device_train_batch_size": getattr(trainer_config, "per_device_train_batch_size", 4),
        "per_device_eval_batch_size": getattr(trainer_config, "per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": getattr(trainer_config, "gradient_accumulation_steps", 4),
        "learning_rate": getattr(trainer_config, "learning_rate", 2e-4),
        "weight_decay": getattr(trainer_config, "weight_decay", 0.01),
        "adam_beta1": getattr(trainer_config, "adam_beta1", 0.9),
        "adam_beta2": getattr(trainer_config, "adam_beta2", 0.999),
        "adam_epsilon": getattr(trainer_config, "adam_epsilon", 1e-8),
        "max_grad_norm": getattr(trainer_config, "max_grad_norm", 1.0),
        "lr_scheduler_type": getattr(trainer_config, "lr_scheduler_type", "cosine"),
        "warmup_ratio": getattr(trainer_config, "warmup_ratio", 0.03),
        "warmup_steps": getattr(trainer_config, "warmup_steps", 0),
        "fp16": fp16,
        "bf16": bf16,
        "logging_strategy": getattr(trainer_config, "logging_strategy", "steps"),
        "logging_steps": getattr(trainer_config, "logging_steps", 10),
        "logging_first_step": getattr(trainer_config, "logging_first_step", True),
        "save_strategy": getattr(trainer_config, "save_strategy", "steps"),
        "save_steps": getattr(trainer_config, "save_steps", 100),
        "save_total_limit": getattr(trainer_config, "save_total_limit", None),
        "load_best_model_at_end": getattr(trainer_config, "load_best_model_at_end", False),
        "metric_for_best_model": getattr(trainer_config, "metric_for_best_model", None),
        "greater_is_better": getattr(trainer_config, "greater_is_better", None),
        "dataloader_num_workers": getattr(trainer_config, "dataloader_num_workers", 0),
        "dataloader_pin_memory": getattr(trainer_config, "dataloader_pin_memory", True),
        "dataloader_drop_last": getattr(trainer_config, "dataloader_drop_last", False),
        "seed": getattr(trainer_config, "seed", 42),
        "remove_unused_columns": getattr(trainer_config, "remove_unused_columns", True),
        "report_to": getattr(trainer_config, "report_to", None),
        "run_name": getattr(trainer_config, "run_name", None),
    }

    for k, v in config_dict.items():
        if k not in raw_args and v is not None:
            raw_args[k] = v

    raw_args = {k: v for k, v in raw_args.items() if v is not None}

    if not torch.cuda.is_available():
        raw_args["bf16"] = False
        raw_args["fp16"] = False
        raw_args["use_cpu"] = True

    essential_keys = {
        "output_dir",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "use_cpu",
        "fp16",
        "bf16",
        "report_to",
        "run_name",
    }

    while True:
        try:
            return TrainingArguments(**raw_args)
        except TypeError as e:
            import re

            m = re.search(r"'([^']+)'", str(e))
            if m and m.group(1) in raw_args and m.group(1) not in essential_keys:
                raw_args.pop(m.group(1), None)
            else:
                non_essential = [k for k in raw_args if k not in essential_keys]
                if non_essential:
                    raw_args.pop(non_essential[0], None)
                else:
                    break
        except ValueError as e:
            if "bf16" in str(e) or "gpu" in str(e):
                raw_args["bf16"] = False
                raw_args["fp16"] = False
                raw_args["use_cpu"] = True
            else:
                break

    return TrainingArguments(**{k: v for k, v in raw_args.items() if k in essential_keys})


def create_sft_config(
    sft_config: ConfigSFTConfig,
    tokenizer: PreTrainedTokenizer,
) -> SFTConfig:
    """Create SFTConfig for SFTTrainer."""
    raw_args = {
        "output_dir": getattr(sft_config, "output_dir", "./checkpoints"),
        "max_seq_length": getattr(sft_config, "max_seq_length", 512),
        "max_length": getattr(sft_config, "max_seq_length", 512),
        "packing": getattr(sft_config, "packing", False),
        "dataset_text_field": getattr(sft_config, "dataset_text_field", "text"),
        "dataset_kwargs": getattr(sft_config, "dataset_kwargs", None),
        "formatting_func": getattr(sft_config, "formatting_func", None),
        "neftune_noise_alpha": getattr(sft_config, "neftune_noise_alpha", None),
    }

    if not torch.cuda.is_available():
        raw_args["bf16"] = False
        raw_args["fp16"] = False
        raw_args["use_cpu"] = True

    res = None
    while True:
        try:
            res = SFTConfig(**raw_args)
            break
        except TypeError as e:
            import re

            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if m:
                bad_arg = m.group(1)
                raw_args.pop(bad_arg, None)
                if bad_arg == "max_seq_length" and "max_length" not in raw_args:
                    raw_args["max_length"] = getattr(sft_config, "max_seq_length", 512)
                elif bad_arg == "max_length" and "max_seq_length" not in raw_args:
                    raw_args["max_seq_length"] = getattr(sft_config, "max_seq_length", 512)
            else:
                break
        except ValueError as e:
            if "bf16" in str(e) or "gpu" in str(e):
                raw_args["bf16"] = False
                raw_args["fp16"] = False
                raw_args["use_cpu"] = True
            else:
                break

    if res is None:
        res = SFTConfig(output_dir="./checkpoints", use_cpu=not torch.cuda.is_available())

    if not hasattr(res, "max_seq_length"):
        try:
            object.__setattr__(
                res,
                "max_seq_length",
                getattr(res, "max_length", getattr(sft_config, "max_seq_length", 512)),
            )
        except Exception:
            try:
                res.max_seq_length = getattr(
                    res, "max_length", getattr(sft_config, "max_seq_length", 512)
                )
            except Exception:
                pass

    return res


def load_datasets(data_config_path: str = "configs/data.yaml") -> DatasetDict:
    """Load and process datasets using data pipeline."""
    pipeline = DataPipeline(data_config_path)
    processed = pipeline.process()
    return processed


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args: TrainingArguments,
    sft_config: SFTConfig | None = None,
    callbacks: list[TrainerCallback] | None = None,
    use_sft_trainer: bool = True,
) -> Trainer | SFTTrainer:
    """Create Trainer or SFTTrainer."""

    if use_sft_trainer and sft_config is not None:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            sft_config=sft_config,
            callbacks=callbacks or [],
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=callbacks or [],
        )

    return trainer


def train(
    training_config: TrainingConfig,
    model_config: Any,
    tokenizer_config: Any,
    data_config_path: str = "configs/data.yaml",
    output_dir: str = "./checkpoints",
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """
    Main training function.

    Returns:
        Dictionary with training results and metrics.
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    set_seed(training_config.trainer.seed)

    runtime_config = training_config.runtime

    log_gpu_memory("Initial")

    logger.info("Loading model and tokenizer...")
    load_result = load_model_and_tokenizer(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        quantization_config=training_config.quantization,
        peft_config=training_config.lora,
        peft_type="LORA",
        runtime_config=runtime_config,
    )

    model = load_result.model
    tokenizer = load_result.tokenizer

    log_gpu_memory("After model load")

    print_model_summary(model)

    logger.info("Loading datasets...")
    datasets = load_datasets(data_config_path)

    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("validation") or datasets.get("val")

    if train_dataset is None:
        raise ValueError("No training dataset found")

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    training_args = create_training_arguments(
        training_config.trainer,
        output_dir,
        runtime_config,
    )

    sft_config = create_sft_config(training_config.sft, tokenizer)

    callbacks = create_callbacks(training_config.callbacks)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        sft_config=sft_config,
        callbacks=callbacks,
        use_sft_trainer=True,
    )

    wandb_run, tb_writer, mlflow_client = setup_experiment_tracking(
        training_config,
        training_config.experiment,
        output_dir,
    )

    try:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info("Training completed!")
        logger.info(f"Training metrics: {train_result.metrics}")

        log_gpu_memory("After training")

        final_checkpoint = trainer.state.best_model_checkpoint or output_dir
        logger.info(f"Best model checkpoint: {final_checkpoint}")

        if training_config.trainer.save_strategy != "no":
            trainer.save_model(final_checkpoint)
            tokenizer.save_pretrained(final_checkpoint)
            logger.info(f"Model saved to {final_checkpoint}")

        eval_results = {}
        if eval_dataset is not None:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")

        return {
            "train_result": train_result,
            "eval_results": eval_results,
            "best_checkpoint": final_checkpoint,
            "model": model,
            "tokenizer": tokenizer,
        }

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if training_config.trainer.save_strategy != "no":
            trainer.save_model(f"{output_dir}/interrupted")
            tokenizer.save_pretrained(f"{output_dir}/interrupted")
        raise

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        cleanup_experiment_tracking(wandb_run, mlflow_client)

        if runtime_config.empty_cache_steps > 0:
            clear_gpu_cache()

        gc.collect()


def merge_and_save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    merged_dtype: str = "bfloat16",
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    push_to_hub: bool = False,
    hub_model_id: str = "",
    hub_token: str | None = None,
    hub_private_repo: bool = False,
):
    """Merge LoRA adapter and save model."""
    logger.info("Merging LoRA adapter into base model...")

    if isinstance(model, PeftModel):
        merged_model = merge_and_unload_peft(model)
    else:
        merged_model = model

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if merged_dtype in dtype_map:
        merged_model = merged_model.to(dtype_map[merged_dtype])

    save_model_and_tokenizer(
        model=merged_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        save_adapter=False,
        save_tokenizer=True,
        merge_and_unload=False,
        merged_dtype=merged_dtype,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        hub_private_repo=hub_private_repo,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning with QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs",
        help="Config directory path",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Data configuration file",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    parser.add_argument(
        "--merge-and-save",
        action="store_true",
        help="Merge LoRA adapter and save merged model",
    )

    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="./artifacts/models/merged",
        help="Output directory for merged model",
    )

    parser.add_argument(
        "--merged-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for merged model",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to Hugging Face Hub",
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="",
        help="Hugging Face Hub model ID",
    )

    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face Hub token",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=None,
        help="Per device train batch size (overrides config)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without training",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading configuration...")
    config_manager = ConfigManager(config_dir=args.config)

    training_config = config_manager.training
    model_config = config_manager.model.model
    tokenizer_config = config_manager.model.tokenizer

    if args.seed:
        training_config.trainer.seed = args.seed
        training_config.trainer.data_seed = args.seed

    if args.learning_rate is not None:
        training_config.trainer.learning_rate = args.learning_rate

    if args.batch_size is not None:
        training_config.trainer.per_device_train_batch_size = args.batch_size

    if args.epochs is not None:
        training_config.trainer.num_train_epochs = args.epochs

    if args.no_wandb:
        training_config.trainer.report_to = [
            r for r in training_config.trainer.report_to if r != "wandb"
        ]

    if args.no_tensorboard:
        training_config.trainer.report_to = [
            r for r in training_config.trainer.report_to if r != "tensorboard"
        ]

    if args.dry_run:
        logger.info("DRY RUN - Configuration:")
        logger.info(f"  Model: {model_config.model_name_or_path}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Epochs: {training_config.trainer.num_train_epochs}")
        logger.info(f"  Batch size: {training_config.trainer.per_device_train_batch_size}")
        logger.info(f"  Grad accum: {training_config.trainer.gradient_accumulation_steps}")
        logger.info(f"  LR: {training_config.trainer.learning_rate}")
        logger.info(f"  LoRA r: {training_config.lora.r}")
        logger.info(f"  Quantization: 4bit={training_config.quantization.load_in_4bit}")
        return

    result = train(
        training_config=training_config,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        data_config_path=args.data_config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    if args.merge_and_save:
        merge_and_save_model(
            model=result["model"],
            tokenizer=result["tokenizer"],
            output_dir=args.merged_output_dir,
            merged_dtype=args.merged_dtype,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hub_token=args.hub_token,
        )

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

"""Smoke tests for the LLM Fine-tuning Pipeline.

These tests verify basic functionality without requiring external resources.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager
from src.data_pipeline import DataPipeline
from src.evaluate import create_argument_parser as eval_parser
from src.evaluate import main as eval_main
from src.train import create_argument_parser
from src.train import main as train_main


class TestConfigLoading:
    """Test configuration loading works."""

    def test_config_manager_loads(self):
        """Test ConfigManager loads all configs."""
        config = ConfigManager(config_dir="configs")

        assert config.training is not None
        assert config.model is not None
        assert config.data is not None
        assert config.logging is not None
        assert config.evaluation is not None

    def test_config_values_exist(self):
        """Test config has expected default values."""
        config = ConfigManager(config_dir="configs")

        # Training config
        assert config.training.trainer.num_train_epochs == 3
        assert config.training.trainer.learning_rate == 2e-4
        assert config.training.lora.r == 64

        # Model config
        assert "Llama" in config.model.model.model_name_or_path

        # Data config
        assert len(config.data.datasets) > 0
        assert config.data.default_template == "alpaca"

    def test_env_var_resolution(self):
        """Test environment variable resolution."""
        os.environ["TEST_VAR"] = "test_value"

        config = ConfigManager(config_dir="configs", env_file=".env.example")

        # Check that placeholders exist (actual resolution depends on env)
        assert config is not None


class TestDataPipeline:
    """Test data pipeline basic operations."""

    def test_pipeline_init(self, temp_dir):
        """Test pipeline initialization."""
        config_dict = {
            "datasets": [],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {},
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)
        assert pipeline is not None

    def test_formatters_available(self):
        """Test all formatters are available."""
        from src.data_pipeline import FORMATTERS

        expected = ["alpaca", "chatml", "llama3", "vicuna", "zephyr", "plain", "custom"]
        for name in expected:
            assert name in FORMATTERS

    def test_formatter_creation(self):
        """Test formatter creation."""
        from src.data_pipeline import get_formatter

        for name in ["alpaca", "chatml", "llama3"]:
            formatter = get_formatter(name)
            assert formatter is not None


class TestModelUtils:
    """Test model utilities basic operations."""

    def test_torch_dtype_conversion(self):
        """Test torch dtype conversion."""
        from src.model_utils import get_torch_dtype

        assert get_torch_dtype("float16") == torch.float16
        assert get_torch_dtype("bfloat16") == torch.bfloat16
        assert get_torch_dtype("float32") == torch.float32
        assert get_torch_dtype("fp16") == torch.float16
        assert get_torch_dtype("bf16") == torch.bfloat16

    def test_bnb_config_creation(self):
        """Test BNB config creation."""
        from src.config import QuantizationConfig
        from src.model_utils import create_bnb_config

        quant_config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

        bnb_config = create_bnb_config(quant_config)

        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"

    def test_lora_config_creation(self):
        """Test LoRA config creation."""
        from src.config import PEFTLoraConfig
        from src.model_utils import create_lora_config

        peft_config = PEFTLoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        lora_config = create_lora_config(peft_config)

        assert lora_config.r == 64
        assert lora_config.target_modules == ["q_proj", "v_proj"]


class TestTrainingComponents:
    """Test training components."""

    def test_argument_parser(self):
        """Test training argument parser."""
        parser = create_argument_parser()

        args = parser.parse_args(["--config", "configs"])
        assert args.config == "configs"

        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_callbacks_creation(self):
        """Test callback creation."""
        from src.config import CallbacksConfig, EarlyStoppingConfig, LoggingCallbackConfig
        from src.train import create_callbacks

        callbacks_config = CallbacksConfig(
            early_stopping=EarlyStoppingConfig(enabled=True),
            logging=LoggingCallbackConfig(enabled=True),
        )

        callbacks = create_callbacks(callbacks_config)
        assert len(callbacks) > 0

    def test_training_args_creation(self):
        """Test TrainingArguments creation."""
        from src.config import RuntimeConfig, TrainerConfig
        from src.train import create_training_arguments

        trainer_config = TrainerConfig(
            output_dir="./test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
        )
        runtime_config = RuntimeConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_arguments(trainer_config, tmpdir, runtime_config)
            assert args.output_dir == tmpdir


class TestEvaluationComponents:
    """Test evaluation components."""

    def test_eval_argument_parser(self):
        """Test evaluation argument parser."""
        parser = eval_parser()

        args = parser.parse_args(["--base-model", "model", "--output-dir", "./out"])
        assert args.base_model == "model"
        assert args.output_dir == "./out"

    def test_metrics_calculator(self):
        """Test metrics calculator initialization."""
        from src.config import BertScoreConfig, BleuConfig, RougeConfig
        from src.evaluate import MetricsCalculator

        calc = MetricsCalculator(
            rouge_config=RougeConfig(enabled=True),
            bleu_config=BleuConfig(enabled=True),
            bertscore_config=BertScoreConfig(enabled=False),
        )

        assert calc.rouge_config.enabled is True
        assert calc.bleu_config.enabled is True
        assert calc.bertscore_config.enabled is False

    def test_prompt_formatter(self):
        """Test prompt formatter."""
        from src.evaluate import PromptFormatter

        formatter = PromptFormatter()

        # Test alpaca
        prompt = formatter.format(
            template="alpaca",
            instruction="Test",
            input_text="",
            output="",
            system_message="System",
        )
        assert "### Instruction:" in prompt

        # Test chatml
        prompt = formatter.format(
            template="chatml",
            instruction="Test",
            input_text="",
            output="",
            system_message="System",
        )
        assert "im_start>system" in prompt


class TestCLICommands:
    """Test CLI commands work."""

    def test_train_help(self, capsys):
        """Test train --help."""
        import sys

        with patch.object(sys, "argv", ["train", "--help"]):
            with pytest.raises(SystemExit):
                train_main()

        out, _ = capsys.readouterr()
        assert "usage" in out.lower() or "help" in out.lower()

    def test_evaluate_help(self, capsys):
        """Test evaluate --help."""
        import sys

        with patch.object(sys, "argv", ["evaluate", "--help"]):
            with pytest.raises(SystemExit):
                eval_main()

        out, _ = capsys.readouterr()
        assert "usage" in out.lower() or "help" in out.lower()

    def test_data_pipeline_help(self, capsys):
        """Test data_pipeline --help."""
        import sys

        from src.data_pipeline import main as data_main

        with patch.object(sys, "argv", ["data_pipeline", "--help"]):
            with pytest.raises(SystemExit):
                data_main()

        out, _ = capsys.readouterr()
        assert "usage" in out.lower() or "help" in out.lower()


class TestEndToEndImports:
    """Test all modules can be imported."""

    def test_all_src_modules_import(self):
        """Test all src modules can be imported."""
        import src.callbacks
        import src.config
        import src.data_pipeline
        import src.evaluate
        import src.logger
        import src.metrics
        import src.model_utils
        import src.train
        import src.utils

        # Check key exports exist
        assert hasattr(src.config, "ConfigManager")
        assert hasattr(src.data_pipeline, "DataPipeline")
        assert hasattr(src.model_utils, "load_model_and_tokenizer")
        assert hasattr(src.train, "train")
        assert hasattr(src.evaluate, "run_evaluation")

    def test_config_classes_exist(self):
        """Test config classes exist."""
        from src.config import (
            DataConfigComplete,
            EvaluationConfigComplete,
            LoggingConfigComplete,
            ModelConfigComplete,
            TrainingConfig,
        )

        # All should be importable
        assert TrainingConfig is not None
        assert ModelConfigComplete is not None
        assert DataConfigComplete is not None
        assert LoggingConfigComplete is not None
        assert EvaluationConfigComplete is not None


class TestFileStructure:
    """Test expected file structure exists."""

    def test_config_files_exist(self):
        """Test config files exist."""
        configs = [
            "configs/data.yaml",
            "configs/training.yaml",
            "configs/model.yaml",
            "configs/logging.yaml",
            "configs/evaluation.yaml",
        ]
        for config in configs:
            assert Path(config).exists(), f"Missing config: {config}"

    def test_source_files_exist(self):
        """Test source files exist."""
        sources = [
            "src/__init__.py",
            "src/config.py",
            "src/data_pipeline.py",
            "src/model_utils.py",
            "src/train.py",
            "src/evaluate.py",
            "src/logger.py",
            "src/metrics.py",
            "src/callbacks.py",
            "src/utils.py",
        ]
        for source in sources:
            assert Path(source).exists(), f"Missing source: {source}"

    def test_test_files_exist(self):
        """Test test files exist."""
        tests = [
            "tests/unit/test_config.py",
            "tests/unit/test_data_pipeline.py",
            "tests/unit/test_train.py",
            "tests/unit/test_evaluate.py",
            "tests/integration/test_data_pipeline.py",
            "tests/integration/test_model_utils.py",
            "tests/integration/test_evaluate.py",
        ]
        for test in tests:
            assert Path(test).exists(), f"Missing test: {test}"

    def test_documentation_exists(self):
        """Test documentation files exist."""
        docs = [
            "README.md",
            "evaluation_report.md",
            "ARCHITECTURE.md",
            "docs/deployment_guide.md",
            "docs/inference_guide.md",
            "docs/troubleshooting_guide.md",
        ]
        for doc in docs:
            assert Path(doc).exists(), f"Missing doc: {doc}"


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_training_config_validation(self):
        """Test training config validation."""
        from src.config import TrainerConfig

        # Valid config
        config = TrainerConfig(
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=4,
        )
        assert config.learning_rate == 2e-4

    def test_lora_config_validation(self):
        """Test LoRA config validation."""
        from src.config import LoRAConfig

        config = LoRAConfig(r=64, lora_alpha=16)
        assert config.r == 64

        # Test bias validation
        with pytest.raises(ValueError):
            LoRAConfig(bias="invalid")

    def test_quantization_config_validation(self):
        """Test quantization config validation."""
        from src.config import QuantizationConfig

        config = QuantizationConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        assert config.bnb_4bit_quant_type == "nf4"

        with pytest.raises(ValueError):
            QuantizationConfig(bnb_4bit_quant_type="invalid")

        with pytest.raises(ValueError):
            QuantizationConfig(bnb_4bit_compute_dtype="invalid")

    def test_split_ratios_validation(self):
        """Test split ratios validation."""
        from src.config import SplittingConfig

        config = SplittingConfig(ratios={"train": 0.8, "validation": 0.1, "test": 0.1})
        assert config.ratios["train"] == 0.8

        with pytest.raises(ValueError):
            SplittingConfig(
                ratios={"train": 0.8, "validation": 0.1, "test": 0.2}
            )  # Doesn't sum to 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

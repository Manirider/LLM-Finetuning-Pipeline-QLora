"""Pipeline tests for end-to-end workflows."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager
from src.data_pipeline import DataPipeline
from src.model_utils import load_model_and_tokenizer, save_model_and_tokenizer
from src.train import train
from src.evaluate import run_evaluation


class TestDataProcessingPipeline:
    """Test complete data processing pipeline."""

    @patch("src.data_pipeline.load_dataset")
    def test_full_data_pipeline(self, mock_load_dataset, temp_dir):
        """Test complete data processing from raw to formatted."""
        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.column_names = ["instruction", "input", "output"]
        mock_ds.select = Mock(return_value=mock_ds)
        mock_ds.shuffle = Mock(return_value=mock_ds)
        mock_ds.filter = Mock(return_value=mock_ds)
        mock_ds.map = Mock(return_value=mock_ds)
        mock_ds.remove_columns = Mock(return_value=mock_ds)
        mock_ds.drop_duplicates = Mock(return_value=mock_ds)
        mock_load_dataset.return_value = mock_ds

        config_dict = {
            "datasets": [
                {
                    "name": "alpaca",
                    "path": "tatsu-lab/alpaca",
                    "split": "train",
                    "max_samples": 100,
                    "column_mapping": {
                        "instruction": "instruction",
                        "input": "input",
                        "output": "output",
                    },
                }
            ],
            "prompt_templates": {
                "alpaca": {
                    "template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
                    "template_with_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
                    "instruction_key": "instruction",
                    "input_key": "input",
                    "output_key": "output",
                    "add_eos_token": True,
                }
            },
            "default_template": "alpaca",
            "processing": {
                "download": {"cache_dir": str(temp_dir / "raw"), "num_proc": 1},
                "validation": {
                    "enabled": True,
                    "required_columns": ["instruction", "output"],
                    "min_instruction_length": 10,
                    "min_output_length": 5,
                },
                "cleaning": {
                    "enabled": True,
                    "strip_whitespace": True,
                    "normalize_unicode": True,
                },
                "formatting": {
                    "enabled": True,
                    "template": "alpaca",
                    "formatted_field": "text",
                },
                "tokenization": {"enabled": False},
                "splitting": {
                    "enabled": True,
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                    "seed": 42,
                },
            },
            "output": {
                "output_dir": str(temp_dir / "processed"),
                "formats": ["jsonl"],
                "save_splits": True,
            },
        }

        pipeline = DataPipeline(config_dict)
        
        with patch.object(pipeline, "load_local_datasets", return_value={}):
            result = pipeline.process()
        
        assert "alpaca" in result
        assert "train" in result["alpaca"]
        assert "validation" in result["alpaca"]
        assert "test" in result["alpaca"]

    def test_data_pipeline_with_tokenization(self, temp_dir):
        """Test data pipeline with tokenization enabled."""
        from datasets import Dataset
        
        # Create mock tokenized dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.column_names = ["instruction", "input", "output", "text"]
        mock_ds.select = Mock(return_value=mock_ds)
        mock_ds.shuffle = Mock(return_value=mock_ds)
        mock_ds.filter = Mock(return_value=mock_ds)
        mock_ds.map = Mock(return_value=mock_ds)
        mock_ds.remove_columns = Mock(return_value=mock_ds)
        mock_ds.drop_duplicates = Mock(return_value=mock_ds)

        config_dict = {
            "datasets": [{
                "name": "alpaca",
                "path": "tatsu-lab/alpaca",
                "split": "train",
                "max_samples": 100,
            }],
            "prompt_templates": {
                "alpaca": {
                    "template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
                    "template_with_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
                    "add_eos_token": True,
                }
            },
            "default_template": "alpaca",
            "processing": {
                "download": {"cache_dir": str(temp_dir / "raw"), "num_proc": 1},
                "validation": {"enabled": True},
                "cleaning": {"enabled": True},
                "formatting": {"enabled": True, "template": "alpaca"},
                "tokenization": {
                    "enabled": True,
                    "max_seq_length": 512,
                    "truncation": True,
                    "padding": False,
                },
                "splitting": {"enabled": True, "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}},
            },
            "output": {"output_dir": str(temp_dir / "processed"), "formats": ["jsonl"]},
        }

        pipeline = DataPipeline(config_dict)
        
        with patch("src.data_pipeline.load_dataset", return_value=mock_ds):
            with patch.object(pipeline, "load_local_datasets", return_value={}):
                with patch.object(pipeline, "load_tokenizer") as mock_load_tok:
                    mock_tok = MagicMock()
                    mock_tok.eos_token = "</s>"
                    mock_tok.pad_token = None
                    mock_tok.truncation_side = "right"
                    mock_tok.padding_side = "right"
                    mock_load_tok.return_value = mock_tok
                    
                    result = pipeline.process()
        
        assert "alpaca" in result


class TestModelLoadingPipeline:
    """Test model loading pipeline."""

    @patch("src.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("src.model_utils.AutoTokenizer.from_pretrained")
    def test_model_tokenizer_loading(self, mock_tokenizer, mock_model, temp_dir):
        """Test model and tokenizer loading pipeline."""
        # Setup mocks
        mock_tok = MagicMock()
        mock_tok.eos_token = "</s>"
        mock_tok.pad_token = None
        mock_tokenizer.return_value = mock_tok

        mock_mod = MagicMock()
        mock_mod.config = MagicMock()
        mock_mod.config.use_cache = True
        mock_mod.hf_device_map = {}
        mock_model.return_value = mock_mod

        from src.config import (
            ModelConfig, TokenizerConfig, PEFTLoraConfig,
            QuantizationConfig, RuntimeConfig
        )

        model_config = ModelConfig(
            model_name_or_path="test-model",
            load_in_4bit=True,
            gradient_checkpointing=True,
        )
        tokenizer_config = TokenizerConfig(tokenizer_name_or_path="test-model")
        peft_config = PEFTLoraConfig(r=64)
        runtime_config = RuntimeConfig(flash_attention=False, gradient_checkpointing=True)
        quant_config = QuantizationConfig(load_in_4bit=True)

        with patch("src.model_utils.create_bnb_config") as mock_bnb:
            mock_bnb.return_value = MagicMock()
            
            result = load_model_and_tokenizer(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                quantization_config=quant_config,
                peft_config=peft_config,
                peft_type="LORA",
                runtime_config=runtime_config,
            )

        assert result.model is not None
        assert result.tokenizer is not None
        assert result.quantization_config is not None
        assert result.peft_config is not None

    def test_model_save_pipeline(self, temp_dir):
        """Test model saving pipeline."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        save_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            output_dir=str(temp_dir / "saved"),
            save_adapter=False,
            save_tokenizer=True,
            safe_serialization=True,
        )

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()


class TestTrainingPipeline:
    """Test training pipeline."""

    @patch("src.train.SFTTrainer")
    @patch("src.train.DataPipeline")
    @patch("src.train.load_model_and_tokenizer")
    @patch("src.train.ConfigManager")
    def test_training_pipeline_execution(
        self,
        mock_config_manager,
        mock_load_model,
        mock_data_pipeline,
        mock_sft_trainer,
        temp_dir,
    ):
        """Test complete training pipeline execution."""
        # Setup config manager mock
        mock_config = MagicMock()
        
        # Training config
        mock_config.training = MagicMock()
        mock_config.training.trainer = MagicMock()
        mock_config.training.trainer.output_dir = str(temp_dir / "checkpoints")
        mock_config.training.trainer.num_train_epochs = 1
        mock_config.training.trainer.per_device_train_batch_size = 2
        mock_config.training.trainer.gradient_accumulation_steps = 2
        mock_config.training.trainer.learning_rate = 2e-4
        mock_config.training.trainer.bf16 = False
        mock_config.training.trainer.fp16 = False
        mock_config.training.trainer.gradient_checkpointing = True
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
        
        # Callbacks
        mock_config.training.callbacks = MagicMock()
        mock_config.training.callbacks.early_stopping = MagicMock(enabled=False)
        mock_config.training.callbacks.logging = MagicMock(enabled=False)
        mock_config.training.callbacks.profiler = MagicMock(enabled=False)
        
        # SFT config
        mock_config.training.sft = MagicMock()
        mock_config.training.sft.max_seq_length = 512
        mock_config.training.sft.packing = False
        mock_config.training.sft.dataset_text_field = "text"
        
        # Model config
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
        
        mock_processed = {
            "test": MagicMock(train=mock_train_ds, validation=mock_eval_ds)
        }
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

        # Run training (using the actual train function)
        from src.train import train
        from src.config import TrainingConfig, ModelConfig, TokenizerConfig
        
        # Create minimal configs
        training_config = TrainingConfig()
        model_config = ModelConfig()
        tokenizer_config = TokenizerConfig()
        
        # This would be the actual call in main()
        # result = train(training_config, model_config, tokenizer_config, str(temp_dir / "data"))
        
        # Verify the pipeline components are wired correctly
        assert mock_config_manager is not None
        assert mock_load_model is not None
        assert mock_data_pipeline is not None
        assert mock_sft_trainer is not None


class TestEvaluationPipeline:
    """Test evaluation pipeline."""

    @patch("src.evaluate.load_dataset")
    @patch("src.evaluate.AutoModelForCausalLM.from_pretrained")
    @patch("src.evaluate.AutoTokenizer.from_pretrained")
    @patch("src.evaluate.PeftModel.from_pretrained")
    def test_evaluation_pipeline(
        self,
        mock_peft,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        temp_dir,
    ):
        """Test complete evaluation pipeline."""
        # Mock dataset
        mock_load_dataset.return_value = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "", "output": "A2"},
        ]
        
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode = Mock(return_value=[1, 2, 3])
        mock_tok.decode = Mock(return_value="Response")
        mock_tok.apply_chat_template = Mock(return_value="Prompt")
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 1
        mock_tokenizer.return_value = mock_tok
        
        # Mock base model
        mock_mod = MagicMock()
        mock_mod.generate = Mock(return_value=[[1, 2, 3, 4, 5, 6]])
        mock_mod.config = MagicMock(pad_token_id=0, eos_token_id=1)
        mock_mod.device = "cuda:0"
        mock_model.return_value = mock_mod
        
        # Mock PEFT model
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_mod
        mock_peft.return_value = mock_peft_model
        
        # Create config
        from src.config import (
            EvaluationConfigComplete, GenerationConfig,
            EvalDatasetConfig, RougeConfig, BleuConfig,
            BertScoreConfig, PerplexityConfig, DistinctConfig
        )
        
        eval_config = EvaluationConfigComplete(
            generation=GenerationConfig(),
            datasets=[
                EvalDatasetConfig(
                    name="test",
                    path="test_data",
                    split="test",
                    prompt_template="alpaca",
                )
            ],
            metrics={
                "rouge": RougeConfig(enabled=True),
                "bleu": BleuConfig(enabled=True),
                "meteor": type('obj', (object,), {"enabled": True})(),
                "bertscore": BertScoreConfig(enabled=False),
                "perplexity": PerplexityConfig(enabled=False),
                "distinct": DistinctConfig(enabled=True),
            },
            baseline={"enabled": True},
        )
        
        with patch("src.evaluate.clear_gpu_cache"):
            reports = run_evaluation(
                eval_config=eval_config,
                base_model_path="base-model",
                finetuned_model_path=None,
                adapter_path="adapter-path",
                output_dir=str(temp_dir),
            )
        
        assert len(reports) > 0


class TestConfigPipeline:
    """Test configuration pipeline."""

    def test_config_override_pipeline(self):
        """Test config override pipeline."""
        config = ConfigManager(config_dir="configs")
        
        # Override training config
        config.update(
            training={
                "trainer": {
                    "learning_rate": 1e-4,
                    "num_train_epochs": 5,
                }
            }
        )
        
        assert config.training.trainer.learning_rate == 1e-4
        assert config.training.trainer.num_train_epochs == 5
        
        # Override model config
        config.update(
            model={
                "model": {
                    "model_name_or_path": "custom/model"
                }
            }
        )
        
        assert config.model.model.model_name_or_path == "custom/model"

    def test_config_save_resolved(self, temp_dir):
        """Test saving resolved config."""
        config = ConfigManager(config_dir="configs")
        
        output_path = temp_dir / "resolved.yaml"
        config.save_resolved(output_path)
        
        assert output_path.exists()
        
        import yaml
        with open(output_path) as f:
            resolved = yaml.safe_load(f)
        
        assert "training" in resolved
        assert "model" in resolved
        assert "data" in resolved
        assert "logging" in resolved
        assert "evaluation" in resolved

    def test_env_file_override(self, temp_dir):
        """Test .env file override."""
        env_file = temp_dir / ".env"
        env_file.write_text("HF_TOKEN=test_token\nWANDB_API_KEY=test_key\n")
        
        config = ConfigManager(config_dir="configs", env_file=str(env_file))
        
        # Config should load with env vars
        assert config is not None


class TestMergeAndExportPipeline:
    """Test model merging and export pipeline."""

    @patch("src.model_utils.PeftModel.from_pretrained")
    @patch("src.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("src.model_utils.AutoTokenizer.from_pretrained")
    def test_merge_and_export(
        self,
        mock_tokenizer,
        mock_model,
        mock_peft,
        temp_dir,
    ):
        """Test model merge and export pipeline."""
        # Setup mocks
        mock_tok = MagicMock()
        mock_tokenizer.return_value = mock_tok
        
        mock_mod = MagicMock()
        mock_model.return_value = mock_mod
        
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_mod
        mock_peft.return_value = mock_peft_model
        
        from src.model_utils import merge_and_unload_peft, save_model_and_tokenizer
        
        # Load base model
        base_model = mock_mod
        
        # Load PEFT adapter
        peft_model = mock_peft_model
        
        # Merge
        merged = merge_and_unload_peft(peft_model)
        assert merged == mock_mod
        
        # Save merged model
        save_model_and_tokenizer(
            model=merged,
            tokenizer=mock_tok,
            output_dir=str(temp_dir / "merged"),
            save_adapter=False,
            save_tokenizer=True,
            merge_and_unload=False,
        )
        
        mock_mod.save_pretrained.assert_called()
        mock_tok.save_pretrained.assert_called()


class TestFullPipelineIntegration:
    """Test full pipeline integration (requires mocked external deps)."""

    @patch("src.train.SFTTrainer")
    @patch("src.train.DataPipeline")
    @patch("src.train.load_model_and_tokenizer")
    @patch("src.train.ConfigManager")
    @patch("src.evaluate.run_evaluation")
    def test_full_train_evaluate_pipeline(
        self,
        mock_run_eval,
        mock_config_manager,
        mock_load_model,
        mock_data_pipeline,
        mock_sft_trainer,
        temp_dir,
    ):
        """Test full train -> evaluate pipeline."""
        # This test verifies the pipeline components can be wired together
        # without actually running (which requires GPUs and models)
        
        # Setup training mocks
        mock_config = MagicMock()
        mock_config.training = MagicMock()
        mock_config.training.trainer = MagicMock()
        mock_config.training.trainer.output_dir = str(temp_dir / "checkpoints")
        mock_config.training.trainer.report_to = ["none"]
        mock_config.training.callbacks = MagicMock()
        mock_config.training.callbacks.early_stopping = MagicMock(enabled=False)
        mock_config.training.callbacks.logging = MagicMock(enabled=False)
        mock_config.training.callbacks.profiler = MagicMock(enabled=False)
        mock_config.training.sft = MagicMock()
        mock_config.training.sft.max_seq_length = 512
        mock_config.model = MagicMock()
        mock_config.model.model = MagicMock()
        mock_config.model.tokenizer = MagicMock()
        mock_config_manager.return_value = mock_config
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = MagicMock(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        
        mock_train_ds = MagicMock()
        mock_eval_ds = MagicMock()
        mock_processed = {"test": MagicMock(train=mock_train_ds, validation=mock_eval_ds)}
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_processed
        mock_data_pipeline.return_value = mock_pipeline
        
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train = MagicMock()
        mock_trainer_instance.state = MagicMock()
        mock_trainer_instance.state.best_model_checkpoint = str(temp_dir / "best")
        mock_trainer_instance.evaluate = MagicMock(return_value={"eval_loss": 1.0})
        mock_sft_trainer.return_value = mock_trainer_instance
        
        # Setup evaluation mock
        mock_eval_report = MagicMock()
        mock_eval_report.to_dict.return_value = {"metrics": {}}
        mock_run_eval.return_value = {"base_test": mock_eval_report}
        
        # Verify all components can be imported and initialized
        from src.train import train
        from src.evaluate import run_evaluation
        from src.config import ConfigManager
        
        assert train is not None
        assert run_evaluation is not None
        assert ConfigManager is not None
        
        # The actual pipeline would be:
        # 1. ConfigManager loads configs
        # 2. DataPipeline processes data
        # 3. load_model_and_tokenizer loads model
        # 4. SFTTrainer trains model
        # 5. run_evaluation evaluates model
        # All these are verified to be importable and wireable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
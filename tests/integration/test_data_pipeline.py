"""Integration tests for the data pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_pipeline import (
    AlpacaFormatter,
    ChatMLFormatter,
    DataPipeline,
    Llama3Formatter,
    PlainFormatter,
    VicunaFormatter,
    ZephyrFormatter,
)


class TestFormatterIntegration:
    """Test formatters with realistic data."""

    def test_alpaca_formatter_complete(self):
        """Test Alpaca formatter with all fields."""
        formatter = AlpacaFormatter()
        example = {
            "instruction": "Summarize the text",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "A fox jumps over a dog.",
        }
        result = formatter.format(example)
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert "Summarize the text" in result
        assert "The quick brown fox" in result
        assert "A fox jumps over a dog." in result

    def test_alpaca_formatter_no_input(self):
        """Test Alpaca formatter without input field."""
        formatter = AlpacaFormatter()
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello there!",
        }
        result = formatter.format(example)
        assert "### Input:" not in result
        assert "### Response:" in result

    def test_chatml_formatter(self):
        """Test ChatML formatter."""
        formatter = ChatMLFormatter(system_message="You are helpful.")
        example = {
            "instruction": "What is AI?",
            "input": "",
            "output": "AI is artificial intelligence.",
        }
        result = formatter.format(example)
        assert "im_start>system" in result
        assert "im_start>user" in result
        assert "im_start>assistant" in result
        assert "You are helpful." in result

    def test_llama3_formatter(self):
        """Test Llama-3 formatter."""
        formatter = Llama3Formatter()
        example = {
            "instruction": "Explain ML",
            "input": "in simple terms",
            "output": "ML is machine learning.",
        }
        result = formatter.format(example)
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_vicuna_formatter(self):
        """Test Vicuna formatter."""
        formatter = VicunaFormatter()
        example = {
            "instruction": "Write a poem",
            "input": "about cats",
            "output": "Cats are fluffy and cute.",
        }
        result = formatter.format(example)
        assert "USER:" in result
        assert "ASSISTANT:" in result

    def test_zephyr_formatter(self):
        """Test Zephyr formatter."""
        formatter = ZephyrFormatter()
        example = {
            "instruction": "Hello",
            "input": "",
            "output": "Hi there!",
        }
        result = formatter.format(example)
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result

    def test_plain_formatter(self):
        """Test Plain formatter."""
        formatter = PlainFormatter(text_key="text")
        example = {"text": "Just some text"}
        result = formatter.format(example)
        assert result == "Just some text"

    def test_custom_formatter(self):
        """Test Custom formatter."""
        from src.data_pipeline import CustomFormatter

        formatter = CustomFormatter(
            template="{instruction} -> {output}",
            template_with_input="{instruction} [{input}] -> {output}",
        )
        ex1 = {"instruction": "A", "input": "", "output": "B"}
        ex2 = {"instruction": "A", "input": "C", "output": "B"}
        assert formatter.format(ex1) == "A -> B"
        assert formatter.format(ex2) == "A [C] -> B"

    def test_formatter_batch(self):
        """Test batch formatting."""
        formatter = AlpacaFormatter()
        examples = {
            "instruction": ["Task 1", "Task 2"],
            "input": ["", "Input 2"],
            "output": ["Response 1", "Response 2"],
        }
        results = formatter.format_batch(examples)
        assert len(results) == 2
        assert "Task 1" in results[0]
        assert "Input 2" in results[1]


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""

    def test_pipeline_initialization(self, temp_dir):
        """Test pipeline initialization with config."""
        config_dict = {
            "datasets": [
                {
                    "name": "test_dataset",
                    "path": "tatsu-lab/alpaca",
                    "split": "train[:10]",
                    "max_samples": 10,
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
                    "eos_token": "\n",
                }
            },
            "default_template": "alpaca",
            "processing": {
                "download": {"cache_dir": str(temp_dir / "raw")},
                "validation": {"enabled": True, "required_columns": ["instruction", "output"]},
                "cleaning": {"enabled": True, "strip_whitespace": True},
                "formatting": {"enabled": True, "template": "alpaca", "formatted_field": "text"},
                "tokenization": {"enabled": False},
                "splitting": {
                    "enabled": True,
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                },
            },
            "output": {
                "output_dir": str(temp_dir / "processed"),
                "formats": ["jsonl"],
                "save_splits": True,
            },
        }
        pipeline = DataPipeline(config_dict)
        assert pipeline.config is not None
        assert len(pipeline.config.datasets) == 1

    @patch("src.data_pipeline.load_dataset")
    def test_download_dataset(self, mock_load_dataset, temp_dir):
        """Test dataset downloading."""
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.column_names = ["instruction", "input", "output"]
        mock_ds.select = Mock(return_value=mock_ds)
        mock_load_dataset.return_value = mock_ds

        config_dict = {
            "datasets": [
                {
                    "name": "alpaca",
                    "path": "tatsu-lab/alpaca",
                    "split": "train",
                    "max_samples": 10,
                }
            ],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {"download": {"cache_dir": str(temp_dir / "raw")}},
            "output": {"output_dir": str(temp_dir / "processed")},
        }
        pipeline = DataPipeline(config_dict)

        with patch.object(pipeline, "load_local_datasets", return_value={}):
            datasets = pipeline.download_datasets()

        assert "alpaca" in datasets
        mock_load_dataset.assert_called_once()

    def test_apply_column_mapping(self, temp_dir):
        """Test column mapping application."""
        from datasets import Dataset

        config_dict = {
            "datasets": [
                {
                    "name": "test",
                    "path": "test",
                    "column_mapping": {
                        "instruction": "prompt",
                        "input": "context",
                        "output": "completion",
                    },
                }
            ],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {},
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict(
            {
                "prompt": ["test"],
                "context": [""],
                "completion": ["result"],
            }
        )
        result = pipeline.apply_column_mapping(ds, pipeline.config.datasets[0].column_mapping)

        assert "instruction" in result.column_names
        assert "input" in result.column_names
        assert "output" in result.column_names

    def test_validate_dataset(self, temp_dir):
        """Test dataset validation."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {
                "validation": {
                    "enabled": True,
                    "required_columns": ["instruction", "output"],
                    "min_instruction_length": 5,
                    "min_output_length": 3,
                    "drop_nulls": True,
                }
            },
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        # Valid dataset
        ds = Dataset.from_dict(
            {
                "instruction": ["Valid instruction", "Short"],
                "input": ["", ""],
                "output": ["Valid output", "Ok"],
            }
        )
        result = pipeline.validate_dataset(ds, "test")
        assert len(result) == 1  # Second filtered out (too short)
        assert result[0]["instruction"] == "Valid instruction"

    def test_clean_dataset(self, temp_dir):
        """Test dataset cleaning."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {
                "cleaning": {
                    "enabled": True,
                    "strip_whitespace": True,
                    "normalize_unicode": True,
                    "remove_html": True,
                    "remove_nulls": True,
                }
            },
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict(
            {
                "instruction": ["  test  ", "  another  "],
                "input": ["", ""],
                "output": ["  result  ", "  more  "],
            }
        )
        result = pipeline.clean_dataset(ds, "test")

        assert result["instruction"][0] == "test"
        assert result["output"][0] == "result"

    def test_format_dataset(self, temp_dir):
        """Test dataset formatting."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {
                "alpaca": {
                    "template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
                    "template_with_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
                    "instruction_key": "instruction",
                    "input_key": "input",
                    "output_key": "output",
                    "add_eos_token": True,
                    "eos_token": "\n",
                }
            },
            "default_template": "alpaca",
            "processing": {
                "formatting": {"enabled": True, "template": "alpaca", "formatted_field": "text"},
                "download": {"num_proc": 1},
            },
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict(
            {
                "instruction": ["Test instruction"],
                "input": [""],
                "output": ["Test output"],
            }
        )
        result = pipeline.format_dataset(ds, "test")

        assert "text" in result.column_names
        assert "### Instruction:" in result["text"][0]
        assert "### Response:" in result["text"][0]

    def test_split_dataset(self, temp_dir):
        """Test dataset splitting."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {
                "splitting": {
                    "enabled": True,
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                    "seed": 42,
                }
            },
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict({"text": [f"sample {i}" for i in range(100)]})
        splits = pipeline.split_dataset(ds)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert len(splits["train"]) == 80
        assert len(splits["validation"]) == 10
        assert len(splits["test"]) == 10

    def test_compute_statistics(self, temp_dir):
        """Test statistics computation."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {
                "statistics": {"enabled": True, "sample_size": 10},
            },
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict(
            {
                "instruction": ["Short", "Medium length instruction", "A" * 100],
                "input": ["", "", ""],
                "output": ["Out", "Medium output text", "B" * 50],
            }
        )
        stats = pipeline.compute_statistics(ds, "test")

        assert stats.num_samples == 3
        assert len(stats.instruction_lengths) == 3

    def test_export_jsonl(self, temp_dir):
        """Test JSONL export."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {},
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        output_path = temp_dir / "test.jsonl"
        pipeline.export_jsonl(ds, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_export_arrow(self, temp_dir):
        """Test Arrow export."""
        from datasets import Dataset

        config_dict = {
            "datasets": [{"name": "test", "path": "test"}],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {},
            "output": {"output_dir": str(temp_dir)},
        }
        pipeline = DataPipeline(config_dict)

        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        output_path = temp_dir / "test_arrow"
        pipeline.export_arrow(ds, output_path)

        assert output_path.exists()
        assert (output_path / "dataset_info.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

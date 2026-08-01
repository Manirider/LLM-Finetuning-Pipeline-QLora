#!/usr/bin/env python
"""
Unit tests for Data Pipeline
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_pipeline import (
    DataPipeline,
    DatasetStatistics,
    PromptFormatter,
    AlpacaFormatter,
    ChatMLFormatter,
    Llama3Formatter,
    VicunaFormatter,
    ZephyrFormatter,
    PlainFormatter,
    CustomFormatter,
    get_formatter,
    FORMATTERS,
    PARQUET_AVAILABLE,
)


# Module-level fixture for temp_config
@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    config_content = """
datasets:
  - name: "test_dataset"
    path: "tatsu-lab/alpaca"
    split: "train[:10]"
    max_samples: 10
    column_mapping:
      instruction: "instruction"
      input: "input"
      output: "output"

prompt_templates:
  alpaca:
    template: "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    template_with_input: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    instruction_key: "instruction"
    input_key: "input"
    output_key: "output"
    add_eos_token: true

default_template: "alpaca"

processing:
  download:
    cache_dir: "./data/raw"
    num_proc: 1
  validation:
    enabled: true
    required_columns: ["instruction", "output"]
    min_instruction_length: 1
    min_output_length: 1
  cleaning:
    enabled: true
    strip_whitespace: true
    normalize_unicode: true
  formatting:
    enabled: true
    template: "alpaca"
    add_eos_token: true
    formatted_field: "text"
  tokenization:
    enabled: false
  splitting:
    enabled: true
    ratios:
      train: 0.8
      validation: 0.1
      test: 0.1
    seed: 42

output:
  output_dir: "./data/processed"
  formats: ["jsonl"]
  save_splits: true
  filenames:
    train: "train.jsonl"
    validation: "val.jsonl"
    test: "test.jsonl"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


class TestFormatters:
    """Test all prompt formatters."""

    def test_alpaca_formatter_with_input(self):
        formatter = AlpacaFormatter()
        example = {
            "instruction": "Write a summary",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "A quick brown fox jumps over a lazy dog.",
        }
        result = formatter.format(example)
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert "Write a summary" in result
        assert "The quick brown fox" in result
        assert "A quick brown fox" in result

    def test_alpaca_formatter_without_input(self):
        formatter = AlpacaFormatter()
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello there!",
        }
        result = formatter.format(example)
        assert "### Instruction:" in result
        assert "### Input:" not in result
        assert "### Response:" in result

    def test_alpaca_formatter_with_eos(self):
        formatter = AlpacaFormatter(add_eos_token=True, eos_token="\n")
        example = {"instruction": "Test", "input": "", "output": "Output"}
        result = formatter.format(example)
        assert result.endswith("\n")

    def test_alpaca_formatter_batch(self):
        formatter = AlpacaFormatter()
        examples = {
            "instruction": ["Task 1", "Task 2"],
            "input": ["", "Input 2"],
            "output": ["Response 1", "Response 2"],
        }
        results = formatter.format_batch(examples)
        assert len(results) == 2
        assert "### Instruction:" in results[0]
        assert "### Input:" not in results[0]
        assert "### Input:" in results[1]

    def test_chatml_formatter(self):
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

    def test_chatml_formatter_with_input(self):
        formatter = ChatMLFormatter()
        example = {
            "instruction": "Summarize",
            "input": "Text to summarize",
            "output": "Summary",
        }
        result = formatter.format(example)
        assert "Text to summarize" in result

    def test_llama3_formatter(self):
        formatter = Llama3Formatter()
        example = {
            "instruction": "Explain ML",
            "input": "in simple terms",
            "output": "ML is...",
        }
        result = formatter.format(example)
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_vicuna_formatter(self):
        formatter = VicunaFormatter()
        example = {
            "instruction": "Write a poem",
            "input": "about cats",
            "output": "Cats are great...",
        }
        result = formatter.format(example)
        assert "USER:" in result
        assert "ASSISTANT:" in result

    def test_zephyr_formatter(self):
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
        formatter = PlainFormatter(text_key="text")
        example = {"text": "Just some text"}
        result = formatter.format(example)
        assert result == "Just some text"

    def test_custom_formatter(self):
        formatter = CustomFormatter(
            template="{instruction} -> {output}",
            template_with_input="{instruction} [{input}] -> {output}",
        )
        ex1 = {"instruction": "A", "input": "", "output": "B"}
        ex2 = {"instruction": "A", "input": "C", "output": "B"}
        assert formatter.format(ex1) == "A -> B"
        assert formatter.format(ex2) == "A [C] -> B"

    def test_get_formatter(self):
        for name in FORMATTERS:
            formatter = get_formatter(name)
            assert isinstance(formatter, PromptFormatter)

    def test_get_formatter_invalid(self):
        with pytest.raises(ValueError):
            get_formatter("nonexistent")


class TestDatasetStatistics:
    """Test DatasetStatistics dataclass."""

    def test_empty_stats(self):
        stats = DatasetStatistics()
        d = stats.to_dict()
        assert d["num_samples"] == 0
        assert d["instruction_lengths"]["mean"] == 0

    def test_stats_with_data(self):
        stats = DatasetStatistics(
            num_samples=100,
            instruction_lengths=[10, 20, 30, 40, 50],
            output_lengths=[5, 15, 25],
        )
        d = stats.to_dict()
        assert d["num_samples"] == 100
        assert d["instruction_lengths"]["mean"] == 30.0
        assert d["instruction_lengths"]["min"] == 10
        assert d["instruction_lengths"]["max"] == 50


class TestDataPipeline:
    """Test DataPipeline class."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_content = """
datasets:
  - name: "test_dataset"
    path: "tatsu-lab/alpaca"
    split: "train[:10]"
    max_samples: 10
    column_mapping:
      instruction: "instruction"
      input: "input"
      output: "output"

prompt_templates:
  alpaca:
    template: "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    template_with_input: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    instruction_key: "instruction"
    input_key: "input"
    output_key: "output"
    add_eos_token: true

default_template: "alpaca"

processing:
  download:
    cache_dir: "./data/raw"
    num_proc: 1
  validation:
    enabled: true
    required_columns: ["instruction", "output"]
    min_instruction_length: 1
    min_output_length: 1
  cleaning:
    enabled: true
    strip_whitespace: true
    normalize_unicode: true
  formatting:
    enabled: true
    template: "alpaca"
    add_eos_token: true
    formatted_field: "text"
  tokenization:
    enabled: false
  splitting:
    enabled: true
    ratios:
      train: 0.8
      validation: 0.1
      test: 0.1
    seed: 42

output:
  output_dir: "./data/processed"
  formats: ["jsonl"]
  save_splits: true
  filenames:
    train: "train.jsonl"
    validation: "val.jsonl"
    test: "test.jsonl"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_pipeline_init_from_yaml(self, temp_config):
        pipeline = DataPipeline(temp_config)
        assert pipeline.config is not None
        assert len(pipeline.config.datasets) == 1
        assert pipeline.config.datasets[0].name == "test_dataset"

    def test_pipeline_init_from_dict(self):
        config_dict = {
            "datasets": [
                {"name": "test", "path": "test/path", "split": "train"}
            ],
            "prompt_templates": {},
            "default_template": "alpaca",
            "processing": {},
            "output": {"output_dir": "./out", "formats": ["jsonl"]},
        }
        pipeline = DataPipeline(config_dict)
        assert pipeline.config.datasets[0].name == "test"

    def test_apply_column_mapping(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        from src.config import ColumnMappingConfig
        ds = Dataset.from_dict({
            "prompt": ["test"],
            "context": [""],
            "completion": ["result"],
        })
        mapping = ColumnMappingConfig(instruction="prompt", input="context", output="completion")
        result = pipeline.apply_column_mapping(ds, mapping)

        assert "instruction" in result.column_names
        assert "input" in result.column_names
        assert "output" in result.column_names

    @patch("src.data_pipeline.load_dataset")
    def test_download_datasets(self, mock_load_dataset, temp_config):
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.column_names = ["instruction", "input", "output"]
        mock_ds.select = Mock(return_value=mock_ds)
        mock_load_dataset.return_value = mock_ds

        pipeline = DataPipeline(temp_config)

        with patch.object(pipeline, "load_local_datasets", return_value={}):
            datasets = pipeline.download_datasets()
        assert "test_dataset" in datasets

    def test_validate_dataset(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({
            "instruction": ["valid instruction", ""],
            "input": ["", ""],
            "output": ["valid output", "y"],
        })
        result = pipeline.validate_dataset(ds, "test")
        assert len(result) == 1  # Empty instruction filtered out

    def test_clean_dataset(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({
            "instruction": ["  test  ", "  another  "],
            "input": ["", ""],
            "output": ["  result  ", "  more  "],
        })
        result = pipeline.clean_dataset(ds, "test")

        assert result["instruction"][0] == "test"
        assert result["output"][0] == "result"

    def test_format_dataset(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({
            "instruction": ["Write a poem"],
            "input": [""],
            "output": ["Roses are red"],
        })
        result = pipeline.format_dataset(ds, "test")

        assert "text" in result.column_names
        assert "### Instruction:" in result["text"][0]
        assert "### Response:" in result["text"][0]

    def test_split_dataset(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": [f"sample {i}" for i in range(100)]})
        splits = pipeline.split_dataset(ds)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert len(splits["train"]) == 80
        assert len(splits["validation"]) == 10
        assert len(splits["test"]) == 10

    def test_compute_statistics(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({
            "instruction": ["Short", "Medium length instruction", "A" * 100],
            "input": ["", "", ""],
            "output": ["Out", "Medium output text", "B" * 50],
        })
        stats = pipeline.compute_statistics(ds, "test")

        assert stats.num_samples == 3
        assert len(stats.instruction_lengths) == 3

    def test_analyze_sequences(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [1, 2], [1, 2, 3, 4, 5]],
        })
        stats = pipeline.analyze_sequences(ds)

        assert stats["count"] == 3
        assert stats["mean"] == 10/3

    def test_export_jsonl(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name
        try:
            pipeline.export_jsonl(ds, temp_path)
            with open(temp_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
        finally:
            os.unlink(temp_path)

    def test_export_parquet(self, temp_config):
        if not PARQUET_AVAILABLE:
            pytest.skip("pyarrow not available")
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": ["a", "b"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
            temp_path = f.name
        try:
            pipeline.export_parquet(ds, temp_path)
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_arrow(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test")
            pipeline.export_arrow(ds, output_path)

            assert os.path.exists(os.path.join(output_path, "dataset_info.json"))

    def test_export_dataset(self, temp_config):
        pipeline = DataPipeline(temp_config)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test")
            pipeline.export_dataset(ds, output_path)
            # Check that at least one file was created
            files = os.listdir(tmpdir)
            assert len([f for f in files if f.startswith("test")]) > 0


class TestCLI:
    """Test CLI argument parsing."""

    def test_create_argument_parser(self):
        from src.data_pipeline import create_argument_parser
        parser = create_argument_parser()
        args = parser.parse_args(["--config", "configs/data.yaml"])
        assert args.config == "configs/data.yaml"

    def test_parser_defaults(self):
        from src.data_pipeline import create_argument_parser
        parser = create_argument_parser()
        args = parser.parse_args([])
        assert args.config == "configs/data.yaml"
        assert args.dataset is None
        assert args.verbose is False


class TestIntegration:
    """Integration-style tests (mocked)."""

    @patch("src.data_pipeline.load_dataset")
    def test_full_pipeline(self, mock_load_dataset, temp_config):
        from datasets import Dataset
        real_ds = Dataset.from_dict({
            "instruction": ["Instruction 1", "Instruction 2"],
            "input": ["", ""],
            "output": ["Output 1", "Output 2"],
        })
        mock_load_dataset.return_value = real_ds

        pipeline = DataPipeline(temp_config)
        pipeline.config.processing["tokenization"] = {"enabled": False}
        pipeline.config.processing["download"] = {"num_proc": 1}

        with patch.object(pipeline, "load_local_datasets", return_value={}):
            with patch.object(pipeline, "export_dataset") as mock_export:
                result = pipeline.process()

        assert mock_export.called


class TestCLI:
    """Test CLI argument parsing."""

    def test_create_argument_parser(self):
        from src.data_pipeline import create_argument_parser
        parser = create_argument_parser()
        args = parser.parse_args(["--config", "configs/data.yaml"])
        assert args.config == "configs/data.yaml"

    def test_parser_defaults(self):
        from src.data_pipeline import create_argument_parser
        parser = create_argument_parser()
        args = parser.parse_args([])
        assert args.config == "configs/data.yaml"
        assert args.dataset is None
        assert args.verbose is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
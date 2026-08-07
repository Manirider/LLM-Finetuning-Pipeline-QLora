"""
Data Pipeline for LLM Fine-tuning

Comprehensive data processing pipeline including:
- Dataset downloading and caching
- Validation
- Cleaning
- Deduplication
- Statistics collection
- Prompt formatting (ChatML, Alpaca, etc.)
- Tokenization
- Sequence analysis
- Train/Validation/Test split
- Multi-format export (JSONL, Parquet, Arrow)
- CLI support
- Logging
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

from src.config import ConfigManager, DataConfigComplete

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Statistics for a dataset."""

    num_samples: int = 0
    num_columns: int = 0
    column_names: list[str] = field(default_factory=list)
    instruction_lengths: list[int] = field(default_factory=list)
    input_lengths: list[int] = field(default_factory=list)
    output_lengths: list[int] = field(default_factory=list)
    total_lengths: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)
    percentiles: dict[str, dict[int, float]] = field(default_factory=dict)
    duplicates_count: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    empty_string_counts: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def stats_for(lengths: list[int]) -> dict[str, Any]:
            if not lengths:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "percentiles": {}}
            return {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "percentiles": {
                    str(p): float(np.percentile(lengths, p))
                    for p in [0, 25, 50, 75, 90, 95, 99, 100]
                },
            }

        return {
            "num_samples": self.num_samples,
            "num_columns": self.num_columns,
            "column_names": self.column_names,
            "instruction_lengths": stats_for(self.instruction_lengths),
            "input_lengths": stats_for(self.input_lengths),
            "output_lengths": stats_for(self.output_lengths),
            "total_lengths": stats_for(self.total_lengths),
            "token_lengths": stats_for(self.token_lengths),
            "percentiles": {
                k: {str(kk): vv for kk, vv in v.items()} for k, v in self.percentiles.items()
            },
            "duplicates_count": self.duplicates_count,
            "null_counts": self.null_counts,
            "empty_string_counts": self.empty_string_counts,
            "language_distribution": self.language_distribution,
        }


class PromptFormatter(ABC):
    """Abstract base class for prompt formatters."""

    @abstractmethod
    def format(self, example: dict[str, Any]) -> str:
        """Format a single example into a prompt string."""

    @abstractmethod
    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        """Format a batch of examples."""


class AlpacaFormatter(PromptFormatter):
    """Alpaca format: Instruction + Input + Output."""

    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


class ChatMLFormatter(PromptFormatter):
    """ChatML format: System + User + Assistant."""

    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        system_message: str = "You are a helpful assistant.",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.system_message = system_message
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        prompt = (
            f"im_start>system\n{self.system_message}im_end>\n"
            f"im_start>user\n{user_content}im_end>\n"
            f"im_start>assistant\n{output}im_end>"
        )

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


class Llama3Formatter(PromptFormatter):
    """Llama-3 format with special tokens."""

    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        system_message: str = "You are a helpful assistant.",
        add_eos_token: bool = True,
        eos_token: str = "<|eot_id|>",
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.system_message = system_message
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.system_message}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


class VicunaFormatter(PromptFormatter):
    """Vicuna format."""

    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        system_message: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.system_message = system_message
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        prompt = f"{self.system_message}\n\nUSER: {user_content}\nASSISTANT: {output}"

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


class ZephyrFormatter(PromptFormatter):
    """Zephyr format."""

    def __init__(
        self,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        system_message: str = "You are a helpful assistant.",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.system_message = system_message
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        prompt = (
            f"<|system|>\n{self.system_message}im_end>\n"
            f"<|user|>\n{user_content}im_end>\n"
            f"<|assistant|>\n{output}im_end>"
        )

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


class PlainFormatter(PromptFormatter):
    """Plain text format for pre-training."""

    def __init__(
        self,
        text_key: str = "text",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.text_key = text_key
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        text = str(example.get(self.text_key, "")).strip()
        if self.add_eos_token and self.eos_token:
            text += self.eos_token
        return text

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        texts = examples[self.text_key]
        if self.add_eos_token and self.eos_token:
            return [t + self.eos_token for t in texts]
        return texts


class CustomFormatter(PromptFormatter):
    """Custom format with user-defined templates."""

    def __init__(
        self,
        template: str = "{instruction}\n\n{output}",
        template_with_input: str = "{instruction}\n\n{input}\n\n{output}",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        add_eos_token: bool = True,
        eos_token: str = "",
    ):
        self.template = template
        self.template_with_input = template_with_input
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

    def format(self, example: dict[str, Any]) -> str:
        instruction = str(example.get(self.instruction_key, "")).strip()
        input_text = str(example.get(self.input_key, "")).strip()
        output = str(example.get(self.output_key, "")).strip()

        if input_text:
            prompt = self.template_with_input.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        else:
            prompt = self.template.format(
                instruction=instruction,
                output=output,
            )

        if self.add_eos_token and self.eos_token:
            prompt += self.eos_token

        return prompt

    def format_batch(self, examples: dict[str, list[Any]]) -> list[str]:
        batch_size = len(examples[self.instruction_key])
        results = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            results.append(self.format(example))
        return results


FORMATTERS: dict[str, type] = {
    "alpaca": AlpacaFormatter,
    "chatml": ChatMLFormatter,
    "llama3": Llama3Formatter,
    "vicuna": VicunaFormatter,
    "zephyr": ZephyrFormatter,
    "plain": PlainFormatter,
    "custom": CustomFormatter,
}


def get_formatter(name: str, **kwargs) -> PromptFormatter:
    """Get a formatter by name."""
    if name not in FORMATTERS:
        raise ValueError(f"Unknown formatter: {name}. Available: {list(FORMATTERS.keys())}")
    return FORMATTERS[name](**kwargs)


class DataPipeline:
    """Main data processing pipeline."""

    def __init__(self, config: DataConfigComplete | dict[str, Any] | str | Path):
        """
        Initialize the data pipeline.

        Args:
            config: Configuration object, dict, or path to YAML config file.
        """
        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.is_file():
                import yaml

                with open(path) as f:
                    raw_dict = yaml.safe_load(f)
                self.config = DataConfigComplete(**raw_dict)
                self.config_manager = None
            else:
                self.config_manager = ConfigManager(config_dir=str(path))
                self.config = self.config_manager.data
        elif isinstance(config, dict):
            self.config = DataConfigComplete(**config)
            self.config_manager = None
        else:
            self.config = config
            self.config_manager = None

        self.tokenizer: PreTrainedTokenizer | None = None
        self.formatter: PromptFormatter | None = None
        self.statistics = DatasetStatistics()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def download_datasets(self) -> dict[str, Dataset]:
        """
        Download datasets from Hugging Face Hub or local paths.

        Returns:
            Dictionary mapping dataset names to loaded datasets.
        """
        logger.info("Starting dataset download...")
        datasets = {}
        download_config = self.config.processing.get("download", {})
        cache_dir = download_config.get("cache_dir", "./data/raw")
        force_redownload = download_config.get("force_redownload", False)
        download_config.get("resume_download", True)
        num_proc = download_config.get("num_proc", 4)
        max_retries = download_config.get("max_retries", 3)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        for ds_config in self.config.datasets:
            name = ds_config.name
            logger.info(f"Loading dataset: {name}")

            for attempt in range(max_retries):
                try:
                    if ds_config.streaming:
                        dataset = load_dataset(
                            ds_config.path,
                            name=ds_config.config_name,
                            data_files=ds_config.data_files,
                            split=ds_config.split,
                            streaming=True,
                            cache_dir=cache_dir,
                        )
                    else:
                        dataset = load_dataset(
                            ds_config.path,
                            name=ds_config.config_name,
                            data_files=ds_config.data_files,
                            split=ds_config.split,
                            cache_dir=cache_dir,
                            download_mode=(
                                "force_redownload"
                                if force_redownload
                                else "reuse_dataset_if_exists"
                            ),
                            num_proc=num_proc,
                        )

                    if ds_config.max_samples and len(dataset) > ds_config.max_samples:
                        dataset = dataset.select(range(ds_config.max_samples))

                    datasets[name] = dataset
                    logger.info(f"Loaded {name}: {len(dataset)} samples")
                    break

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {name}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)

        return datasets

    def load_local_datasets(self) -> dict[str, Dataset]:
        """Load datasets from local files (JSONL, CSV, Parquet)."""
        logger.info("Loading local datasets...")
        datasets = {}

        external = self.config.processing.get("external_sources", {})

        if external.get("local_jsonl", {}).get("enabled", False):
            paths = external["local_jsonl"]
            for split_name, path in [
                ("train", paths.get("train_path")),
                ("validation", paths.get("val_path")),
                ("test", paths.get("test_path")),
            ]:
                if path and Path(path).exists():
                    datasets[f"local_{split_name}"] = load_dataset(
                        "json", data_files=path, split="train"
                    )

        if external.get("local_csv", {}).get("enabled", False):
            paths = external["local_csv"]
            text_column = paths.get("text_column", "text")
            for split_name, path in [
                ("train", paths.get("train_path")),
                ("validation", paths.get("val_path")),
                ("test", paths.get("test_path")),
            ]:
                if path and Path(path).exists():
                    ds = load_dataset("csv", data_files=path, split="train")
                    if text_column in ds.column_names:
                        ds = ds.rename_column(text_column, "text")
                    datasets[f"local_csv_{split_name}"] = ds

        if external.get("local_parquet", {}).get("enabled", False):
            paths = external["local_parquet"]
            for split_name, path in [
                ("train", paths.get("train_path")),
                ("validation", paths.get("val_path")),
                ("test", paths.get("test_path")),
            ]:
                if path and Path(path).exists():
                    datasets[f"local_parquet_{split_name}"] = load_dataset(
                        "parquet", data_files=path, split="train"
                    )

        if external.get("webdataset", {}).get("enabled", False):
            urls = external["webdataset"].get("urls", [])
            if urls:
                logger.warning("WebDataset support requires additional setup, skipping...")

        return datasets

    def apply_column_mapping(self, dataset: Dataset, column_mapping: Any) -> Dataset:
        """Apply column mapping to standardize dataset columns."""
        if not hasattr(column_mapping, "instruction"):
            return dataset

        mapping = {}
        if column_mapping.instruction and column_mapping.instruction != "instruction":
            mapping[column_mapping.instruction] = "instruction"
        if column_mapping.input and column_mapping.input != "input":
            mapping[column_mapping.input] = "input"
        if column_mapping.output and column_mapping.output != "output":
            mapping[column_mapping.output] = "output"
        if column_mapping.text and column_mapping.text != "text":
            mapping[column_mapping.text] = "text"

        if mapping:
            dataset = dataset.rename_columns(mapping)
        return dataset

    def validate_dataset(self, dataset: Dataset, dataset_name: str = "dataset") -> Dataset:
        """
        Validate dataset against configuration rules.

        Args:
            dataset: Dataset to validate.
            dataset_name: Name for logging.

        Returns:
            Validated dataset.
        """
        val_config = self.config.processing.get("validation", {})
        if not val_config.get("enabled", True):
            return dataset

        logger.info(f"Validating {dataset_name}...")

        required_columns = val_config.get("required_columns", ["instruction", "output"])
        missing_cols = [c for c in required_columns if c not in dataset.column_names]
        if missing_cols:
            raise ValueError(f"Missing required columns in {dataset_name}: {missing_cols}")

        min_inst_len = val_config.get("min_instruction_length", 10)
        max_inst_len = val_config.get("max_instruction_length", 8192)
        min_out_len = val_config.get("min_output_length", 5)
        max_out_len = val_config.get("max_output_length", 16384)

        def is_valid(example):
            inst = example.get("instruction", "")
            out = example.get("output", "")

            if val_config.get("drop_nulls", True):
                if inst is None or out is None:
                    return False

            if val_config.get("drop_empty_strings", True):
                if not str(inst).strip() or not str(out).strip():
                    return False

            inst_len = len(str(inst))
            out_len = len(str(out))

            if inst_len < min_inst_len or inst_len > max_inst_len:
                return False
            if out_len < min_out_len or out_len > max_out_len:
                return False

            return True

        original_len = len(dataset)
        num_proc = self.config.processing.get("download", {}).get("num_proc", 4)
        dataset = dataset.filter(is_valid, num_proc=num_proc)
        filtered = original_len - len(dataset)

        if filtered > 0:
            logger.info(f"Filtered {filtered} invalid samples from {dataset_name}")

        if val_config.get("check_duplicates", True):
            subset = val_config.get("duplicate_subset", ["instruction", "input", "output"])
            subset = [c for c in subset if c in dataset.column_names]
            if subset and len(dataset) > 0:
                original_len = len(dataset)
                df = dataset.to_pandas()
                df = df.drop_duplicates(subset=subset)
                dataset = Dataset.from_pandas(df)
                if "__index_level_0__" in dataset.column_names:
                    dataset = dataset.remove_columns(["__index_level_0__"])
                dup_count = original_len - len(dataset)
                if dup_count > 0:
                    logger.info(f"Removed {dup_count} duplicates from {dataset_name}")
                    self.statistics.duplicates_count += dup_count

        if val_config.get("detect_language", False):
            try:
                from langdetect import detect

                langs = Counter()
                for ex in dataset.select(range(min(1000, len(dataset)))):
                    text = str(ex.get("instruction", "")) + " " + str(ex.get("output", ""))
                    try:
                        lang = detect(text)
                        langs[lang] += 1
                    except Exception:
                        pass
                self.statistics.language_distribution = dict(langs)
                expected = val_config.get("expected_language", "en")
                if expected not in langs:
                    logger.warning(f"Expected language {expected} not detected in {dataset_name}")
            except ImportError:
                logger.warning("langdetect not installed, skipping language detection")

        return dataset

    def clean_dataset(self, dataset: Dataset, dataset_name: str = "dataset") -> Dataset:
        """
        Clean dataset according to configuration.

        Args:
            dataset: Dataset to clean.
            dataset_name: Name for logging.

        Returns:
            Cleaned dataset.
        """
        clean_config = self.config.processing.get("cleaning", {})
        if not clean_config.get("enabled", True):
            return dataset

        logger.info(f"Cleaning {dataset_name}...")

        strip_ws = clean_config.get("strip_whitespace", True)
        norm_unicode = clean_config.get("normalize_unicode", True)
        rem_html = clean_config.get("remove_html", False)

        def clean_example(example):
            for key, value in example.items():
                if isinstance(value, str):
                    if strip_ws:
                        value = value.strip()
                    if norm_unicode:
                        import unicodedata

                        value = unicodedata.normalize("NFKC", value)
                    if rem_html:
                        import re

                        value = re.sub(r"<[^>]+>", "", value)
                example[key] = value
            return example

        num_proc = self.config.processing.get("download", {}).get("num_proc", 4)
        dataset = dataset.map(clean_example, num_proc=num_proc)

        if clean_config.get("remove_nulls", True):

            def has_no_nulls(example):
                return all(v is not None for v in example.values())

            dataset = dataset.filter(has_no_nulls)

        if clean_config.get("remove_duplicates", True):
            subset = clean_config.get("duplicate_subset", ["instruction", "input", "output"])
            subset = [c for c in subset if c in dataset.column_names]
            if subset and len(dataset) > 0:
                original_len = len(dataset)
                df = dataset.to_pandas()
                df = df.drop_duplicates(subset=subset)
                dataset = Dataset.from_pandas(df)
                if "__index_level_0__" in dataset.column_names:
                    dataset = dataset.remove_columns(["__index_level_0__"])
                dup_count = original_len - len(dataset)
                if dup_count > 0:
                    logger.info(
                        f"Removed {dup_count} duplicates during cleaning from {dataset_name}"
                    )

        for cleaner_name in clean_config.get("custom_cleaners", []):
            if hasattr(self, f"_custom_clean_{cleaner_name}"):
                dataset = dataset.map(getattr(self, f"_custom_clean_{cleaner_name}"))

        return dataset

    def format_dataset(self, dataset: Dataset, dataset_name: str = "dataset") -> Dataset:
        """
        Format dataset using the configured prompt template.

        Args:
            dataset: Dataset to format.
            dataset_name: Name for logging.

        Returns:
            Formatted dataset with 'text' column.
        """
        format_config = self.config.processing.get("formatting", {})
        if not format_config.get("enabled", True):
            return dataset

        logger.info(
            f"Formatting {dataset_name} with template: {format_config.get('template', 'alpaca')}"
        )

        template_name = format_config.get("template", "alpaca")
        system_message = format_config.get("system_message", "You are a helpful assistant.")
        format_config.get("include_input", True)
        add_eos_token = format_config.get("add_eos_token", True)
        formatted_field = format_config.get("formatted_field", "text")
        keep_original = format_config.get("keep_original_columns", False)

        eos_token = ""
        if self.tokenizer and hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
            eos_token = self.tokenizer.eos_token

        prompt_templates = self.config.prompt_templates
        if template_name not in prompt_templates:
            template_name = self.config.default_template

        template_config = prompt_templates.get(template_name, {})

        def _get_cfg_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Get keys from template config
        instruction_key = (
            _get_cfg_val(template_config, "instruction_key", "instruction") or "instruction"
        )
        input_key = _get_cfg_val(template_config, "input_key", "input") or "input"
        output_key = _get_cfg_val(template_config, "output_key", "output") or "output"

        if template_name == "chatml":
            self.formatter = ChatMLFormatter(
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                system_message=system_message,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        elif template_name == "llama3":
            self.formatter = Llama3Formatter(
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                system_message=system_message,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        elif template_name == "vicuna":
            self.formatter = VicunaFormatter(
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                system_message=system_message,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        elif template_name == "zephyr":
            self.formatter = ZephyrFormatter(
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                system_message=system_message,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        elif template_name == "plain":
            self.formatter = PlainFormatter(
                text_key=template_config.get("text_key", "text"),
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        elif template_name == "custom":
            self.formatter = CustomFormatter(
                template=template_config.get("template", "{instruction}\n\n{output}"),
                template_with_input=template_config.get(
                    "template_with_input", "{instruction}\n\n{input}\n\n{output}"
                ),
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )
        else:
            self.formatter = AlpacaFormatter(
                instruction_key=instruction_key,
                input_key=input_key,
                output_key=output_key,
                add_eos_token=add_eos_token,
                eos_token=eos_token,
            )

        formatter = self.formatter

        def format_batch(examples):
            texts = formatter.format_batch(examples)
            return {formatted_field: texts}

        remove_columns = (
            [] if keep_original else [c for c in dataset.column_names if c != formatted_field]
        )
        num_proc = self.config.processing.get("download", {}).get("num_proc", 4)
        dataset = dataset.map(
            format_batch,
            batched=True,
            batch_size=1000,
            remove_columns=remove_columns,
            num_proc=num_proc,
        )

        return dataset

    def load_tokenizer(self, tokenizer_name_or_path: str | None = None) -> PreTrainedTokenizer:
        """
        Load tokenizer for tokenization.

        Args:
            tokenizer_name_or_path: Path or name of tokenizer. Uses config if not provided.

        Returns:
            Loaded tokenizer.
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("transformers and tokenizers required for tokenization")

        if tokenizer_name_or_path is None:
            if self.config_manager:
                tokenizer_name_or_path = self.config_manager.model.tokenizer.tokenizer_name_or_path
            else:
                tokenizer_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

        logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            padding_side="right",
            truncation_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def tokenize_dataset(
        self,
        dataset: Dataset,
        dataset_name: str = "dataset",
        text_field: str = "text",
    ) -> Dataset:
        """
        Tokenize dataset.

        Args:
            dataset: Dataset to tokenize.
            dataset_name: Name for logging.
            text_field: Field containing text to tokenize.

        Returns:
            Tokenized dataset.
        """
        tok_config = self.config.processing.get("tokenization", {})
        if not tok_config.get("enabled", True):
            return dataset

        if self.tokenizer is None:
            tokenizer = self.load_tokenizer()
            if tokenizer is not None:
                self.tokenizer = tokenizer

        logger.info(f"Tokenizing {dataset_name}...")

        max_seq_length = tok_config.get("max_seq_length", 2048)
        truncation = tok_config.get("truncation", True)
        truncation_side = tok_config.get("truncation_side", "right")
        padding = tok_config.get("padding", False)
        padding_side = tok_config.get("padding_side", "right")
        pad_to_multiple_of = tok_config.get("pad_to_multiple_of", 8)
        add_special_tokens = tok_config.get("add_special_tokens", True)
        num_proc = tok_config.get("num_proc", 4)
        batch_size = tok_config.get("batch_size", 1000)
        compute_stats = tok_config.get("compute_stats", True)

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "truncation_side"):
                self.tokenizer.truncation_side = truncation_side
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = padding_side

        def tokenize_batch(examples):
            return self.tokenizer(
                examples[text_field],
                truncation=truncation,
                max_length=max_seq_length,
                padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                add_special_tokens=add_special_tokens,
                return_attention_mask=True,
                return_token_type_ids=False,
            )

        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=[c for c in dataset.column_names if c != text_field],
            num_proc=num_proc,
        )

        if compute_stats and "input_ids" in tokenized.column_names:
            lengths = [len(ids) for ids in tokenized["input_ids"]]
            self.statistics.token_lengths = lengths

            percentiles = tok_config.get("percentiles", [0, 25, 50, 75, 90, 95, 99, 100])
            for p in percentiles:
                self.statistics.percentiles.setdefault("tokens", {})[p] = float(
                    np.percentile(lengths, p)
                )

        return tokenized

    def analyze_sequences(self, dataset: Dataset) -> dict[str, Any]:
        """
        Analyze token sequence statistics.

        Args:
            dataset: Tokenized dataset.

        Returns:
            Dictionary of sequence statistics.
        """
        if "input_ids" not in dataset.column_names:
            logger.warning("No input_ids column found, skipping sequence analysis")
            return {}

        lengths = [len(ids) for ids in dataset["input_ids"]]
        lengths = np.array(lengths)

        stats = {
            "count": len(lengths),
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "percentiles": {
                str(p): float(np.percentile(lengths, p)) for p in [0, 25, 50, 75, 90, 95, 99, 100]
            },
            "truncated_count": int(np.sum(lengths >= 2048)),
            "truncated_ratio": float(np.mean(lengths >= 2048)),
        }

        logger.info(f"Sequence analysis: {stats}")
        return stats

    def compute_statistics(
        self, dataset: Dataset, dataset_name: str = "dataset"
    ) -> DatasetStatistics:
        """
        Compute comprehensive dataset statistics.

        Args:
            dataset: Dataset to analyze.
            dataset_name: Name for logging.

        Returns:
            DatasetStatistics object.
        """
        stats_config = self.config.processing.get("statistics", {})
        if not stats_config.get("enabled", True):
            return self.statistics

        logger.info(f"Computing statistics for {dataset_name}...")

        self.statistics.num_samples = len(dataset)
        self.statistics.num_columns = len(dataset.column_names)
        self.statistics.column_names = dataset.column_names

        sample_size = stats_config.get("sample_size")
        if sample_size and len(dataset) > sample_size:
            sample = dataset.shuffle(seed=42).select(range(sample_size))
        else:
            sample = dataset

        percentiles = stats_config.get("percentiles", [0, 25, 50, 75, 90, 95, 99, 100])

        for col in ["instruction", "input", "output"]:
            if col in dataset.column_names:
                lengths = [len(str(ex[col])) for ex in sample]
                setattr(self.statistics, f"{col}_lengths", lengths)
                self.statistics.percentiles[col] = {
                    p: float(np.percentile(lengths, p)) for p in percentiles
                }

        if "text" in dataset.column_names:
            total_lengths = [len(str(ex["text"])) for ex in sample]
            self.statistics.total_lengths = total_lengths
            self.statistics.percentiles["total"] = {
                p: float(np.percentile(total_lengths, p)) for p in percentiles
            }

        if "input_ids" in dataset.column_names:
            token_lengths = [len(ex["input_ids"]) for ex in sample]
            self.statistics.token_lengths = token_lengths
            self.statistics.percentiles["tokens"] = {
                p: float(np.percentile(token_lengths, p)) for p in percentiles
            }

        for col in dataset.column_names:
            null_count = sum(1 for ex in sample if ex[col] is None)
            empty_count = sum(
                1 for ex in sample if isinstance(ex[col], str) and not ex[col].strip()
            )
            if null_count > 0:
                self.statistics.null_counts[col] = null_count
            if empty_count > 0:
                self.statistics.empty_string_counts[col] = empty_count

        save_path = stats_config.get("save_path", "./data/processed/statistics.json")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(self.statistics.to_dict(), f, indent=2)
            logger.info(f"Statistics saved to {save_path}")

        return self.statistics

    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """
        Split dataset into train/validation/test.

        Args:
            dataset: Dataset to split.

        Returns:
            DatasetDict with train/validation/test splits.
        """
        split_config = self.config.processing.get("splitting", {})
        if not split_config.get("enabled", True):
            return DatasetDict({"train": dataset})

        logger.info("Splitting dataset...")

        ratios = split_config.get("ratios", {"train": 0.9, "validation": 0.05, "test": 0.05})
        seed = split_config.get("seed", 42)
        shuffle = split_config.get("shuffle", True)
        stratify_by = split_config.get("stratify_by", None)
        min_train = split_config.get("min_train_samples", 1)
        min_val = split_config.get("min_val_samples", 1)
        min_test = split_config.get("min_test_samples", 1)
        method = split_config.get("method", "random")

        total = len(dataset)
        if total == 0:
            return DatasetDict({"train": dataset, "validation": dataset, "test": dataset})

        r_train = ratios.get("train", 0.9)
        r_val = ratios.get("validation", 0.05)
        r_test = ratios.get("test", 0.05)

        if total < (min_train + min_val + min_test):
            train_size = max(1, min(total, int(total * r_train)))
            rem = total - train_size
            val_size = max(0, min(rem, int(total * r_val)))
            test_size = total - train_size - val_size
        else:
            train_size = max(int(total * r_train), min_train)
            val_size = max(int(total * r_val), min_val)
            test_size = max(int(total * r_test), min_test)
            if train_size + val_size + test_size > total:
                scale = total / (train_size + val_size + test_size)
                train_size = int(train_size * scale)
                val_size = int(val_size * scale)
                test_size = total - train_size - val_size

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        if method == "random":
            train_ds = dataset.select(range(train_size))
            val_ds = dataset.select(range(train_size, train_size + val_size))
            test_ds = dataset.select(
                range(train_size + val_size, train_size + val_size + test_size)
            )
        elif method == "sequential":
            train_ds = dataset.select(range(train_size))
            val_ds = dataset.select(range(train_size, train_size + val_size))
            test_ds = dataset.select(
                range(train_size + val_size, train_size + val_size + test_size)
            )
        elif method == "stratified" and stratify_by and stratify_by in dataset.column_names:
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(dataset))
            labels = np.array(dataset[stratify_by])

            train_idx, temp_idx = train_test_split(
                indices, train_size=ratios["train"], random_state=seed, stratify=labels
            )
            val_ratio = ratios["validation"] / (ratios["validation"] + ratios["test"])
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_ratio, random_state=seed, stratify=labels[temp_idx]
            )

            train_ds = dataset.select(train_idx.tolist())
            val_ds = dataset.select(val_idx.tolist())
            test_ds = dataset.select(test_idx.tolist())
        else:
            train_ds = dataset.select(range(train_size))
            val_ds = dataset.select(range(train_size, train_size + val_size))
            test_ds = dataset.select(
                range(train_size + val_size, train_size + val_size + test_size)
            )

        logger.info(
            f"Split sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
        )

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )

    def export_jsonl(self, dataset: Dataset, output_path: str | Path):
        """Export dataset to JSONL format."""
        logger.info(f"Exporting to JSONL: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(str(output_path), lines=True, force_ascii=False)

    def export_parquet(
        self, dataset: Dataset, output_path: str | Path, compression: str = "snappy"
    ):
        """Export dataset to Parquet format."""
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow required for Parquet export")
        logger.info(f"Exporting to Parquet: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(output_path))

    def export_arrow(self, dataset: Dataset, output_path: str | Path):
        """Export dataset to Arrow format."""
        logger.info(f"Exporting to Arrow: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))

    def export_dataset(self, dataset: Dataset, base_path: str | Path, formats: list[str] = None):
        """
        Export dataset to multiple formats.

        Args:
            dataset: Dataset to export.
            base_path: Base output path (without extension).
            formats: List of formats to export. Defaults to config.
        """
        if formats is None:
            formats = self.config.output.formats

        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        compression = getattr(self.config.output, "compression", "snappy")

        for fmt in formats:
            if fmt == "jsonl":
                self.export_jsonl(dataset, base_path.with_suffix(".jsonl"))
            elif fmt == "parquet":
                self.export_parquet(dataset, base_path.with_suffix(".parquet"), compression)
            elif fmt == "arrow":
                self.export_arrow(dataset, base_path)
            else:
                logger.warning(f"Unknown export format: {fmt}")

    def process(self, dataset_name: str | None = None) -> DatasetDict:
        """
        Run the complete data processing pipeline.

        Args:
            dataset_name: Specific dataset to process. If None, processes all.

        Returns:
            DatasetDict with processed splits.
        """
        logger.info("Starting data processing pipeline...")

        datasets = self.download_datasets()
        local_datasets = self.load_local_datasets()
        datasets.update(local_datasets)

        if dataset_name:
            if dataset_name not in datasets:
                raise ValueError(
                    f"Dataset {dataset_name} not found. Available: {list(datasets.keys())}"
                )
            datasets = {dataset_name: datasets[dataset_name]}

        processed_datasets = {}
        for name, dataset in datasets.items():
            logger.info(f"Processing dataset: {name}")

            # Apply column mapping
            for ds_config in self.config.datasets:
                if ds_config.name == name:
                    dataset = self.apply_column_mapping(dataset, ds_config.column_mapping)
                    break

            dataset = self.validate_dataset(dataset, name)
            dataset = self.clean_dataset(dataset, name)
            dataset = self.format_dataset(dataset, name)

            if self.config.processing.get("tokenization", {}).get("enabled", True):
                self.load_tokenizer()
                dataset = self.tokenize_dataset(dataset, name)

            self.compute_statistics(dataset, name)
            self.analyze_sequences(dataset)

            splits = self.split_dataset(dataset)
            processed_datasets[name] = splits

        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, splits in processed_datasets.items():
            for split_name, split_ds in splits.items():
                if self.config.output.save_splits:
                    filenames = self.config.output.filenames
                    filename = filenames.get(split_name, f"{split_name}.arrow")
                    if name != list(processed_datasets.keys())[0]:
                        filename = f"{name}_{filename}"
                    self.export_dataset(split_ds, output_dir / filename)

        if self.config.output.save_statistics:
            stats_path = output_dir / self.config.output.stats_filename
            with open(stats_path, "w") as f:
                json.dump(self.statistics.to_dict(), f, indent=2)

        logger.info("Data processing pipeline completed successfully!")
        return processed_datasets


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML file",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Specific dataset to process (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--formats",
        "-f",
        nargs="+",
        default=None,
        help="Export formats (jsonl, parquet, arrow)",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download, use cached",
    )

    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Skip tokenization step",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path",
    )

    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Prompt template (alpaca, chatml, llama3, vicuna, zephyr, plain, custom)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset",
    )

    parser.add_argument(
        "--split-ratios",
        type=str,
        default=None,
        help="Train/val/test ratios as comma-separated (e.g., 0.8,0.1,0.1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Initializing Data Pipeline...")
    logger.info(f"Config: {args.config}")

    pipeline = DataPipeline(args.config)

    if args.output_dir:
        pipeline.config.output.output_dir = args.output_dir

    if args.formats:
        pipeline.config.output.formats = args.formats

    if args.template:
        pipeline.config.processing.setdefault("formatting", {})["template"] = args.template

    if args.max_samples:
        for ds in pipeline.config.datasets:
            ds.max_samples = args.max_samples

    if args.split_ratios:
        ratios = [float(x) for x in args.split_ratios.split(",")]
        if len(ratios) == 3:
            pipeline.config.processing.setdefault("splitting", {})["ratios"] = {
                "train": ratios[0],
                "validation": ratios[1],
                "test": ratios[2],
            }

    if args.skip_tokenization:
        pipeline.config.processing.setdefault("tokenization", {})["enabled"] = False

    if args.skip_download:
        logger.warning("Skip download not fully implemented, using cached datasets")

    if args.dry_run:
        logger.info("DRY RUN - would process datasets:")
        for ds in pipeline.config.datasets:
            logger.info(f"  - {ds.name}: {ds.path} (split: {ds.split})")
        return

    pipeline.process(dataset_name=args.dataset)


if __name__ == "__main__":
    main()

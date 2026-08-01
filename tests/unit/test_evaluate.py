"""Unit tests for src/evaluate.py."""

from unittest.mock import MagicMock, Mock

import pytest
from datasets import Dataset
from src.evaluate import (
    GenerationConfig,
    GenerationResult,
    ModelEvaluator,
    PromptFormatter,
    create_argument_parser,
)
from src.metrics import MetricsCalculator


@pytest.fixture
def configs():
    return {
        "rouge_types": ["rouge1", "rouge2", "rougeL"],
        "bleu_smoothing": "exp",
        "distinct_n": [1, 2],
    }


class TestPromptFormatter:
    """Test PromptFormatter class."""

    def test_format_alpaca_with_input(self):
        formatter = PromptFormatter(template_name="alpaca")
        prompt = formatter.format(
            instruction="Summarize this text",
            input_text="The quick brown fox jumps over the lazy dog.",
        )
        assert "Summarize this text" in prompt
        assert "The quick brown fox" in prompt

    def test_format_alpaca_no_input(self):
        formatter = PromptFormatter(template_name="alpaca")
        prompt = formatter.format(instruction="Write a story")
        assert "Write a story" in prompt

    def test_custom_template(self):
        formatter = PromptFormatter(custom_template="Input: {instruction}\nOutput: {output}")
        prompt = formatter.format(instruction="Hello")
        assert "Input: Hello" in prompt


class TestMetricsCalculator:
    """Test MetricsCalculator in evaluate."""

    def test_calculate_distinct(self, configs):
        calc = MetricsCalculator(**configs)
        texts = [
            "the cat sat on the mat",
            "the dog ran fast",
            "a bird flew high",
        ]
        result = calc.calculate_distinct_n(texts)
        assert "distinct_1" in result
        assert "distinct_2" in result
        assert 0 <= result["distinct_1"] <= 1
        assert 0 <= result["distinct_2"] <= 1


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.generate = Mock(return_value=MagicMock())
        model.parameters = Mock(return_value=iter([]))
        model.config = Mock()
        model.config.pad_token_id = 0
        model.config.eos_token_id = 1
        model.device = "cpu"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        inputs = MagicMock()
        inputs.input_ids.shape = [1, 3]
        inputs.to.return_value = inputs
        tokenizer.return_value = inputs
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="Generated response")
        tokenizer.apply_chat_template = Mock(return_value="Formatted prompt")
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    def test_generate_response(self, mock_model, mock_tokenizer):
        evaluator = ModelEvaluator(
            mock_model, mock_tokenizer, GenerationConfig(), PromptFormatter()
        )
        gen_output = MagicMock()
        gen_output.__getitem__.return_value = [1, 2, 3, 4, 5, 6, 7]
        mock_model.generate.return_value = [gen_output]
        mock_tokenizer.decode.return_value = "This is a response"

        result = evaluator.generate("Test prompt")
        assert isinstance(result, GenerationResult)

    def test_evaluate_dataset(self, mock_model, mock_tokenizer):
        evaluator = ModelEvaluator(
            mock_model, mock_tokenizer, GenerationConfig(), PromptFormatter()
        )
        gen_output = MagicMock()
        gen_output.__getitem__.return_value = [1, 2, 3, 4, 5]
        mock_model.generate.return_value = [gen_output]
        mock_tokenizer.decode.return_value = "Response"

        dataset = Dataset.from_dict(
            {
                "instruction": ["Q1", "Q2"],
                "input": ["", ""],
                "output": ["A1", "A2"],
            }
        )

        results, preds, refs = evaluator.evaluate_dataset(dataset, max_samples=2)

        assert len(results) == 2
        assert len(preds) == 2
        assert len(refs) == 2

    def test_benchmark_performance(self, mock_model, mock_tokenizer):
        evaluator = ModelEvaluator(
            mock_model, mock_tokenizer, GenerationConfig(), PromptFormatter()
        )
        gen_output = MagicMock()
        gen_output.__getitem__.return_value = [1, 2, 3, 4, 5]
        mock_model.generate.return_value = [gen_output]
        mock_tokenizer.decode.return_value = "Response"

        dataset = Dataset.from_dict(
            {
                "instruction": ["Q"] * 5,
                "input": [""] * 5,
                "output": ["A"] * 5,
            }
        )

        perf = evaluator.benchmark_performance(dataset, num_warmup=1, num_runs=3)
        assert perf is not None


class TestArgumentParser:
    """Test CLI argument parser."""

    def test_default_config(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--base-model", "gpt2"])
        assert args.config == "configs"
        assert args.base_model == "gpt2"

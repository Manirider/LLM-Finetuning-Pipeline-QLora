"""Integration tests for evaluation module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import (
    BertScoreConfig,
    BleuConfig,
    DistinctConfig,
    EvalDatasetConfig,
    PerplexityConfig,
    RougeConfig,
)
from src.config import GenerationConfig as ConfigGenerationConfig
from src.evaluate import (
    EvaluationReport,
    GenerationResult,
    MetricResult,
    MetricsCalculator,
    ModelEvaluator,
    PromptFormatter,
    generate_comparison_table,
    run_evaluation,
    save_reports,
)


class TestPromptFormatterIntegration:
    """Test prompt formatter with various templates."""

    def test_all_templates(self):
        """Test all supported prompt templates."""
        templates = {
            "alpaca": {
                "with_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                "without_input": "### Instruction:\n{instruction}\n\n### Response:\n",
            },
            "chatml": {
                "with_input": "im_start>system\n{system}im_end>\nim_start>user\n{instruction}\n\n{input}im_end>\nim_start>assistant\n",
                "without_input": "im_start>system\n{system}im_end>\nim_start>user\n{instruction}im_end>\nim_start>assistant\n",
            },
            "llama3": {
                "with_input": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "without_input": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            },
            "vicuna": {
                "with_input": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {instruction}\n\n{input}\nASSISTANT: ",
                "without_input": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {instruction}\nASSISTANT: ",
            },
            "zephyr": {
                "with_input": "<|system|>\n{system}<|end|>\n<|user|>\n{instruction}\n\n{input}<|end|>\n<|assistant|>\n",
                "without_input": "<|system|>\n{system}<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
            },
        }

        for template_name, _expected in templates.items():
            formatter = PromptFormatter(
                template_name=template_name, system_message="You are helpful."
            )

            # Test with input
            prompt = formatter.format(instruction="Test", input_text="Input here")
            assert "Test" in prompt
            assert "Input here" in prompt

            # Test without input
            prompt = formatter.format(instruction="Test", input_text="")
            assert "Test" in prompt
            assert "Input here" not in prompt

    def test_custom_template(self):
        """Test custom prompt template."""
        formatter = PromptFormatter(
            template_name="custom",
            template="{instruction} -> {output}",
            template_with_input="{instruction} [{input}] -> {output}",
        )

        prompt = formatter.format(instruction="Task", input_text="Context", output="Result")
        assert "Task" in prompt
        assert "Context" in prompt
        assert "Result" in prompt


class TestMetricsCalculatorIntegration:
    """Test metrics calculator with mocked dependencies."""

    @patch("src.evaluate.rouge_scorer")
    def test_rouge_calculation(self, mock_rouge_scorer):
        """Test ROUGE score calculation."""
        # Mock rouge scorer
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {
            "rouge1": MagicMock(fmeasure=0.8),
            "rouge2": MagicMock(fmeasure=0.6),
            "rougeL": MagicMock(fmeasure=0.7),
            "rougeLsum": MagicMock(fmeasure=0.7),
        }
        mock_rouge_scorer.RougeScorer.return_value = mock_scorer

        rouge_config = RougeConfig(
            enabled=True,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        )

        calculator = MetricsCalculator(rouge_config=rouge_config)

        predictions = ["The cat sat on the mat", "A dog ran fast"]
        references = ["The cat sat on the mat", "A dog ran quickly"]

        result = calculator.calculate_rouge(predictions, references)

        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert "rougeLsum" in result
        assert all(0 <= v <= 1 for v in result.values())

    @patch("src.evaluate.nltk")
    def test_bleu_calculation(self, mock_nltk):
        """Test BLEU score calculation."""
        mock_nltk.translate.bleu_score.corpus_bleu.return_value = 0.45

        bleu_config = BleuConfig(enabled=True, max_order=4)

        calculator = MetricsCalculator(bleu_config=bleu_config)

        predictions = [["the", "cat", "sat"], ["a", "dog", "ran"]]
        references = [[["the", "cat", "sat"]], [["a", "dog", "ran", "fast"]]]

        result = calculator.calculate_bleu(predictions, references)

        assert "bleu" in result
        assert result["bleu"] == 0.45

    @patch("src.evaluate.nltk")
    def test_meteor_calculation(self, mock_nltk):
        """Test METEOR score calculation."""
        mock_nltk.translate.meteor_score.meteor_score.return_value = 0.55

        calculator = MetricsCalculator()

        predictions = ["The cat sat on the mat"]
        references = [["The cat sat on the mat"]]

        result = calculator.calculate_meteor(predictions, references)

        assert "meteor" in result
        assert result["meteor"] == 0.55

    @patch("src.evaluate.bert_score")
    def test_bertscore_calculation(self, mock_bert_score):
        """Test BERTScore calculation."""
        import torch

        mock_bert_score.score.return_value = (
            torch.tensor([0.9]),
            torch.tensor([0.85]),
            torch.tensor([0.87]),
        )

        bertscore_config = BertScoreConfig(
            enabled=True,
            metrics=["precision", "recall", "f1"],
        )

        calculator = MetricsCalculator(bertscore_config=bertscore_config)

        predictions = ["The cat sat"]
        references = [["The cat sat"]]

        result = calculator.calculate_bertscore(predictions, references)

        assert "bertscore_precision" in result
        assert "bertscore_recall" in result
        assert "bertscore_f1" in result

    def test_distinct_calculation(self):
        """Test Distinct-n calculation."""
        distinct_config = DistinctConfig(enabled=True, n_grams=[1, 2, 3, 4])

        calculator = MetricsCalculator(distinct_config=distinct_config)

        texts = [
            "the cat sat on the mat",
            "the dog ran fast",
            "a bird flew high",
        ]

        result = calculator.calculate_distinct_n(texts)

        assert "distinct_1" in result
        assert "distinct_2" in result
        assert "distinct_3" in result
        assert "distinct_4" in result
        assert all(0 <= v <= 1 for v in result.values())

    def test_calculate_all_metrics(self):
        """Test calculating all metrics together."""
        rouge_config = RougeConfig(enabled=True)
        bleu_config = BleuConfig(enabled=True)
        bertscore_config = BertScoreConfig(enabled=False)
        perplexity_config = PerplexityConfig(enabled=False)
        distinct_config = DistinctConfig(enabled=True)

        with (
            patch.object(MetricsCalculator, "calculate_rouge") as mock_rouge,
            patch.object(MetricsCalculator, "calculate_bleu") as mock_bleu,
            patch.object(MetricsCalculator, "calculate_meteor") as mock_meteor,
            patch.object(MetricsCalculator, "calculate_bertscore") as mock_bertscore,
            patch.object(MetricsCalculator, "calculate_distinct_n") as mock_distinct,
        ):
            mock_rouge.return_value = {"rouge1": 0.8, "rouge2": 0.6}
            mock_bleu.return_value = {"bleu": 0.45}
            mock_meteor.return_value = {"meteor": 0.55}
            mock_bertscore.return_value = {}
            mock_distinct.return_value = {"distinct_1": 0.9, "distinct_2": 0.8}

            calculator = MetricsCalculator(
                rouge_config=rouge_config,
                bleu_config=bleu_config,
                bertscore_config=bertscore_config,
                perplexity_config=perplexity_config,
                distinct_config=distinct_config,
            )

            predictions = ["pred1", "pred2"]
            references = [["ref1"], ["ref2"]]
            gen_texts = ["gen1", "gen2"]

            metrics = calculator.calculate_all(predictions, references, gen_texts)

            assert "rouge1" in metrics
            assert "rouge2" in metrics
            assert "bleu" in metrics
            assert "meteor" in metrics
            assert "distinct_1" in metrics
            assert "distinct_2" in metrics


class TestModelEvaluatorIntegration:
    """Test ModelEvaluator with mocked model."""

    @patch("src.evaluate.AutoModelForCausalLM")
    @patch("src.evaluate.AutoTokenizer")
    def test_model_evaluator_generate(self, mock_tokenizer_class, mock_model_class):
        """Test model evaluator generation."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer.apply_chat_template = Mock(return_value="Formatted prompt")
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7]]
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_model.config.eos_token_id = 1
        mock_model.device = "cuda:0"
        mock_model_class.from_pretrained.return_value = mock_model

        # Create evaluator
        generation_config = ConfigGenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        evaluator = ModelEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=generation_config,
            prompt_formatter=PromptFormatter(template_name="alpaca"),
        )

        # Test generation
        result = evaluator.generate_response("Test prompt")

        assert isinstance(result, GenerationResult)
        assert result.response == "Generated response"
        assert result.latency_ms >= 0

    def test_model_evaluator_evaluate_dataset(self):
        """Test model evaluator on dataset."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_model.config.eos_token_id = 1
        mock_model.device = "cuda:0"

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.apply_chat_template = Mock(return_value="Prompt")
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        generation_config = ConfigGenerationConfig(max_new_tokens=256)

        evaluator = ModelEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=generation_config,
            prompt_formatter=PromptFormatter(template_name="alpaca"),
        )

        dataset = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "", "output": "A2"},
        ]

        results, preds, refs = evaluator.evaluate_dataset(dataset, max_samples=2)

        assert len(results) == 2
        assert len(preds) == 2
        assert len(refs) == 2
        assert preds[0] == "Response"
        assert refs[0] == "A1"

    def test_model_evaluator_benchmark(self):
        """Test model evaluator benchmarking."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.config = MagicMock()
        mock_model.config.pad_token_id = 0
        mock_model.config.eos_token_id = 1
        mock_model.device = "cuda:0"

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.apply_chat_template = Mock(return_value="Prompt")
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        generation_config = ConfigGenerationConfig(max_new_tokens=256)

        evaluator = ModelEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=generation_config,
            prompt_formatter=PromptFormatter(template_name="alpaca"),
        )

        dataset = [{"instruction": "Q", "input": "", "output": "A"}] * 5

        perf = evaluator.benchmark_performance(dataset, num_warmup=1, num_runs=3)

        assert "avg_latency_ms" in perf
        assert "avg_throughput_tokens_per_sec" in perf
        assert "avg_memory_mb" in perf
        assert "peak_memory_mb" in perf


class TestEvaluationReporting:
    """Test evaluation reporting functions."""

    def test_evaluation_report_to_dict(self):
        """Test EvaluationReport serialization."""
        report = EvaluationReport(
            model_name="test_model",
            dataset_name="test_dataset",
            generation_config={"max_new_tokens": 256},
            samples_evaluated=10,
            metrics={
                "rouge1": MetricResult("rouge1", 0.8),
                "bleu": MetricResult("bleu", 0.4),
            },
            performance={"avg_latency_ms": 100, "avg_memory_mb": 1000},
            generation_results=[],
            timestamp="2024-01-01 00:00:00",
        )

        d = report.to_dict()

        assert d["model_name"] == "test_model"
        assert d["dataset_name"] == "test_dataset"
        assert d["samples_evaluated"] == 10
        assert d["metrics"]["rouge1"]["value"] == 0.8
        assert d["performance"]["avg_latency_ms"] == 100

    def test_generate_comparison_table(self):
        """Test comparison table generation."""
        report1 = EvaluationReport(
            model_name="base",
            dataset_name="test",
            generation_config={},
            samples_evaluated=10,
            metrics={
                "rouge1": MetricResult("rouge1", 0.7),
                "bleu": MetricResult("bleu", 0.3),
            },
            performance={"avg_latency_ms": 200, "avg_memory_mb": 1000},
            generation_results=[],
            timestamp="",
        )
        report2 = EvaluationReport(
            model_name="finetuned",
            dataset_name="test",
            generation_config={},
            samples_evaluated=10,
            metrics={
                "rouge1": MetricResult("rouge1", 0.8),
                "bleu": MetricResult("bleu", 0.4),
            },
            performance={"avg_latency_ms": 250, "avg_memory_mb": 1200},
            generation_results=[],
            timestamp="",
        )

        table = generate_comparison_table({"base_test": report1, "finetuned_test": report2})

        assert "base" in table
        assert "finetuned" in table
        assert "rouge1" in table
        assert "bleu" in table
        assert "0.7000" in table
        assert "0.8000" in table

    def test_save_reports(self, temp_dir):
        """Test saving evaluation reports."""
        report = EvaluationReport(
            model_name="test",
            dataset_name="test",
            generation_config={},
            samples_evaluated=1,
            metrics={"rouge1": MetricResult("rouge1", 0.85)},
            performance={"avg_latency_ms": 100},
            generation_results=[],
            timestamp="2024-01-01 00:00:00",
        )

        save_reports({"test_test": report}, str(temp_dir), formats=["json"])

        json_path = temp_dir / "test_test.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["model_name"] == "test"
        assert data["metrics"]["rouge1"]["value"] == 0.85


class TestRunEvaluationIntegration:
    """Integration tests for run_evaluation function."""

    @patch("src.evaluate.load_dataset")
    @patch("src.evaluate.AutoModelForCausalLM.from_pretrained")
    @patch("src.evaluate.AutoTokenizer.from_pretrained")
    @patch("src.evaluate.PeftModel.from_pretrained")
    def test_run_evaluation_pipeline(
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
        from src.config import EvaluationConfigComplete, GenerationConfig

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
                "meteor": type("obj", (object,), {"enabled": True})(),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

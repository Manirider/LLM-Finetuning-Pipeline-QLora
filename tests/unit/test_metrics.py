"""Unit tests for src/metrics.py."""

from src.metrics import (
    MetricResult,
    MetricsCalculator,
    compute_bleu,
    compute_rouge,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_initialization(self):
        res = MetricResult(name="rouge1", value=0.85)
        assert res.name == "rouge1"
        assert res.value == 0.85
        assert isinstance(res.details, dict)


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_rouge(self):
        calculator = MetricsCalculator()
        predictions = ["The quick brown fox jumps over the lazy dog."]
        references = ["The quick brown fox jumped over the lazy dog."]

        results = calculator.calculate_rouge(predictions, references)
        assert "rouge1" in results
        assert "rouge2" in results
        assert "rougeL" in results
        assert isinstance(results["rouge1"].value, float)

    def test_calculate_distinct_n(self):
        calculator = MetricsCalculator(distinct_n=[1, 2])
        predictions = [
            "the cat sat on the mat",
            "the dog sat on the rug",
        ]

        results = calculator.calculate_distinct_n(predictions)
        assert "distinct_1" in results
        assert "distinct_2" in results
        assert 0.0 <= results["distinct_1"].value <= 1.0

    def test_calculate_bleu(self):
        calculator = MetricsCalculator()
        predictions = ["The quick brown fox jumps over the lazy dog."]
        references = [["The quick brown fox jumps over the lazy dog."]]

        results = calculator.calculate_bleu(predictions, references)
        assert "bleu" in results
        assert isinstance(results["bleu"].value, float)

    def test_compute_convenience_functions(self):
        preds = ["hello world"]
        refs = ["hello world"]

        rouge_res = compute_rouge(preds, refs)
        assert isinstance(rouge_res, dict)
        assert "rouge1" in rouge_res

        bleu_res = compute_bleu(preds, [[refs[0]]])
        assert isinstance(bleu_res, float)

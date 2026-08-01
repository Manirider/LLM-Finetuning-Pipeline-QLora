"""
Evaluation Metrics

Comprehensive metrics for LLM evaluation including:
- ROUGE (1, 2, L, Lsum)
- BLEU
- METEOR
- BERTScore
- Perplexity
- Distinct-n
- Custom metrics
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __float__(self) -> float:
        return float(self.value)

    def __le__(self, other: Any) -> bool:
        val = other.value if isinstance(other, MetricResult) else other
        return self.value <= val

    def __ge__(self, other: Any) -> bool:
        val = other.value if isinstance(other, MetricResult) else other
        return self.value >= val

    def __lt__(self, other: Any) -> bool:
        val = other.value if isinstance(other, MetricResult) else other
        return self.value < val

    def __gt__(self, other: Any) -> bool:
        val = other.value if isinstance(other, MetricResult) else other
        return self.value > val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MetricResult):
            return self.name == other.name and self.value == other.value
        return self.value == other


class MetricsCalculator:
    """Central calculator for all evaluation metrics."""

    def __init__(
        self,
        rouge_types: List[str] = None,
        bleu_smoothing: str = "exp",
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        bertscore_layers: int = 17,
        perplexity_model: str = "gpt2-large",
        distinct_n: List[int] = None,
    ):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.bleu_smoothing = bleu_smoothing
        self.bertscore_model = bertscore_model
        self.bertscore_layers = bertscore_layers
        self.perplexity_model = perplexity_model
        self.distinct_n = distinct_n or [1, 2, 3, 4]
        
        # Initialize scorers
        self._rouge_scorer = None
        self._smoothing_fn = None
        self._init_scorers()

    def _init_scorers(self):
        """Initialize metric scorers."""
        if ROUGE_AVAILABLE:
            self._rouge_scorer = rouge_scorer.RougeScorer(
                self.rouge_types,
                use_stemmer=True,
            )
        
        if NLTK_AVAILABLE:
            sf = SmoothingFunction()
            if self.bleu_smoothing == "exp":
                self._smoothing_fn = sf.method1
            elif self.bleu_smoothing == "floor":
                self._smoothing_fn = sf.method2
            elif self.bleu_smoothing == "add-k":
                self._smoothing_fn = sf.method3
            else:
                self._smoothing_fn = sf.method1

    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, MetricResult]:
        """Calculate ROUGE scores."""
        if not ROUGE_AVAILABLE or self._rouge_scorer is None:
            return {rt: MetricResult(rt, 0.0) for rt in self.rouge_types}

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        scores = {rt: [] for rt in self.rouge_types}

        for pred, ref in zip(predictions, references):
            result = self._rouge_scorer.score(ref, pred)
            for rt in self.rouge_types:
                scores[rt].append(result[rt].fmeasure)

        results = {}
        for rt in self.rouge_types:
            values = np.array(scores[rt])
            results[rt] = MetricResult(
                name=rt,
                value=float(np.mean(values)),
                details={
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                },
            )

        return results

    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        max_order: int = 4,
    ) -> Dict[str, MetricResult]:
        """Calculate BLEU scores."""
        if not NLTK_AVAILABLE:
            return {"bleu": MetricResult("bleu", 0.0)}

        # Tokenize
        tokenized_preds = [p.split() for p in predictions]
        tokenized_refs = [[r.split() for r in refs] for refs in references]

        # Calculate BLEU
        weights = tuple(1.0 / max_order for _ in range(max_order))
        bleu_score = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=weights,
            smoothing_function=self._smoothing_fn,
        )

        # Also calculate per-sentence BLEU
        sentence_bleus = []
        for pred, refs in zip(tokenized_preds, tokenized_refs):
            sb = sentence_bleu(
                refs,
                pred,
                weights=weights,
                smoothing_function=self._smoothing_fn,
            )
            sentence_bleus.append(sb)

        return {
            "bleu": MetricResult(
                name="bleu",
                value=float(bleu_score),
                details={
                    "sentence_bleu_mean": float(np.mean(sentence_bleus)),
                    "sentence_bleu_std": float(np.std(sentence_bleus)),
                    "max_order": max_order,
                },
            ),
        }

    def calculate_meteor(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, MetricResult]:
        """Calculate METEOR score."""
        if not NLTK_AVAILABLE:
            return {"meteor": MetricResult("meteor", 0.0)}

        scores = []
        for pred, refs in zip(predictions, references):
            try:
                # METEOR expects tokenized input
                pred_tokens = pred.split()
                ref_tokens = [r.split() for r in refs]
                score = meteor_score(ref_tokens, pred_tokens)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return {
            "meteor": MetricResult(
                name="meteor",
                value=float(np.mean(scores)),
                details={
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "count": len(scores),
                },
            ),
        }

    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en",
        batch_size: int = 32,
    ) -> Dict[str, MetricResult]:
        """Calculate BERTScore."""
        if not BERTSCORE_AVAILABLE:
            return {
                "bertscore_precision": MetricResult("bertscore_precision", 0.0),
                "bertscore_recall": MetricResult("bertscore_recall", 0.0),
                "bertscore_f1": MetricResult("bertscore_f1", 0.0),
            }

        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=self.bertscore_model,
            num_layers=self.bertscore_layers,
            batch_size=batch_size,
            lang=lang,
            verbose=False,
        )

        return {
            "bertscore_precision": MetricResult(
                name="bertscore_precision",
                value=float(P.mean()),
                details={
                    "std": float(P.std()),
                    "min": float(P.min()),
                    "max": float(P.max()),
                },
            ),
            "bertscore_recall": MetricResult(
                name="bertscore_recall",
                value=float(R.mean()),
                details={
                    "std": float(R.std()),
                    "min": float(R.min()),
                    "max": float(R.max()),
                },
            ),
            "bertscore_f1": MetricResult(
                name="bertscore_f1",
                value=float(F1.mean()),
                details={
                    "std": float(F1.std()),
                    "min": float(F1.min()),
                    "max": float(F1.max()),
                },
            ),
        }

    def calculate_perplexity(
        self,
        texts: List[str],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        stride: int = 512,
        max_length: int = 1024,
        batch_size: int = 8,
    ) -> Dict[str, MetricResult]:
        """Calculate perplexity using a language model."""
        if not TORCH_AVAILABLE:
            return {"perplexity": MetricResult("perplexity", 0.0)}

        # Use provided model or load default
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.perplexity_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.perplexity_model)

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(model.device)
            
            # Sliding window
            for i in range(0, input_ids.size(1), stride):
                end = min(i + stride, input_ids.size(1))
                chunk = input_ids[:, i:end]
                
                if chunk.size(1) < 2:
                    continue
                    
                with torch.no_grad():
                    outputs = model(chunk, labels=chunk)
                    loss = outputs.loss
                    total_loss += loss.item() * chunk.size(1)
                    total_tokens += chunk.size(1)

        perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

        return {
            "perplexity": MetricResult(
                name="perplexity",
                value=float(perplexity),
                details={
                    "total_loss": total_loss,
                    "total_tokens": total_tokens,
                    "model": self.perplexity_model,
                },
            ),
        }

    def calculate_distinct_n(
        self,
        texts: List[str],
    ) -> Dict[str, MetricResult]:
        """Calculate Distinct-n scores."""
        results = {}
        
        for n in self.distinct_n:
            ngrams = set()
            total = 0
            
            for text in texts:
                tokens = text.split()
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    ngrams.add(ngram)
                    total += 1
            
            distinct = len(ngrams) / total if total > 0 else 0.0
            results[f"distinct_{n}"] = MetricResult(
                name=f"distinct_{n}",
                value=distinct,
                details={
                    "unique_ngrams": len(ngrams),
                    "total_ngrams": total,
                },
            )
        
        return results

    def calculate_all(
        self,
        predictions: List[str],
        references: List[str],
        references_list: Optional[List[List[str]]] = None,
        texts_for_perplexity: Optional[List[str]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, MetricResult]:
        """Calculate all available metrics."""
        results = {}
        
        # ROUGE
        results.update(self.calculate_rouge(predictions, references))
        
        # BLEU
        if references_list is not None:
            results.update(self.calculate_bleu(predictions, references_list))
        else:
            # Use single references
            refs_list = [[r] for r in references]
            results.update(self.calculate_bleu(predictions, refs_list))
        
        # METEOR
        if references_list is not None:
            results.update(self.calculate_meteor(predictions, references_list))
        
        # BERTScore
        results.update(self.calculate_bertscore(predictions, references))
        
        # Distinct-n
        results.update(self.calculate_distinct_n(predictions))
        
        # Perplexity (if model provided)
        if model is not None and texts_for_perplexity is not None:
            results.update(self.calculate_perplexity(
                texts_for_perplexity, model, tokenizer
            ))
        
        return results


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = None,
) -> Dict[str, float]:
    """Convenience function to compute ROUGE scores."""
    calculator = MetricsCalculator(rouge_types=rouge_types)
    results = calculator.calculate_rouge(predictions, references)
    return {k: v.value for k, v in results.items()}


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_order: int = 4,
) -> float:
    """Convenience function to compute BLEU score."""
    calculator = MetricsCalculator()
    results = calculator.calculate_bleu(predictions, references, max_order)
    return results["bleu"].value


def compute_meteor(
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """Convenience function to compute METEOR score."""
    calculator = MetricsCalculator()
    results = calculator.calculate_meteor(predictions, references)
    return results["meteor"].value


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Tuple[float, float, float]:
    """Convenience function to compute BERTScore (P, R, F1)."""
    calculator = MetricsCalculator()
    results = calculator.calculate_bertscore(predictions, references)
    return (
        results["bertscore_precision"].value,
        results["bertscore_recall"].value,
        results["bertscore_f1"].value,
    )


def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2-large",
) -> float:
    """Convenience function to compute perplexity."""
    calculator = MetricsCalculator(perplexity_model=model_name)
    results = calculator.calculate_perplexity(texts)
    return results["perplexity"].value


def compute_distinct_n(texts: List[str], n: int = 4) -> Dict[str, float]:
    """Convenience function to compute Distinct-n scores."""
    calculator = MetricsCalculator(distinct_n=list(range(1, n+1)))
    results = calculator.calculate_distinct_n(texts)
    return {k: v.value for k, v in results.items()}


class MetricTracker:
    """Track and aggregate metrics during training/evaluation."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        self.best_steps: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

            # Track best
            if name not in self.best_metrics:
                self.best_metrics[name] = value
                self.best_steps[name] = step
            elif value < self.best_metrics[name]:  # Assuming lower is better
                self.best_metrics[name] = value
                self.best_steps[name] = step

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        for name, values in self.metrics.items():
            values_array = np.array(values)
            summary[name] = {
                "current": values[-1] if values else None,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "best": self.best_metrics.get(name),
                "best_step": self.best_steps.get(name),
                "count": len(values),
            }
        return summary

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute diversity metrics (Distinct-n, Self-BLEU, etc.)."""
    results = {}
    
    # Distinct-n
    distinct = compute_distinct_n(texts)
    results.update(distinct)
    
    # Self-BLEU (BLEU of each text against others)
    if len(texts) > 1:
        self_bleu = 0.0
        for i, text in enumerate(texts):
            others = texts[:i] + texts[i+1:]
            bleu = compute_bleu(text, [o.split() for o in others])
            self_bleu += bleu
        results["self_bleu"] = self_bleu / len(texts)
    
    return results


def compute_length_statistics(texts: List[str]) -> Dict[str, float]:
    """Compute length statistics for texts."""
    lengths = [len(t.split()) for t in texts]
    return {
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "median_length": np.median(lengths),
    }


def compute_token_statistics(texts: List[str], tokenizer) -> Dict[str, float]:
    """Compute token-level statistics using tokenizer."""
    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))
    
    return {
        "mean_tokens": np.mean(token_counts),
        "std_tokens": np.std(token_counts),
        "min_tokens": np.min(token_counts),
        "max_tokens": np.max(token_counts),
        "median_tokens": np.median(token_counts),
    }


def aggregate_metrics(results_list: List[Dict[str, MetricResult]]) -> Dict[str, MetricResult]:
    """Aggregate metrics from multiple runs."""
    aggregated = {}
    for results in results_list:
        for name, result in results.items():
            if name not in aggregated:
                aggregated[name] = []
            aggregated[name].append(result.value)

    final = {}
    for name, values in aggregated.items():
        values_array = np.array(values)
        final[name] = MetricResult(
            name=name,
            value=float(np.mean(values)),
            details={
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "runs": len(values),
            },
        )
    return final


def print_metrics_report(results: Dict[str, MetricResult]) -> str:
    """Format metrics results as a readable report."""
    lines = ["=" * 50, "METRICS REPORT", "=" * 50]
    for name, result in sorted(results.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  Value: {result.value:.4f}")
        if result.details:
            for k, v in result.details.items():
                lines.append(f"  {k}: {v}")
    lines.append("=" * 50)
    return "\n".join(lines)


__all__ = [
    "MetricResult",
    "MetricsCalculator",
    "MetricTracker",
    "compute_rouge",
    "compute_bleu",
    "compute_meteor",
    "compute_bertscore",
    "compute_perplexity",
    "compute_distinct_n",
    "compute_diversity_metrics",
    "compute_length_statistics",
    "compute_token_statistics",
    "aggregate_metrics",
    "print_metrics_report",
]
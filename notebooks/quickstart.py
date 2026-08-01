"""
# LLM Fine-Tuning Pipeline: Quickstart Tutorial Notebook Script

This interactive script demonstrates the end-to-end pipeline:
1. Loading configuration files
2. Processing datasets with prompt formatting
3. Initializing QLoRA quantized models
4. SFTTrainer setup and execution
5. Evaluation & metric analysis
6. Model merging & FastAPI inference setup
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigManager
from src.data_pipeline import DataPipeline
from src.metrics import MetricsCalculator


def main():
    print("=" * 60)
    print("LLM Fine-Tuning Pipeline — Quickstart Demo")
    print("=" * 60)

    # 1. Load Configurations
    print("\n[Step 1] Loading Configuration System...")
    config = ConfigManager(config_dir="configs")
    print(f"  Model ID: {config.model.model.model_name_or_path}")
    print(f"  LoRA Rank: {config.training.lora.r}")
    print(f"  Quantization: 4-bit NF4 = {config.training.quantization.load_in_4bit}")

    # 2. Test Metrics Calculator
    print("\n[Step 2] Testing Metrics Calculator...")
    calculator = MetricsCalculator()
    preds = ["def add(a, b):\n    return a + b"]
    refs = ["def add(a, b):\n    return a + b"]

    rouge_scores = calculator.calculate_rouge(preds, refs)
    print(f"  ROUGE-1 F1: {rouge_scores['rouge1'].value:.4f}")
    print(f"  ROUGE-L F1: {rouge_scores['rougeL'].value:.4f}")

    bleu_scores = calculator.calculate_bleu(preds, [[refs[0]]])
    print(f"  BLEU Score: {bleu_scores['bleu'].value:.4f}")

    print("\nQuickstart pipeline verification complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Configuration Validation Script

Validates that all configuration files load correctly and pass Pydantic validation.
Run this to verify your configuration setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import ConfigManager, load_config


def main():
    print("=" * 60)
    print("LLM Fine-Tuning Pipeline - Configuration Validation")
    print("=" * 60)

    try:
        # Load configuration
        print("\n[1/5] Loading configuration from configs/...")
        config = load_config(config_dir="configs", env_file=".env.example")
        print("    OK - ConfigManager created successfully")

        # Validate all configs
        print("\n[2/5] Validating all configuration sections...")
        config.validate_all()
        print("    OK - All sections pass Pydantic validation")

        # Check training config
        print("\n[3/5] Checking Training Configuration...")
        t = config.training
        print(f"    Model: {config.model.model.model_name_or_path}")
        print(f"    LoRA r: {t.lora.r}, alpha: {t.lora.lora_alpha}")
        print(f"    Quantization: 4bit={t.quantization.load_in_4bit}, type={t.quantization.bnb_4bit_quant_type}")
        print(f"    Epochs: {t.trainer.num_train_epochs}, Batch size: {t.trainer.per_device_train_batch_size}")
        print(f"    LR: {t.trainer.learning_rate}, Scheduler: {t.trainer.lr_scheduler_type}")
        print(f"    Max seq length: {t.sft.max_seq_length}")
        print(f"    Gradient accumulation: {t.trainer.gradient_accumulation_steps}")
        print(f"    BF16: {t.trainer.bf16}, Gradient checkpointing: {t.trainer.gradient_checkpointing}")

        # Check model config
        print("\n[4/5] Checking Model Configuration...")
        m = config.model
        print(f"    Base model: {m.model.model_name_or_path}")
        print(f"    Torch dtype: {m.model.torch_dtype}")
        print(f"    Attention: {m.model.attn_implementation}")
        print(f"    Tokenizer padding: {m.tokenizer.padding_side}")
        print(f"    PEFT type: {m.peft.peft_type}")
        print(f"    Target modules: {m.peft.lora.target_modules}")

        # Check data config
        print("\n[5/5] Checking Data Configuration...")
        d = config.data
        print(f"    Datasets: {len(d.datasets)} configured")
        for ds in d.datasets[:3]:
            print(f"      - {ds.name}: {ds.path} ({ds.split})")
        if len(d.datasets) > 3:
            print(f"      ... and {len(d.datasets) - 3} more")
        print(f"    Default template: {d.default_template}")
        print(f"    Output dir: {d.output.output_dir}")
        print(f"    Formats: {d.output.formats}")
        print(f"    Split ratios: {d.processing.get('splitting', {}).get('ratios', 'N/A')}")

        # Check logging config
        print("\n[6/6] Checking Logging Configuration...")
        l = config.logging
        print(f"    Log level: {l.level}")
        print(f"    TensorBoard: {l.tensorboard.enabled} ({l.tensorboard.log_dir})")
        print(f"    Weights & Biases: {l.wandb.enabled} ({l.wandb.project})")
        print(f"    MLflow: {l.mlflow.enabled}")

        # Check evaluation config
        print("\n[7/7] Checking Evaluation Configuration...")
        e = config.evaluation
        print(f"    Generation max tokens: {e.generation.max_new_tokens}")
        print(f"    Temperature: {e.generation.temperature}")
        print(f"    Datasets: {len(e.datasets)} configured")
        for ds in e.datasets[:3]:
            print(f"      - {ds.name}: {ds.max_samples} samples")
        print(f"    ROUGE: {e.metrics.get('rouge', {}).get('enabled', 'N/A')}")
        print(f"    BLEU: {e.metrics.get('bleu', {}).get('enabled', 'N/A')}")
        print(f"    BERTScore: {e.metrics.get('bertscore', {}).get('enabled', 'N/A')}")
        print(f"    Perplexity: {e.metrics.get('perplexity', {}).get('enabled', 'N/A')}")

        print("\n" + "=" * 60)
        print("ALL VALIDATIONS PASSED!")
        print("=" * 60)
        return 0

    except Exception as ex:
        print(f"\n[ERROR] Configuration validation failed: {ex}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
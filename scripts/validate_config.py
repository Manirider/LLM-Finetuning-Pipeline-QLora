#!/usr/bin/env python
"""
Configuration Validation Script

Validates that all configuration files load correctly and pass Pydantic validation.
Run this to verify your configuration setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
        print("\n[2/5] Validating configuration models...")
        config.validate_all()
        print("    OK - All configurations passed Pydantic validation")

        # Test training config access
        print("\n[3/5] Testing training configuration access...")
        training = config.training
        print(f"    Model: {training.trainer.run_name}")
        print(f"    Epochs: {training.trainer.num_train_epochs}")
        print(f"    Batch size: {training.trainer.per_device_train_batch_size}")
        print(f"    Learning rate: {training.trainer.learning_rate}")
        print(f"    LoRA rank: {training.lora.r}")
        print(f"    LoRA alpha: {training.lora.lora_alpha}")
        print(f"    Quantization: {training.quantization.bnb_4bit_quant_type}")
        print(f"    Compute dtype: {training.quantization.bnb_4bit_compute_dtype}")
        print("    OK - Training config accessible")

        # Test model config access
        print("\n[4/5] Testing model configuration access...")
        model = config.model
        print(f"    Base model: {model.model.model_name_or_path}")
        print(f"    Torch dtype: {model.model.torch_dtype}")
        print(f"    Attention: {model.model.attn_implementation}")
        print(f"    Tokenizer padding: {model.tokenizer.padding_side}")
        print(f"    PEFT type: {model.peft.peft_type}")
        print(f"    LoRA target modules: {model.peft.lora.target_modules}")
        print("    OK - Model config accessible")

        # Test data config access
        print("\n[5/5] Testing data configuration access...")
        data = config.data
        print(f"    Datasets configured: {len(data.datasets)}")
        for ds in data.datasets:
            print(f"      - {ds.name}: {ds.path} ({ds.split})")
        print(f"    Default template: {data.default_template}")
        print(f"    Output dir: {data.output.output_dir}")
        print(f"    Split ratios: {data.splitting.ratios if hasattr(data, 'splitting') else 'N/A'}")
        print("    OK - Data config accessible")

        # Test logging config access
        print("\n[+] Testing logging configuration access...")
        logging = config.logging
        print(f"    Log level: {logging.level}")
        print(f"    TensorBoard enabled: {logging.tensorboard.enabled}")
        print(f"    W&B enabled: {logging.wandb.enabled}")
        print(f"    W&B project: {logging.wandb.project}")
        print("    OK - Logging config accessible")

        # Test evaluation config access
        print("\n[+] Testing evaluation configuration access...")
        eval_config = config.evaluation
        print(f"    Generation max_new_tokens: {eval_config.generation.max_new_tokens}")
        print(f"    Temperature: {eval_config.generation.temperature}")
        print(f"    Evaluation datasets: {len(eval_config.datasets)}")
        for ds in eval_config.datasets:
            print(f"      - {ds.name}: {ds.max_samples} samples")
        print("    OK - Evaluation config accessible")

        # Test environment variable resolution
        print("\n[+] Testing environment variable resolution...")
        wandb_key = config.logging.wandb.api_key
        print(f"    W&B API key resolved: {'***' if wandb_key != '${WANDB_API_KEY}' else 'NOT SET (using placeholder)'}")
        openai_key = config.evaluation.llm_judge.api_key if hasattr(config.evaluation, 'llm_judge') else 'N/A'
        print(f"    OpenAI API key resolved: {'***' if openai_key != '${OPENAI_API_KEY}' else 'NOT SET (using placeholder)'}")
        print("    OK - Environment variable resolution working")

        # Test config save
        print("\n[+] Testing configuration export...")
        config.save_resolved("configs/resolved_config.yaml")
        print("    OK - Resolved configuration saved to configs/resolved_config.yaml")

        print("\n" + "=" * 60)
        print("ALL VALIDATIONS PASSED!")
        print("=" * 60)
        print("\nConfiguration system is ready for use.")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Merge LoRA adapter into base model.

Usage:
    python -m scripts.merge_adapter \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter_path ./adapters/best \
        --output_path ./artifacts/models/merged/v1.0.0 \
        --dtype bfloat16
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model_utils import merge_and_unload_peft, save_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="",
        help="Hub model ID",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hub token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum shard size",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    logger.info("Merging adapter...")
    merged_model = merge_and_unload_peft(model)

    logger.info(f"Converting to {args.dtype}")
    merged_model = merged_model.to(torch_dtype)

    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Saving merged model to: {args.output_path}")
    save_model_and_tokenizer(
        model=merged_model,
        tokenizer=tokenizer,
        output_dir=args.output_path,
        save_adapter=False,
        save_tokenizer=True,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        hub_private_repo=False,
        commit_message="Upload merged model",
    )

    logger.info("Merge completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Unit tests for src/config.py."""

import os
import sys
from pathlib import Path

import pytest

from src.config import ConfigManager


def test_env_resolution():
    """Test that environment variables are resolved."""
    os.environ["TEST_HF_TOKEN"] = "hf_test_token_123"
    os.environ["TEST_WANDB_KEY"] = "wandb_test_key_456"

    config = ConfigManager(config_dir="configs", env_file=".env.example")
    assert config is not None


def test_config_override():
    """Test programmatic config override."""
    config = ConfigManager(config_dir="configs")
    config.update(
        training={
            "trainer": {"learning_rate": 1e-4, "num_train_epochs": 5},
        }
    )
    assert config.training.trainer.learning_rate == 1e-4
    assert config.training.trainer.num_train_epochs == 5


def test_config_access():
    """Test dot-notation config access."""
    config = ConfigManager(config_dir="configs")
    lr = config.get("training", "trainer.learning_rate")
    assert isinstance(lr, float)

    missing = config.get("training", "nonexistent.key", "default_value")
    assert missing == "default_value"


def test_resolved_config_save(tmp_path):
    """Test saving resolved configuration."""
    config = ConfigManager(config_dir="configs")
    output_path = tmp_path / "resolved_test.yaml"
    config.save_resolved(output_path)
    assert output_path.exists()
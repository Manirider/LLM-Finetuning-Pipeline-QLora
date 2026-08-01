"""Unit tests for src/utils.py."""

import random
from unittest.mock import MagicMock

import numpy as np
import torch
from src.utils import get_device, get_device_map, set_seed


class TestUtils:
    """Tests for utility functions."""

    def test_set_seed(self):
        set_seed(1234, deterministic=False)
        val1 = random.randint(0, 100000)
        np_val1 = np.random.randint(0, 100000)
        torch_val1 = torch.randint(0, 100000, (1,)).item()

        set_seed(1234, deterministic=False)
        val2 = random.randint(0, 100000)
        np_val2 = np.random.randint(0, 100000)
        torch_val2 = torch.randint(0, 100000, (1,)).item()

        assert val1 == val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_get_device(self):
        dev = get_device("cpu")
        assert dev.type == "cpu"

        dev_auto = get_device("auto")
        assert dev_auto.type in ["cuda", "cpu", "mps"]

    def test_get_device_map(self):
        model = MagicMock()
        device_map = get_device_map(model, device_map="auto")
        assert isinstance(device_map, (dict, str))

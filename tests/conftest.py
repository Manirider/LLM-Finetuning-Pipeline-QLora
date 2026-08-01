"""Shared test fixtures and configuration.

This conftest sets up CPU-only testing environment without heavy mocking.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

# Force CPU-only mode for testing BEFORE any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="Generated response")
    tokenizer.apply_chat_template = Mock(return_value="Formatted prompt")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.model_max_length = 4096
    tokenizer.__len__ = Mock(return_value=32000)
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.pad_token_id = 0
    model.config.eos_token_id = 1
    model.config.hidden_size = 4096
    model.config.num_hidden_layers = 32
    model.config.vocab_size = 32000
    model.device = "cpu"
    model.generate = Mock(return_value=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=1000000), requires_grad=True)])
    model.named_parameters = Mock(return_value=[("layer.weight", Mock(numel=Mock(return_value=1000000), requires_grad=True))])
    model.eval = Mock()
    model.train = Mock()
    return model


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    from datasets import Dataset

    data = {
        "instruction": [
            "Write a summary of the text.",
            "Explain quantum computing.",
            "Write a Python function.",
            "Translate to French.",
            "Answer the question.",
        ],
        "input": [
            "The quick brown fox jumps over the lazy dog.",
            "",
            "Calculate fibonacci numbers.",
            "Hello world",
            "What is 2+2?",
        ],
        "output": [
            "A fox jumps over a dog.",
            "Quantum computing uses qubits...",
            "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            "Bonjour le monde",
            "4",
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_generation_config():
    """Create a sample generation config."""
    from src.config import GenerationConfig
    return GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1,
    )


@pytest.fixture
def sample_eval_dataset_config():
    """Create a sample evaluation dataset config."""
    from src.config import EvalDatasetConfig
    return EvalDatasetConfig(
        name="test_dataset",
        path="test/path",
        split="test",
        prompt_template="alpaca",
        system_message="You are a helpful assistant.",
    )


@pytest.fixture
def mock_peft_model(mock_model):
    """Create a mock PEFT model."""
    peft_model = MagicMock()
    peft_model.base_model = mock_model
    peft_model.config = mock_model.config
    peft_model.generate = mock_model.generate
    peft_model.merge_and_unload = Mock(return_value=mock_model)
    peft_model.parameters = mock_model.parameters
    peft_model.named_parameters = mock_model.named_parameters
    return peft_model


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "### Instruction:\nTest prompt 1\n\n### Response:\n",
        "### Instruction:\nTest prompt 2\n\n### Response:\n",
        "### Instruction:\nTest prompt 3\n\n### Response:\n",
    ]


@pytest.fixture
def sample_responses():
    """Sample responses for testing."""
    return [
        "Response 1",
        "Response 2",
        "Response 3",
    ]


@pytest.fixture
def sample_references():
    """Sample reference texts for testing."""
    return [
        "Reference 1",
        "Reference 2",
        "Reference 3",
    ]


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_dataset_dict(
        num_samples: int = 10,
        include_input: bool = True,
        template: str = "alpaca"
    ) -> Dict[str, List]:
        """Create a dataset dictionary."""
        import random
        import string

        instructions = [
            f"Task {i}: " + "".join(random.choices(string.ascii_lowercase, k=20))
            for i in range(num_samples)
        ]
        inputs = [
            "".join(random.choices(string.ascii_lowercase + " ", k=30)) if include_input else ""
            for _ in range(num_samples)
        ]
        outputs = [
            "".join(random.choices(string.ascii_lowercase + " ", k=40))
            for _ in range(num_samples)
        ]

        return {
            "instruction": instructions,
            "input": inputs,
            "output": outputs,
        }

    @staticmethod
    def create_formatted_prompts(
        num_prompts: int = 5,
        template: str = "alpaca"
    ) -> List[str]:
        """Create formatted prompts."""
        templates = {
            "alpaca": "### Instruction:\n{instruction}\n\n### Response:\n",
            "chatml": "im_start>user\n{instruction}im_end>\nim_start>assistant\n",
        }
        tpl = templates.get(template, templates["alpaca"])
        return [tpl.format(instruction=f"Task {i}") for i in range(num_prompts)]


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "pipeline: Pipeline tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
        elif "pipeline" in str(item.fspath):
            item.add_marker(pytest.mark.pipeline)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
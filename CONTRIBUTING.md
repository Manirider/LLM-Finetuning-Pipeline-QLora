# Contributing to LLM Fine-Tuning Pipeline

Thank you for your interest in contributing! This guide will help you get started.


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Conventions](#commit-conventions)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- Docker & Docker Compose (for containerized workflows)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/llm-finetuning-pipeline-lora.git
cd llm-finetuning-pipeline-lora

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Verify Installation

```bash
# Run the test suite
make test

# Run linting
make lint

# Run type checking
make typecheck
```

## Project Structure

```
src/
├── config.py          # Pydantic configuration management
├── data_pipeline.py   # Data loading, cleaning, formatting
├── model_utils.py     # Model loading, QLoRA, PEFT utilities
├── train.py           # SFTTrainer orchestration
├── evaluate.py        # Evaluation metrics & reporting
├── inference.py       # FastAPI inference server
├── callbacks.py       # Training callbacks
├── metrics.py         # Metric computation
├── logger.py          # Structured logging
└── utils.py           # Shared utilities
```

## Coding Standards

### Style

- **Formatter**: We use [Ruff](https://docs.astral.sh/ruff/) for formatting (`ruff format`)
- **Linter**: Ruff for linting (`ruff check`)
- **Type Checker**: [mypy](https://mypy-lang.org/) with strict mode for `src/`
- **Line Length**: 100 characters

### Conventions

1. **Type Hints**: All public functions must have complete type annotations.
2. **Docstrings**: Google-style docstrings on all public classes and functions.
3. **Imports**: Use absolute imports (`from src.config import ...`).
4. **Constants**: Use `UPPER_SNAKE_CASE`. No magic numbers.
5. **Configuration**: All tunable parameters must be externalized in YAML configs.
6. **Logging**: Use `logging.getLogger(__name__)` — never `print()`.
7. **Error Handling**: Catch specific exceptions. Never use bare `except:`.

### Example Docstring

```python
def load_model_and_tokenizer(
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    quantization_config: Optional[QuantizationConfig] = None,
) -> ModelLoadResult:
    """Load a pretrained model and tokenizer with optional quantization.

    Args:
        model_config: Model configuration specifying name, dtype, and device map.
        tokenizer_config: Tokenizer configuration for padding and special tokens.
        quantization_config: Optional BitsAndBytes quantization settings for QLoRA.

    Returns:
        ModelLoadResult containing the loaded model, tokenizer, and metadata.

    Raises:
        ValueError: If the model name is empty or invalid.
        RuntimeError: If CUDA is required but unavailable.

    Example:
        >>> result = load_model_and_tokenizer(model_cfg, tok_cfg)
        >>> print(result.model.config.hidden_size)
        4096
    """
```

## Testing

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Smoke tests only
make test-smoke

# With coverage report
make test-cov
```

### Writing Tests

- Place unit tests in `tests/unit/test_<module>.py`
- Place integration tests in `tests/integration/test_<module>.py`
- Place smoke tests in `tests/smoke/test_smoke.py`
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.smoke`
- Target **>90% code coverage**
- Mock external dependencies (GPU, network, HF Hub)

### Test Structure

```python
"""Tests for src/<module>.py."""

import pytest
from unittest.mock import MagicMock, patch

from src.<module> import SomeClass


class TestSomeClass:
    """Tests for SomeClass."""

    def test_initialization(self):
        """Test default initialization."""
        obj = SomeClass()
        assert obj.value == expected

    def test_edge_case(self):
        """Test behavior with empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SomeClass(value="")
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature develop
   ```

2. **Make your changes** following the coding standards above.

3. **Write tests** for any new functionality.

4. **Run the full validation suite**:
   ```bash
   make lint
   make typecheck
   make test-cov
   ```

5. **Commit** using conventional commit messages (see below).

6. **Push** your branch and open a Pull Request against `develop`.

7. **Address review feedback** promptly.

### PR Checklist

- [ ] Tests pass locally (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Coverage ≥ 90%
- [ ] Documentation updated (if applicable)
- [ ] No hardcoded values (use config YAML)
- [ ] No secrets or API keys committed


## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code restructuring |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `build` | Build system or dependencies |
| `ci` | CI/CD configuration |
| `chore` | Maintenance tasks |

### Examples

```
feat(data): add Alpaca prompt formatter with validation
fix(train): resolve OOM during gradient accumulation
docs(readme): add architecture diagram and quick start
test(metrics): add edge case tests for ROUGE calculator
```



## Reporting Issues

When opening an issue, please include:

1. **Environment**: OS, Python version, CUDA version, GPU model
2. **Steps to reproduce**: Minimal config and commands
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Logs**: Relevant log output (redact API keys!)
6. **Configuration**: Relevant YAML config sections

Use the appropriate issue template if available.

## Questions?

If you have questions about contributing, please open a
[Discussion](https://github.com/your-org/llm-finetuning-pipeline-lora/discussions).

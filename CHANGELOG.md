# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete LLM Fine-tuning Pipeline with QLoRA/LoRA support
- Configuration-driven architecture with Pydantic validation
- Data pipeline: download, validation, cleaning, formatting, tokenization, splitting
- Model utilities: QLoRA, PEFT, Flash Attention 2/3, gradient checkpointing
- Training: SFTTrainer with callbacks, W&B/TensorBoard/MLflow tracking
- Evaluation: ROUGE, BLEU, METEOR, BERTScore, Perplexity, Distinct-n
- Inference: FastAPI server, batch processing, streaming
- Comprehensive documentation and troubleshooting guides

### Changed
- N/A (initial release)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A


## [1.0.0] - 2024-XX-XX

### Added
- Initial release of LLM Fine-tuning Pipeline
- QLoRA/LoRA fine-tuning with 4-bit NF4 quantization
- Flash Attention 2/3 support for memory-efficient training
- Gradient checkpointing and CPU offloading
- Multi-GPU DDP/FSDP/DeepSpeed support
- Comprehensive evaluation metrics
- Production-ready inference server


## Version Format

This project follows [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes to public APIs
- **Minor** (X.Y.0): New features, backward compatible
- **Patch** (X.Y.Z): Bug fixes, backward compatible
- **Pre-release**: `-alpha`, `-beta`, `-rc` suffixes

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. GitHub Actions will automatically:
   - Run tests and validation
   - Build and publish to PyPI
   - Build and push Docker image to GHCR
   - Create GitHub Release
   - Generate release notes

## Links

- [PyPI](https://pypi.org/project/llm-finetuning-pipeline/)
- [GitHub Releases](https://github.com/your-org/llm-finetuning-pipeline-lora/releases)
- [Documentation](https://github.com/your-org/llm-finetuning-pipeline-lora/tree/main/docs)
- [Docker Hub](https://hub.docker.com/r/your-org/llm-finetuning-pipeline)
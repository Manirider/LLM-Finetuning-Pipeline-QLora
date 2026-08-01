"""Unit tests for src/inference.py."""

from unittest.mock import MagicMock

from src.inference import ModelInstance, ModelStatus


class TestModelStatus:
    """Tests for ModelStatus enum."""

    def test_status_values(self):
        assert ModelStatus.LOADING == "loading"
        assert ModelStatus.READY == "ready"
        assert ModelStatus.ERROR == "error"
        assert ModelStatus.UNLOADED == "unloaded"


class TestModelInstance:
    """Tests for ModelInstance wrapper."""

    def test_model_instance_defaults(self):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        cfg = MagicMock()

        inst = ModelInstance(model=mock_model, tokenizer=mock_tok, config=cfg)
        assert inst.status == ModelStatus.UNLOADED
        assert inst.request_count == 0
        assert inst.total_tokens_generated == 0

    def test_model_instance_custom_status(self):
        inst = ModelInstance(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=MagicMock(),
            status=ModelStatus.READY,
        )
        assert inst.status == ModelStatus.READY

"""Integration tests for model utilities."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model_utils import (
    apply_peft_model,
    count_parameters,
    create_bnb_config,
    create_lora_config,
    get_model_memory_footprint,
    load_base_model,
    load_model_and_tokenizer,
    load_tokenizer,
    merge_and_unload_peft,
    optimize_model_memory,
    prepare_model_for_training,
    print_model_summary,
    save_model_and_tokenizer,
    setup_flash_attention,
    setup_gradient_checkpointing,
)
from src.config import (
    ModelConfig,
    TokenizerConfig,
    PEFTLoraConfig,
    QuantizationConfig,
    RuntimeConfig,
    LoRAConfig,
    TrainingConfig,
)


class TestModelLoadingIntegration:
    """Integration tests for model loading."""

    def test_create_bnb_config_4bit(self):
        """Test BitsAndBytes config creation for 4-bit."""
        quant_config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        bnb_config = create_bnb_config(quant_config)
        
        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_compute_dtype == torch.bfloat16
        assert bnb_config.bnb_4bit_use_double_quant is True

    def test_create_bnb_config_8bit(self):
        """Test BitsAndBytes config creation for 8-bit."""
        quant_config = QuantizationConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        bnb_config = create_bnb_config(quant_config)
        
        assert bnb_config.load_in_8bit is True
        assert bnb_config.llm_int8_threshold == 6.0

    def test_create_lora_config(self):
        """Test LoRA config creation."""
        peft_config = PEFTLoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            use_rslora=True,
            target_modules=["q_proj", "v_proj"],
        )
        lora_config = create_lora_config(peft_config)
        
        assert lora_config.r == 64
        assert lora_config.lora_alpha == 16
        assert set(lora_config.target_modules) == {"q_proj", "v_proj"}

    @patch("src.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("src.model_utils.AutoTokenizer.from_pretrained")
    def test_load_tokenizer(self, mock_tokenizer, mock_model):
        """Test tokenizer loading."""
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tok
        
        tokenizer_config = TokenizerConfig(
            tokenizer_name_or_path="test-model",
            padding_side="right",
        )
        
        with patch("src.model_utils.AutoTokenizer.from_pretrained", return_value=mock_tok):
            tokenizer = load_tokenizer(tokenizer_config)
        
        assert tokenizer == mock_tok
        assert tokenizer.pad_token == "<eos>"

    @patch("src.model_utils.AutoModelForCausalLM.from_pretrained")
    @patch("src.model_utils.AutoTokenizer.from_pretrained")
    def test_load_base_model(self, mock_tokenizer, mock_model_class):
        """Test base model loading."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.use_cache = True
        mock_model_class.return_value = mock_model
        
        model_config = ModelConfig(
            model_name_or_path="test-model",
            torch_dtype="bfloat16",
            device_map="auto",
            load_in_4bit=True,
        )
        
        quant_config = QuantizationConfig(load_in_4bit=True)
        bnb_config = create_bnb_config(quant_config)
        
        with patch("src.model_utils.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            model = load_base_model(model_config, quantization_config=bnb_config)
        
        assert model == mock_model

    def test_setup_flash_attention(self):
        """Test flash attention setup."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        
        model = setup_flash_attention(mock_model, enable=True, version=2)
        assert model.config._attn_implementation in ["flash_attention_2", "eager"]

    def test_setup_gradient_checkpointing(self):
        """Test gradient checkpointing setup."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        
        model = setup_gradient_checkpointing(mock_model, True)
        mock_model.gradient_checkpointing_enable.assert_called_once()
        assert mock_model.config.use_cache is False

    @patch("src.model_utils.prepare_model_for_kbit_training")
    def test_prepare_model_for_training(self, mock_prepare):
        """Test model preparation for k-bit training."""
        mock_model = MagicMock()
        mock_prepare.return_value = mock_model
        
        model = prepare_model_for_training(mock_model)
        mock_prepare.assert_called_once()

    def test_apply_peft_model_lora(self):
        """Test PEFT model application with LoRA."""
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        
        peft_config = PEFTLoraConfig(r=64)
        
        with patch("src.model_utils.get_peft_model", return_value=mock_peft_model):
            model = apply_peft_model(mock_model, peft_config, "LORA")
        
        assert model == mock_peft_model

    def test_count_parameters(self):
        """Test parameter counting."""
        mock_model = MagicMock()
        
        # Create mock parameters
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        
        param2 = MagicMock()
        param2.numel.return_value = 2000
        param2.requires_grad = False
        
        param3 = MagicMock()
        param3.numel.return_value = 500
        param3.requires_grad = True
        
        mock_model.named_parameters.return_value = [
            ("lora_A", param1),
            ("base_param", param2),
            ("lora_B", param3),
        ]
        
        stats = count_parameters(mock_model)
        
        assert stats.total_params == 3500
        assert stats.trainable_params == 1500
        assert stats.frozen_params == 2000
        assert stats.lora_params == 1500

    def test_get_model_memory_footprint(self):
        """Test memory footprint calculation."""
        mock_model = MagicMock()
        
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.element_size.return_value = 2  # bfloat16
        
        buffer1 = MagicMock()
        buffer1.numel.return_value = 500
        buffer1.element_size.return_value = 2
        
        mock_model.parameters.return_value = [param1]
        mock_model.buffers.return_value = [buffer1]
        
        mem = get_model_memory_footprint(mock_model)
        
        assert mem["parameters_memory_mb"] > 0
        assert mem["buffers_memory_mb"] > 0
        assert mem["total_memory_mb"] > 0

    def test_optimize_model_memory(self):
        """Test model memory optimization."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model.register_forward_hook = MagicMock()
        
        model = optimize_model_memory(
            mock_model,
            enable_flash_attention=True,
            gradient_checkpointing=True,
        )
        
        assert model.config._attn_implementation in ["flash_attention_2", "eager"]
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_merge_and_unload_peft(self):
        """Test PEFT model merging."""
        mock_peft_model = MagicMock()
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged
        
        model = merge_and_unload_peft(mock_peft_model)
        
        assert model == mock_merged
        mock_peft_model.merge_and_unload.assert_called_once()

    def test_save_model_and_tokenizer(self):
        """Test model and tokenizer saving."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_and_tokenizer(
                mock_model,
                mock_tokenizer,
                tmpdir,
                save_adapter=False,
                safe_serialization=True,
            )
            
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

    def test_print_model_summary(self, capfd):
        """Test model summary printing."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LlamaForCausalLM"
        mock_model.config.model_type = "llama"
        
        param1 = MagicMock()
        param1.numel.return_value = 1000000000
        param1.requires_grad = True
        
        param2 = MagicMock()
        param2.numel.return_value = 50000000
        param2.requires_grad = True
        
        mock_model.named_parameters.return_value = [
            ("base", param1),
            ("lora_A", param2),
        ]
        
        print_model_summary(mock_model, input_shape=(1, 512))
        
        out, _ = capfd.readouterr()
        assert "MODEL SUMMARY" in out
        assert "LlamaForCausalLM" in out
        assert "Trainable" in out


class TestModelPipelineIntegration:
    """Integration tests for the complete model pipeline."""

    @patch("src.model_utils.load_model_config")
    @patch("src.model_utils.load_base_model")
    @patch("src.model_utils.load_tokenizer")
    @patch("src.model_utils.apply_peft_model")
    @patch("src.model_utils.prepare_model_for_training")
    @patch("src.model_utils.optimize_model_memory")
    def test_load_model_and_tokenizer_pipeline(
        self,
        mock_optimize,
        mock_prepare,
        mock_apply_peft,
        mock_load_tokenizer,
        mock_load_base,
        mock_load_config,
    ):
        """Test complete model and tokenizer loading pipeline."""
        mock_load_config.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.use_cache = True
        mock_load_base.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_peft_model = MagicMock()
        mock_apply_peft.return_value = mock_peft_model
        
        mock_prepared = MagicMock()
        mock_prepare.return_value = mock_prepared
        
        mock_optimized = MagicMock()
        mock_optimized.hf_device_map = {}
        mock_optimize.return_value = mock_optimized
        
        model_config = ModelConfig(
            model_name_or_path="test-model",
            load_in_4bit=True,
            gradient_checkpointing=True,
        )
        tokenizer_config = TokenizerConfig(
            tokenizer_name_or_path="test-model",
        )
        peft_config = PEFTLoraConfig(r=64)
        runtime_config = RuntimeConfig(
            flash_attention=True,
            gradient_checkpointing=True,
        )
        quant_config = QuantizationConfig(load_in_4bit=True)
        
        with patch("src.model_utils.create_bnb_config") as mock_bnb:
            mock_bnb.return_value = MagicMock()
            
            result = load_model_and_tokenizer(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                quantization_config=quant_config,
                peft_config=peft_config,
                peft_type="LORA",
                runtime_config=runtime_config,
            )
        
        assert result.model is not None
        assert result.tokenizer is not None
        assert result.quantization_config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Tests for model module."""

import pytest


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_import_quantization_config(self):
        from src.models import (
            QuantizationConfig,
            get_default_quantization_config,
            estimate_memory_usage,
        )

        assert QuantizationConfig is not None
        assert get_default_quantization_config is not None
        assert estimate_memory_usage is not None

    def test_estimate_memory_usage(self):
        from src.models import estimate_memory_usage

        memory_fp32 = estimate_memory_usage(7_000_000_000, "fp32")
        memory_nf4 = estimate_memory_usage(7_000_000_000, "nf4")

        assert memory_fp32 > memory_nf4
        assert memory_fp32 > 20

    def test_quantization_to_dict(self):
        from src.models import QuantizationConfig

        config = QuantizationConfig()
        config_dict = config.to_dict()

        assert "load_in_4bit" in config_dict
        assert config_dict["load_in_4bit"] is True


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_import_model_factory(self):
        from src.models import ModelFactory, create_model, MODEL_FAMILIES

        assert ModelFactory is not None
        assert create_model is not None
        assert "llama" in MODEL_FAMILIES

    def test_model_family_detection(self):
        from src.models import ModelFactory

        factory = ModelFactory("meta-llama/Llama-2-7b-hf")
        assert factory.model_family == "llama"

        factory = ModelFactory("Qwen/Qwen2-7B")
        assert factory.model_family == "qwen"

        factory = ModelFactory("google/gemma-7b")
        assert factory.model_family == "gemma"

    def test_target_modules(self):
        from src.models import ModelFactory

        factory = ModelFactory("meta-llama/Llama-2-7b-hf", model_family="llama")

        modules = factory.get_target_modules()
        assert "q_proj" in modules
        assert "v_proj" in modules

        full_modules = factory.get_target_modules(use_full_modules=True)
        assert len(full_modules) > len(modules)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_import_lora_config(self):
        from src.models import LoRAConfig, get_default_lora_config

        assert LoRAConfig is not None
        assert get_default_lora_config is not None

    def test_default_lora_config(self):
        from src.models import LoRAConfig

        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32

    def test_get_default_lora_config(self):
        from src.models import get_default_lora_config

        config = get_default_lora_config("llama", rank=8)
        assert config.r == 8
        assert config.target_modules is not None


class TestPEFTModel:
    """Tests for PEFT model wrapper."""

    def test_import_peft_model(self):
        from src.models import PEFTModelWrapper, create_peft_model

        assert PEFTModelWrapper is not None
        assert create_peft_model is not None

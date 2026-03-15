"""Tests for configuration module."""

import pytest

from src.config import (
    QLoRAConfig,
    QuantizationConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    ModelConfig,
    get_model_target_modules,
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_values(self):
        config = QuantizationConfig()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_custom_values(self):
        config = QuantizationConfig(
            load_in_4bit=False,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
        )
        assert config.load_in_4bit is False
        assert config.bnb_4bit_compute_dtype == "bfloat16"
        assert config.bnb_4bit_quant_type == "fp4"
        assert config.bnb_4bit_use_double_quant is False

    def test_to_dict(self):
        config = QuantizationConfig()
        config_dict = config.to_dict()
        assert "load_in_4bit" in config_dict
        assert "bnb_4bit_compute_dtype" in config_dict


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "none"

    def test_custom_values(self):
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj"],
            bias="all",
        )
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert len(config.target_modules) == 3
        assert config.bias == "all"

    def test_to_dict(self):
        config = LoRAConfig(r=8)
        config_dict = config.to_dict()
        assert config_dict["r"] == 8


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        config = DataConfig()
        assert config.train_file == "data/processed/train.json"
        assert config.validation_file is None
        assert config.max_seq_length == 512
        assert config.train_split == 0.9

    def test_custom_values(self):
        config = DataConfig(
            train_file="custom/train.json",
            validation_file="custom/val.json",
            max_seq_length=1024,
        )
        assert config.train_file == "custom/train.json"
        assert config.validation_file == "custom/val.json"
        assert config.max_seq_length == 1024


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.output_dir == "./output"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 1
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
        assert config.fp16 is False

    def test_custom_values(self):
        config = TrainingConfig(
            output_dir="./custom_output",
            num_train_epochs=5,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
        )
        assert config.output_dir == "./custom_output"
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 2
        assert config.learning_rate == 1e-4


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.model_family is None
        assert config.trust_remote_code is True

    def test_custom_values(self):
        config = ModelConfig(
            model_name="Qwen/Qwen2-7B",
            model_family="qwen",
            trust_remote_code=False,
        )
        assert config.model_name == "Qwen/Qwen2-7B"
        assert config.model_family == "qwen"
        assert config.trust_remote_code is False


class TestQLoRAConfig:
    """Tests for QLoRAConfig."""

    def test_default_config(self):
        config = QLoRAConfig()
        assert config.model is not None
        assert config.quantization is not None
        assert config.lora is not None
        assert config.data is not None
        assert config.training is not None
        assert config.seed == 42

    def test_validation_errors(self):
        config = QLoRAConfig()
        config.data.max_seq_length = 32
        errors = config.validate()
        assert len(errors) > 0
        assert "max_seq_length" in errors[0]

    def test_yaml_roundtrip(self, tmp_path):
        config = QLoRAConfig()
        config.training.output_dir = str(tmp_path / "output")

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))

        loaded_config = QLoRAConfig.from_yaml(str(yaml_path))
        assert loaded_config.training.output_dir == config.training.output_dir


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_model_target_modules_llama(self):
        modules = get_model_target_modules("llama")
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_get_model_target_modules_qwen(self):
        modules = get_model_target_modules("qwen")
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_get_model_target_modules_gemma(self):
        modules = get_model_target_modules("gemma")
        assert "q_proj" in modules
        assert "v_proj" in modules

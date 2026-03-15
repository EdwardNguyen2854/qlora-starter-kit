"""
QLoRA Configuration Module.

Provides configuration classes for QLoRA fine-tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


ModelFamily = Literal["llama", "qwen", "gemma"]
QuantizationType = Literal["nf4", "fp4"]
ComputeDtype = Literal["float32", "float16", "bfloat16", "auto"]
BiasType = Literal["none", "all", "lora_only"]
TaskType = Literal["CAUSAL_LM", "SEQ_CLS"]


@dataclass
class QuantizationConfig:
    """Configuration for 4-bit quantization."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: ComputeDtype = "float16"
    bnb_4bit_quant_type: QuantizationType = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: BiasType = "none"
    task_type: TaskType = "CAUSAL_LM"
    inference_mode: bool = False

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.r * 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
        }


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    train_file: str = "data/processed/train.json"
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    data_format: Literal["json", "csv"] = "json"
    text_field: str = "text"
    prompt_field: str = "prompt"
    response_field: str = "response"
    max_seq_length: int = 512
    train_split: float = 0.9
    validation_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""

    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    max_grad_norm: float = 0.3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_peft: bool = True
    neftune_noise_alpha: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    model_family: Optional[ModelFamily] = None
    trust_remote_code: bool = True
    use_cache: bool = False
    torch_dtype: ComputeDtype = "auto"
    device_map: str = "auto"


@dataclass
class QLoRAConfig:
    """Main configuration class for QLoRA fine-tuning."""

    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    quantization: QuantizationConfig = field(
        default_factory=lambda: QuantizationConfig()
    )
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig())
    data: DataConfig = field(default_factory=lambda: DataConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "qlora-starter-kit"

    @classmethod
    def from_yaml(cls, path: str) -> "QLoRAConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QLoRAConfig":
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        quantization_config = QuantizationConfig(
            **config_dict.get("quantization", {})
        )
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            model=model_config,
            quantization=quantization_config,
            lora=lora_config,
            data=data_config,
            training=training_config,
            seed=config_dict.get("seed", 42),
            use_wandb=config_dict.get("use_wandb", False),
            wandb_project=config_dict.get("wandb_project", "qlora-starter-kit"),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "quantization": self.quantization.to_dict(),
            "lora": self.lora.to_dict(),
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "seed": self.seed,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.model.model_family is None:
            errors.append("model.model_family must be specified")

        if self.data.max_seq_length < 64:
            errors.append("data.max_seq_length must be at least 64")

        if self.training.gradient_accumulation_steps < 1:
            errors.append("training.gradient_accumulation_steps must be >= 1")

        if self.lora.r < 1:
            errors.append("lora.r must be at least 1")

        if self.training.bf16 and self.training.fp16:
            errors.append("Cannot use both bf16 and fp16")

        return errors


DEFAULT_CONFIGS = {
    "llama-7b": {
        "model": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "model_family": "llama",
        },
        "lora": {
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
    },
    "llama-13b": {
        "model": {
            "model_name": "meta-llama/Llama-2-13b-hf",
            "model_family": "llama",
        },
        "lora": {
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
    },
    "qwen-7b": {
        "model": {
            "model_name": "Qwen/Qwen2-7B",
            "model_family": "qwen",
        },
        "lora": {
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
    },
    "gemma-7b": {
        "model": {
            "model_name": "google/gemma-7b",
            "model_family": "gemma",
        },
        "lora": {
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
    },
}


def get_model_target_modules(model_family: ModelFamily) -> List[str]:
    """Get default target modules for a model family."""
    target_modules_map = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    return target_modules_map.get(model_family, ["q_proj", "v_proj"])


def create_config(model_family: ModelFamily, **overrides) -> QLoRAConfig:
    """Create a configuration for a specific model family with overrides."""
    base_config = DEFAULT_CONFIGS.get(model_family)

    if base_config is None:
        available = ", ".join(DEFAULT_CONFIGS.keys())
        raise ValueError(
            f"Unknown model family: {model_family}. Available: {available}"
        )

    config_dict = {
        "model": base_config.get("model", {}),
        "lora": base_config.get("lora", {}),
    }

    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            if section not in config_dict:
                config_dict[section] = {}
            config_dict[section][field] = value
        else:
            config_dict[key] = value

    return QLoRAConfig.from_dict(config_dict)

"""
Quantization module for QLoRA.

Provides 4-bit NF4 quantization using bitsandbytes.
"""

from typing import Any, Dict, Literal, Optional

from transformers import BitsAndBytesConfig as HFBitsAndBytesConfig


class QuantizationConfig:
    """Configuration for 4-bit quantization with bitsandbytes."""

    def __init__(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: Literal["float32", "float16", "bfloat16"] = "float16",
        bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ):
        """Initialize quantization config.

        Args:
            load_in_4bit: Whether to load model in 4-bit
            bnb_4bit_compute_dtype: Compute dtype for 4-bit
            bnb_4bit_quant_type: Quantization type (nf4 or fp4)
            bnb_4bit_use_double_quant: Use double quantization
        """
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

    def to_hf_config(self) -> HFBitsAndBytesConfig:
        """Convert to HuggingFace BitsAndBytesConfig.

        Returns:
            HuggingFace BitsAndBytesConfig
        """
        dtype_map = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
        }

        import torch

        return HFBitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=dtype_map.get(
                self.bnb_4bit_compute_dtype, torch.float16
            ),
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create from dictionary."""
        return cls(
            load_in_4bit=config_dict.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=config_dict.get(
                "bnb_4bit_compute_dtype", "float16"
            ),
            bnb_4bit_quant_type=config_dict.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config_dict.get(
                "bnb_4bit_use_double_quant", True
            ),
        )


def get_default_quantization_config(
    compute_dtype: Literal["float32", "float16", "bfloat16"] = "float16",
) -> QuantizationConfig:
    """Get default quantization config for QLoRA.

    Args:
        compute_dtype: Compute dtype to use

    Returns:
        QuantizationConfig with recommended defaults
    """
    return QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def estimate_memory_usage(
    num_parameters: int,
    quantization_type: Literal["fp32", "fp16", "int8", "nf4"] = "nf4",
) -> float:
    """Estimate memory usage for a model.

    Args:
        num_parameters: Number of model parameters
        quantization_type: Type of quantization

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "nf4": 0.5,
    }

    bytes_per = bytes_per_param.get(quantization_type, 2)
    memory_gb = (num_parameters * bytes_per) / (1024**3)

    return memory_gb


def get_model_memory_requirements(
    model_name: str,
    quantization_type: Literal["fp32", "fp16", "int8", "nf4"] = "nf4",
) -> Dict[str, float]:
    """Get estimated memory requirements for common models.

    Args:
        model_name: Model name or family
        quantization_type: Type of quantization

    Returns:
        Dictionary with memory estimates in GB
    """
    model_params = {
        "llama-2-7b": 7_000_000_000,
        "llama-2-13b": 13_000_000_000,
        "llama-2-70b": 70_000_000_000,
        "llama-3-8b": 8_000_000_000,
        "llama-3-70b": 70_000_000_000,
        "qwen-1.8b": 1_800_000_000,
        "qwen-7b": 7_000_000_000,
        "qwen-14b": 14_000_000_000,
        "gemma-2b": 2_000_000_000,
        "gemma-7b": 7_000_000_000,
    }

    model_key = model_name.lower().replace("-", "-").replace("_", "-")

    for key, params in model_params.items():
        if key in model_key:
            return {
                "parameters": params,
                "memory_fp32": estimate_memory_usage(params, "fp32"),
                "memory_fp16": estimate_memory_usage(params, "fp16"),
                "memory_int8": estimate_memory_usage(params, "int8"),
                "memory_nf4": estimate_memory_usage(params, "nf4"),
            }

    return {}

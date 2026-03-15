"""
Model factory module for QLoRA.

Handles loading of Llama, Qwen, and Gemma models with quantization.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.models.quantization import QuantizationConfig


MODEL_FAMILIES = {
    "llama": {
        "default_target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "min_target_modules": ["q_proj", "v_proj"],
    },
    "qwen": {
        "default_target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "min_target_modules": ["q_proj", "v_proj"],
    },
    "gemma": {
        "default_target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "min_target_modules": ["q_proj", "v_proj"],
    },
}


class ModelFactory:
    """Factory for creating and loading models."""

    def __init__(
        self,
        model_name: str,
        model_family: Optional[Literal["llama", "qwen", "gemma"]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize ModelFactory.

        Args:
            model_name: HuggingFace model name or local path
            model_family: Model family (llama, qwen, gemma). Auto-detected if None
            trust_remote_code: Trust remote code
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.model_family = model_family or self._detect_model_family(model_name)
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "./models")

    def _detect_model_family(self, model_name: str) -> Optional[str]:
        """Auto-detect model family from model name.

        Args:
            model_name: Model name

        Returns:
            Detected model family or None
        """
        model_lower = model_name.lower()

        if "llama" in model_lower:
            return "llama"
        elif "qwen" in model_lower:
            return "qwen"
        elif "gemma" in model_lower:
            return "gemma"

        return None

    def get_target_modules(
        self,
        target_modules: Optional[List[str]] = None,
        use_minimal: bool = False,
    ) -> List[str]:
        """Get target modules for LoRA.

        Args:
            target_modules: Custom target modules
            use_minimal: Use minimal target modules

        Returns:
            List of target module names
        """
        if target_modules:
            return target_modules

        if self.model_family is None:
            return ["q_proj", "v_proj"]

        family_config = MODEL_FAMILIES.get(self.model_family, {})

        if use_minimal:
            return family_config.get("min_target_modules", ["q_proj", "v_proj"])

        return family_config.get(
            "default_target_modules", ["q_proj", "v_proj"]
        )

    def load_tokenizer(
        self,
        use_fast: bool = True,
        padding_side: str = "right",
    ) -> PreTrainedTokenizer:
        """Load tokenizer.

        Args:
            use_fast: Use fast tokenizer
            padding_side: Padding side (left or right)

        Returns:
            Loaded tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast,
            cache_dir=self.cache_dir,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = padding_side

        return tokenizer

    def load_model(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        use_cache: bool = False,
    ) -> PreTrainedModel:
        """Load model with optional quantization.

        Args:
            quantization_config: Quantization config
            device_map: Device mapping
            torch_dtype: Torch dtype
            use_cache: Use KV cache

        Returns:
            Loaded model
        """
        hf_quantization_config = None

        if quantization_config is not None:
            hf_quantization_config = quantization_config.to_hf_config()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=hf_quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            use_cache=use_cache,
            cache_dir=self.cache_dir,
        )

        return model

    def load_model_and_tokenizer(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        use_cache: bool = False,
        padding_side: str = "right",
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load both model and tokenizer.

        Args:
            quantization_config: Quantization config
            device_map: Device mapping
            torch_dtype: Torch dtype
            use_cache: Use KV cache
            padding_side: Padding side

        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_model(
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_cache=use_cache,
        )

        tokenizer = self.load_tokenizer(padding_side=padding_side)

        return model, tokenizer


def create_model(
    model_name: str,
    model_family: Optional[Literal["llama", "qwen", "gemma"]] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Convenience function to create model and tokenizer.

    Args:
        model_name: Model name
        model_family: Model family
        quantization_config: Quantization config
        device_map: Device mapping
        torch_dtype: Torch dtype
        trust_remote_code: Trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    factory = ModelFactory(
        model_name=model_name,
        model_family=model_family,
        trust_remote_code=trust_remote_code,
    )

    return factory.load_model_and_tokenizer(
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

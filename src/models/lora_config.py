"""
LoRA configuration module for QLoRA.

Handles LoRA adapter configuration and PEFT integration.
"""

from typing import Any, Dict, List, Literal, Optional

from peft import LoraConfig as PEFTLoRAConfig


class LoRAConfig:
    """Configuration for LoRA adapters."""

    def __init__(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: Literal["none", "all", "lora_only"] = "none",
        task_type: Literal["CAUSAL_LM", "SEQ_CLS"] = "CAUSAL_LM",
        inference_mode: bool = False,
        modules_to_save: Optional[List[str]] = None,
    ):
        """Initialize LoRA config.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Target modules for LoRA
            bias: Bias configuration
            task_type: Task type
            inference_mode: Use inference mode
            modules_to_save: Additional modules to save
        """
        self.r = r
        self.lora_alpha = lora_alpha or r * 2
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.modules_to_save = modules_to_save

    def to_peft_config(self) -> PEFTLoRAConfig:
        """Convert to PEFT LoraConfig.

        Returns:
            PEFT LoraConfig
        """
        return PEFTLoRAConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            modules_to_save=self.modules_to_save,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
            "modules_to_save": self.modules_to_save,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        return cls(
            r=config_dict.get("r", 16),
            lora_alpha=config_dict.get("lora_alpha"),
            lora_dropout=config_dict.get("lora_dropout", 0.05),
            target_modules=config_dict.get("target_modules"),
            bias=config_dict.get("bias", "none"),
            task_type=config_dict.get("task_type", "CAUSAL_LM"),
            inference_mode=config_dict.get("inference_mode", False),
            modules_to_save=config_dict.get("modules_to_save"),
        )


def get_default_lora_config(
    model_family: str,
    rank: int = 16,
    use_full_modules: bool = False,
) -> LoRAConfig:
    """Get default LoRA config for a model family.

    Args:
        model_family: Model family (llama, qwen, gemma)
        rank: LoRA rank
        use_full_modules: Use full target modules

    Returns:
        LoRAConfig
    """
    target_modules_map = {
        "llama": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "qwen": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "gemma": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }

    min_modules_map = {
        "llama": ["q_proj", "v_proj"],
        "qwen": ["q_proj", "v_proj"],
        "gemma": ["q_proj", "v_proj"],
    }

    if use_full_modules:
        target_modules = target_modules_map.get(model_family)
    else:
        target_modules = min_modules_map.get(model_family, ["q_proj", "v_proj"])

    return LoRAConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def estimate_lora_parameters(
    base_model_params: int,
    rank: int = 16,
    num_target_modules: int = 2,
    hidden_size: int = 4096,
) -> int:
    """Estimate number of LoRA trainable parameters.

    Args:
        base_model_params: Number of base model parameters
        rank: LoRA rank
        num_target_modules: Number of target modules
        hidden_size: Hidden size of model

    Returns:
        Estimated number of LoRA parameters
    """
    lora_params_per_module = 2 * rank * hidden_size

    return lora_params_per_module * num_target_modules

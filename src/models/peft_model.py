"""
PEFT model wrapper for QLoRA.

Provides utilities for wrapping base models with LoRA adapters.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    inject_adapter_in_model,
)


class PEFTModelWrapper:
    """Wrapper for PEFT model operations."""

    def __init__(
        self,
        base_model: Any,
        lora_config: Optional[LoraConfig] = None,
    ):
        """Initialize PEFT model wrapper.

        Args:
            base_model: Base model to wrap
            lora_config: LoRA configuration
        """
        self.base_model = base_model
        self.lora_config = lora_config
        self.peft_model: Optional[PeftModel] = None

    def apply_lora(self, lora_config: LoraConfig) -> PeftModel:
        """Apply LoRA adapters to base model.

        Args:
            lora_config: LoRA configuration

        Returns:
            PEFT model
        """
        self.lora_config = lora_config
        self.peft_model = get_peft_model(self.base_model, lora_config)
        return self.peft_model

    def get_trainable_parameters(self) -> tuple[int, int]:
        """Get number of trainable vs total parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied. Call apply_lora() first.")

        trainable_params = 0
        total_params = 0

        for _, param in self.peft_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return trainable_params, total_params

    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics."""
        trainable, total = self.get_trainable_parameters()
        trainable_pct = 100 * trainable / total if total > 0 else 0

        print(f"trainable params: {trainable:,}")
        print(f"all params: {total:,}")
        print(f"trainable%: {trainable_pct:.4f}")

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str = "default",
        safe_serialization: bool = True,
    ) -> None:
        """Save LoRA adapters.

        Args:
            save_directory: Directory to save adapters
            adapter_name: Name of adapter to save
            safe_serialization: Use safe serialization
        """
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied. Call apply_lora() first.")

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(
            save_directory,
            adapter_name=adapter_name,
            safe_serialization=safe_serialization,
        )

    @classmethod
    def load_adapter(
        cls,
        base_model: Any,
        adapter_path: str,
        adapter_name: str = "default",
    ) -> "PEFTModelWrapper":
        """Load LoRA adapters.

        Args:
            base_model: Base model
            adapter_path: Path to saved adapter
            adapter_name: Name of adapter

        Returns:
            PEFTModelWrapper
        """
        peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=adapter_name,
        )

        wrapper = cls(base_model, None)
        wrapper.peft_model = peft_model

        return wrapper

    def merge_and_unload(
        self,
        adapter_name: str = "default",
    ) -> Any:
        """Merge LoRA weights into base model and unload.

        Args:
            adapter_name: Name of adapter to merge

        Returns:
            Merged model
        """
        if self.peft_model is None:
            raise RuntimeError("LoRA not applied. Call apply_lora() first.")

        return self.peft_model.merge_and_unload()

    def disable_adapter(self) -> None:
        """Disable LoRA adapters."""
        if self.peft_model is not None:
            self.peft_model.disable_adapter()

    def enable_adapter(self) -> None:
        """Enable LoRA adapters."""
        if self.peft_model is not None:
            self.peft_model.enable_adapter()


def create_peft_model(
    base_model: Any,
    lora_config: Union[LoraConfig, Dict[str, Any]],
    adapter_name: str = "default",
) -> tuple[PeftModel, int]:
    """Create PEFT model with LoRA.

    Args:
        base_model: Base model
        lora_config: LoRA config (PEFT LoraConfig or dict)
        adapter_name: Name for adapter

    Returns:
        Tuple of (peft_model, num_trainable_params)
    """
    if isinstance(lora_config, dict):
        lora_config = LoraConfig(**lora_config)

    peft_model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    return peft_model, trainable_params

"""
Training module for QLoRA fine-tuning.

Provides SFTTrainer wrapper with logging and checkpointing.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from peft import LoraConfig
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


class QLoRATrainer:
    """Trainer for QLoRA fine-tuning."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        lora_config: Optional[LoraConfig] = None,
        training_args: Optional[TrainingArguments] = None,
    ):
        """Initialize QLoRA Trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            lora_config: LoRA configuration
            training_args: Training arguments
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config = lora_config
        self.training_args = training_args
        self.trainer: Optional[SFTTrainer] = None

    def create_trainer(
        self,
        max_seq_length: int = 512,
        packing: bool = False,
        dataset_text_field: str = "text",
        neftune_noise_alpha: Optional[float] = None,
        compute_metrics: Optional[Any] = None,
    ) -> SFTTrainer:
        """Create SFTTrainer.

        Args:
            max_seq_length: Maximum sequence length
            packing: Use packing
            dataset_text_field: Text field in dataset
            neftune_noise_alpha: NEFTune noise alpha
            compute_metrics: Compute metrics function

        Returns:
            Configured SFTTrainer
        """
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.lora_config,
            max_seq_length=max_seq_length,
            packing=packing,
            dataset_text_field=dataset_text_field,
            args=self.training_args,
            neftune_noise_alpha=neftune_noise_alpha,
            compute_metrics=compute_metrics,
        )

        return self.trainer

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training.

        Args:
            resume_from_checkpoint: Checkpoint to resume from

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise RuntimeError("Call create_trainer() first.")

        output = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )

        return output.metrics

    def evaluate(
        self,
    ) -> Dict[str, Any]:
        """Run evaluation.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Call create_trainer() first.")

        return self.trainer.evaluate()

    def save_model(
        self,
        output_dir: Optional[str] = None,
        safe_serialization: bool = True,
    ) -> None:
        """Save model and tokenizer.

        Args:
            output_dir: Output directory
            safe_serialization: Use safe serialization
        """
        if self.trainer is None:
            raise RuntimeError("Call create_trainer() first.")

        output_dir = output_dir or self.training_args.output_dir

        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def get_model(self) -> PreTrainedModel:
        """Get the trained model."""
        if self.trainer is None:
            raise RuntimeError("Call create_trainer() first.")
        return self.trainer.model


def create_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    save_total_limit: int = 3,
    max_grad_norm: float = 0.3,
    bf16: bool = True,
    fp16: bool = False,
    gradient_checkpointing: bool = True,
    report_to: str = "none",
    run_name: Optional[str] = None,
    seed: int = 42,
) -> TrainingArguments:
    """Create TrainingArguments.

    Args:
        output_dir: Output directory
        num_train_epochs: Number of epochs
        per_device_train_batch_size: Train batch size
        per_device_eval_batch_size: Eval batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        logging_steps: Logging steps
        save_steps: Save steps
        eval_steps: Eval steps
        save_total_limit: Save limit
        max_grad_norm: Max gradient norm
        bf16: Use bf16
        fp16: Use fp16
        gradient_checkpointing: Use gradient checkpointing
        report_to: Report to
        run_name: Run name
        seed: Random seed

    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        ddp_find_unused_parameters=False,
    )

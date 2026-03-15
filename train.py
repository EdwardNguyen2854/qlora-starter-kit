#!/usr/bin/env python3
"""
Main training script for QLoRA fine-tuning.

Usage:
    python train.py --config configs/qlora_config.yaml
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train QLoRA model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qlora_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Override model name in config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory in config",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Override number of epochs in config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed in config",
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample data before training",
    )

    args = parser.parse_args()

    from src.config import QLoRAConfig
    from src.data.dataset_loader import DatasetLoader
    from src.data.preprocessor import DataPreprocessor
    from src.models import (
        QuantizationConfig,
        LoRAConfig,
        ModelFactory,
    )
    from src.training import QLoRATrainer, create_training_args

    config = QLoRAConfig.from_yaml(args.config)

    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.seed:
        config.seed = args.seed

    set_seed(config.seed)

    print("=" * 50)
    print("QLoRA Training")
    print("=" * 50)
    print(f"Model: {config.model.model_name}")
    print(f"Output: {config.training.output_dir}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"LoRA rank: {config.lora.r}")
    print("=" * 50)

    if args.create_sample_data:
        from scripts.prepare_data import create_sample_data

        sample_path = Path("data/processed/train.json")
        create_sample_data(sample_path, 100)

    print("\nLoading tokenizer...")
    factory = ModelFactory(
        model_name=config.model.model_name,
        model_family=config.model.model_family,
        trust_remote_code=config.model.trust_remote_code,
    )

    tokenizer = factory.load_tokenizer()

    print("Loading model with quantization...")
    quant_config = QuantizationConfig.from_dict(config.quantization.to_dict())

    model = factory.load_model(
        quantization_config=quant_config,
        device_map=config.model.device_map,
        torch_dtype=config.model.torch_dtype,
        use_cache=config.model.use_cache,
    )

    print("Loading data...")
    data_loader = DatasetLoader(
        file_path=config.data.train_file,
        data_format=config.data.data_format,
        text_field=config.data.text_field,
        prompt_field=config.data.prompt_field,
        response_field=config.data.response_field,
    )

    datasets = data_loader.to_huggingface_dataset(
        train_split=config.data.train_split,
        validation_split=config.data.validation_split,
    )

    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("validation")

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    print("\nTokenizing data...")
    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
    )

    train_dataset = preprocessor.prepare_dataset(
        train_dataset,
        remove_columns=["text"],
    )

    if eval_dataset:
        eval_dataset = preprocessor.prepare_dataset(
            eval_dataset,
            remove_columns=["text"],
        )

    print("Setting up LoRA...")
    target_modules = factory.get_target_modules(config.lora.target_modules)

    lora_config_dict = config.lora.to_dict()
    lora_config_dict["target_modules"] = target_modules

    from peft import LoraConfig

    lora_config = LoraConfig(**lora_config_dict)

    print("Creating trainer...")
    training_args = create_training_args(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        max_grad_norm=config.training.max_grad_norm,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        seed=config.seed,
    )

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.training.output_dir.split("/")[-1],
        )
        training_args.report_to = "wandb"

    trainer = QLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_config=lora_config,
        training_args=training_args,
    )

    trainer.create_trainer(
        max_seq_length=config.data.max_seq_length,
        dataset_text_field="text",
        neftune_noise_alpha=config.training.neftune_noise_alpha,
    )

    print("\nTraining...")
    trainer.train()

    print("\nSaving model...")
    trainer.save_model()

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model export script for QLoRA models.

Exports LoRA adapters to merged model or HF format.
"""

import argparse
from pathlib import Path


def merge_and_export(
    adapter_path: str,
    output_path: str,
    base_model: str,
) -> None:
    """Merge LoRA adapters and export.

    Args:
        adapter_path: Path to LoRA adapters
        output_path: Output path
        base_model: Base model name or path
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapters...")
    merged_model = model.merge_and_unload()

    print(f"Saving to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


def export_adapters(
    adapter_path: str,
    output_path: str,
) -> None:
    """Export adapters without merging.

    Args:
        adapter_path: Path to adapters
        output_path: Output path
    """
    from shutil import copytree, ignore_patterns

    print(f"Copying adapters from {adapter_path} to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    copytree(adapter_path, output_path, ignore=ignore_patterns("*.safetensors"))

    print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export QLoRA model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapters",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for exported model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model (required for merge)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adapters with base model",
    )

    args = parser.parse_args()

    if args.merge:
        if not args.base_model:
            parser.error("--base_model is required when using --merge")
        merge_and_export(args.adapter_path, args.output_path, args.base_model)
    else:
        export_adapters(args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()

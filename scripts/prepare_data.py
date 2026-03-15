#!/usr/bin/env python3
"""
Data preparation script for QLoRA fine-tuning.

Converts raw JSON/CSV data to processed format suitable for training.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def process_json(input_path: Path, output_path: Path) -> None:
    """Process JSON data file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(data)} examples from {input_path}")
    print(f"Saved to {output_path}")


def process_csv(input_path: Path, output_path: Path, prompt_col: str, response_col: str) -> None:
    """Process CSV data file.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSON file
        prompt_col: Column name for prompt
        response_col: Column name for response
    """
    df = pd.read_csv(input_path)

    if prompt_col not in df.columns or response_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{prompt_col}' and '{response_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    data = []
    for _, row in df.iterrows():
        item = {
            "prompt": str(row[prompt_col]),
            "response": str(row[response_col]),
        }
        data.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(data)} examples from {input_path}")
    print(f"Saved to {output_path}")


def create_sample_data(output_path: Path, num_samples: int = 100) -> None:
    """Create sample training data.

    Args:
        output_path: Path to output JSON file
        num_samples: Number of samples to create
    """
    sample_data = []

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
    ]

    responses = [
        "The capital of France is Paris, known for the Eiffel Tower and rich culture.",
        "Quantum computing uses quantum mechanical phenomena to perform computation. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "Regular exercise improves cardiovascular health, strengthens muscles, boosts mood, helps maintain healthy weight, and reduces risk of chronic diseases like diabetes and heart disease.",
        "The water cycle describes how water evaporates from oceans and lakes, forms clouds, falls as precipitation, and returns to bodies of water. This continuous process is driven by solar energy.",
    ]

    for i in range(num_samples):
        idx = i % len(prompts)
        sample_data.append({
            "prompt": prompts[idx],
            "response": responses[idx],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"Created {num_samples} sample examples in {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare data for QLoRA fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Input file format (default: json)",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name for prompt (CSV only)",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="response",
        help="Column name for response (CSV only)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data instead of processing input",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of sample examples to create",
    )

    args = parser.parse_args()

    if args.create_sample:
        output_path = Path(args.output) if args.output else Path("data/processed/train.json")
        create_sample_data(output_path, args.num_samples)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --create-sample is used")

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        process_json(input_path, output_path)
    elif args.format == "csv":
        process_csv(input_path, output_path, args.prompt_column, args.response_column)
    else:
        raise ValueError(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()

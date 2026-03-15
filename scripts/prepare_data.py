#!/usr/bin/env python3
"""
Data preparation script for QLoRA fine-tuning.

Converts raw JSON/CSV data to processed format suitable for training.
Also provides built-in sample datasets for quick start.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SAMPLE_DATASETS = {
    "instruction": "data/samples/instruction_following.json",
    "qa": "data/samples/qa.json",
    "code": "data/samples/code_generation.json",
    "tech_qa": "data/samples/tech_qa.csv",
}

HF_DATASETS = {
    "alpaca": {
        "hub_id": "yahma/alpaca-cleaned",
        "transform": "alpaca",
    },
    "dolly": {
        "hub_id": "databricks/databricks-dolly-15k",
        "transform": "dolly",
    },
    "guanaco": {
        "hub_id": "OpenAssistant/oasst1",
        "transform": "guanaco",
    },
}


def list_datasets() -> None:
    """List available built-in datasets."""
    print("Available built-in datasets:")
    for name, path in SAMPLE_DATASETS.items():
        print(f"  {name}: {path}")
    print("\nAvailable HuggingFace datasets:")
    for name, config in HF_DATASETS.items():
        print(f"  {name}: {config['hub_id']}")


def copy_builtin_dataset(dataset_name: str, output_path: Path) -> None:
    """Copy a built-in dataset to the output path.

    Args:
        dataset_name: Name of the built-in dataset
        output_path: Destination path
    """
    if dataset_name not in SAMPLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(SAMPLE_DATASETS.keys())}"
        )

    source_path = Path(SAMPLE_DATASETS[dataset_name])

    if not source_path.exists():
        raise FileNotFoundError(f"Built-in dataset not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(source_path, output_path)

    print(f"Copied dataset '{dataset_name}' to {output_path}")


def transform_alpaca(dataset) -> list[dict]:
    """Transform Alpaca format to {prompt, response} format."""
    result = []
    for item in dataset:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}"
        else:
            prompt = f"Instruction: {instruction}"
        result.append({"prompt": prompt, "response": output})
    return result


def transform_dolly(dataset) -> list[dict]:
    """Transform Dolly format to {prompt, response} format."""
    result = []
    for item in dataset:
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        result.append({"prompt": instruction, "response": response})
    return result


def transform_guanaco(dataset) -> list[dict]:
    """Transform Guanaco (OpenAssistant/oasst1) format to {prompt, response} format."""
    result = []
    for item in dataset:
        conversations = item.get("conversations", [])
        prompt_parts = []
        response_text = ""
        for msg in conversations:
            if msg.get("from") == "human":
                prompt_parts.append(msg.get("value", ""))
            elif msg.get("from") == "gpt":
                response_text = msg.get("value", "")
        if prompt_parts and response_text:
            result.append({"prompt": "\n".join(prompt_parts), "response": response_text})
    return result


def download_hf_dataset(dataset_name: str, output_path: Path, limit: int | None = None) -> None:
    """Download and transform a HuggingFace dataset.

    Args:
        dataset_name: Name of the HF dataset (alpaca, dolly, guanaco)
        output_path: Destination path
        limit: Optional limit on number of examples to download
    """
    if dataset_name not in HF_DATASETS:
        raise ValueError(
            f"Unknown HuggingFace dataset '{dataset_name}'. "
            f"Available: {list(HF_DATASETS.keys())}"
        )

    config = HF_DATASETS[dataset_name]
    hub_id = config["hub_id"]
    transform_type = config["transform"]

    print(f"Downloading dataset '{dataset_name}' from HuggingFace Hub: {hub_id}")
    dataset = load_dataset(hub_id, split="train")
    print(f"Downloaded {len(dataset)} examples")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limiting to {len(dataset)} examples")

    transform_funcs = {
        "alpaca": transform_alpaca,
        "dolly": transform_dolly,
        "guanaco": transform_guanaco,
    }

    transform_func = transform_funcs[transform_type]
    data = transform_func(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} examples to {output_path}")


def process_json(input_path: Path, output_path: Path) -> None:
    """Process JSON data file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    """
    with open(input_path, encoding="utf-8") as f:
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
        "--list-datasets",
        action="store_true",
        help="List available built-in datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(SAMPLE_DATASETS.keys()) + list(HF_DATASETS.keys()),
        help="Use a built-in dataset or download from HuggingFace",
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples to download (for HuggingFace datasets)",
    )

    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    if args.dataset:
        output_path = Path(args.output) if args.output else Path(f"data/processed/{args.dataset}.json")
        if args.dataset in HF_DATASETS:
            download_hf_dataset(args.dataset, output_path, args.limit)
        else:
            copy_builtin_dataset(args.dataset, output_path)
        return

    if args.create_sample:
        output_path = Path(args.output) if args.output else Path("data/processed/train.json")
        create_sample_data(output_path, args.num_samples)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --create-sample, --dataset, or --list-datasets is used")

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

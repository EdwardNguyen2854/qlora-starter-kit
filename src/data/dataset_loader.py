"""
Data loading module for QLoRA fine-tuning.

Supports JSON and CSV formats with prompt-response pairs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Load and preprocess datasets for QLoRA fine-tuning."""

    def __init__(
        self,
        file_path: str,
        data_format: Literal["json", "csv"] = "json",
        text_field: str = "text",
        prompt_field: str = "prompt",
        response_field: str = "response",
    ):
        """Initialize DatasetLoader.

        Args:
            file_path: Path to data file
            data_format: Format of data file (json or csv)
            text_field: Field name for full text (for JSON with single field)
            prompt_field: Field name for prompt
            response_field: Field name for response
        """
        self.file_path = Path(file_path)
        self.data_format = data_format
        self.text_field = text_field
        self.prompt_field = prompt_field
        self.response_field = response_field

    def load(self) -> List[Dict[str, str]]:
        """Load data from file.

        Returns:
            List of dictionaries with 'text' key
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        if self.data_format == "json":
            return self._load_json()
        elif self.data_format == "csv":
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

    def _load_json(self) -> List[Dict[str, str]]:
        """Load data from JSON file.

        Expected formats:
        1. [{"prompt": "...", "response": "..."}]
        2. [{"text": "..."}]
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of objects")

        processed_data = []
        for item in data:
            if self.text_field in item:
                processed_data.append({"text": item[self.text_field]})
            elif self.prompt_field in item and self.response_field in item:
                text = f"{item[self.prompt_field]}\n{item[self.response_field]}"
                processed_data.append({"text": text})
            else:
                raise ValueError(
                    f"Item must contain either '{self.text_field}' or "
                    f"'{self.prompt_field}' and '{self.response_field}'"
                )

        return processed_data

    def _load_csv(self) -> List[Dict[str, str]]:
        """Load data from CSV file.

        Expected columns: prompt, response (or text)
        """
        df = pd.read_csv(self.file_path)

        processed_data = []
        for _, row in df.iterrows():
            if self.text_field in row:
                processed_data.append({"text": str(row[self.text_field])})
            elif self.prompt_field in df.columns and self.response_field in df.columns:
                text = f"{row[self.prompt_field]}\n{row[self.response_field]}"
                processed_data.append({"text": text})
            else:
                raise ValueError(
                    f"CSV must contain either '{self.text_field}' column or "
                    f"'{self.prompt_field}' and '{self.response_field}' columns"
                )

        return processed_data

    def to_huggingface_dataset(
        self,
        train_split: float = 0.9,
        validation_split: float = 0.1,
    ) -> Dict[str, Dataset]:
        """Convert to HuggingFace Dataset with train/val split.

        Args:
            train_split: Fraction for training
            validation_split: Fraction for validation

        Returns:
            Dictionary with 'train' and 'validation' Dataset objects
        """
        data = self.load()

        if train_split + validation_split > 1.0:
            raise ValueError("train_split + validation_split must be <= 1.0")

        if validation_split > 0:
            train_data, val_data = train_test_split(
                data,
                train_size=train_split,
                random_state=42,
            )
            return {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
            }
        else:
            return {"train": Dataset.from_list(data)}


def load_dataset_from_file(
    file_path: str,
    data_format: Literal["json", "csv"] = "json",
    text_field: str = "text",
    prompt_field: str = "prompt",
    response_field: str = "response",
    train_split: float = 0.9,
    validation_split: float = 0.1,
) -> Dict[str, Dataset]:
    """Convenience function to load dataset from file.

    Args:
        file_path: Path to data file
        data_format: Format of data file
        text_field: Field name for text
        prompt_field: Field name for prompt
        response_field: Field name for response
        train_split: Fraction for training
        validation_split: Fraction for validation

    Returns:
        Dictionary with 'train' and optionally 'validation' Dataset
    """
    loader = DatasetLoader(
        file_path=file_path,
        data_format=data_format,
        text_field=text_field,
        prompt_field=prompt_field,
        response_field=response_field,
    )
    return loader.to_huggingface_dataset(
        train_split=train_split,
        validation_split=validation_split,
    )

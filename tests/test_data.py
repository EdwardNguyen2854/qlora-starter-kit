"""Tests for data loading module."""

import json
import tempfile
from pathlib import Path

import pytest


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    @pytest.fixture
    def json_file(self, tmp_path):
        data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is the capital of France?", "response": "Paris"},
        ]
        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return str(file_path)

    @pytest.fixture
    def csv_file(self, tmp_path):
        import pandas as pd

        data = pd.DataFrame({
            "prompt": ["What is 2+2?", "What is 2+3?"],
            "response": ["4", "5"],
        })
        file_path = tmp_path / "test.csv"
        data.to_csv(file_path, index=False)
        return str(file_path)

    @pytest.fixture
    def text_file(self, tmp_path):
        data = [{"text": "This is a test text."}, {"text": "Another text."}]
        file_path = tmp_path / "test_text.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return str(file_path)

    def test_load_json_prompt_response(self, json_file):
        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path=json_file,
            data_format="json",
            prompt_field="prompt",
            response_field="response",
        )
        data = loader.load()

        assert len(data) == 2
        assert "text" in data[0]
        assert "What is 2+2?" in data[0]["text"]

    def test_load_json_text_field(self, text_file):
        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path=text_file,
            data_format="json",
            text_field="text",
        )
        data = loader.load()

        assert len(data) == 2
        assert data[0]["text"] == "This is a test text."

    def test_load_csv(self, csv_file):
        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path=csv_file,
            data_format="csv",
            prompt_field="prompt",
            response_field="response",
        )
        data = loader.load()

        assert len(data) == 2
        assert "text" in data[0]

    def test_to_huggingface_dataset(self, json_file):
        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path=json_file,
            data_format="json",
        )
        datasets = loader.to_huggingface_dataset(
            train_split=0.8,
            validation_split=0.2,
        )

        assert "train" in datasets
        assert "validation" in datasets
        assert len(datasets["train"]) >= 1
        assert len(datasets["validation"]) >= 1

    def test_file_not_found(self):
        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path="nonexistent.json",
            data_format="json",
        )

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_invalid_format(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("invalid content")

        from src.data import DatasetLoader

        loader = DatasetLoader(
            file_path=str(file_path),
            data_format="json",
        )

        with pytest.raises(json.JSONDecodeError):
            loader.load()


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    def test_import_preprocessor(self):
        from src.data import DataPreprocessor, prepare_dataset

        assert DataPreprocessor is not None
        assert prepare_dataset is not None

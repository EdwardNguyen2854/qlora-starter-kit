"""
Data preprocessing module for QLoRA fine-tuning.

Handles tokenization and formatting for language model training.
"""

from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


class DataPreprocessor:
    """Preprocess and tokenize datasets for QLoRA training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        formatting_func: Optional[Callable[[Dict[str, str]], str]] = None,
    ):
        """Initialize DataPreprocessor.

        Args:
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            formatting_func: Optional function to format data samples
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.formatting_func = formatting_func

    def preprocess(self, examples: Dict[str, List[Any]]) -> Dict[str, List]:
        """Tokenize examples.

        Args:
            examples: Batch of examples from dataset

        Returns:
            Tokenized examples
        """
        if self.formatting_func:
            texts = [self.formatting_func({"text": t}) for t in examples["text"]]
        else:
            texts = examples["text"]

        tokenized = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(
        self,
        dataset: Dataset,
        batched: bool = True,
        batch_size: int = 1000,
        remove_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """Prepare dataset with tokenization.

        Args:
            dataset: HuggingFace Dataset
            batched: Process in batches
            batch_size: Batch size for processing
            remove_columns: Columns to remove after processing

        Returns:
            Tokenized dataset
        """
        if remove_columns is None:
            remove_columns = [col for col in dataset.column_names if col != "text"]

        return dataset.map(
            self.preprocess,
            batched=batched,
            batch_size=batch_size,
            remove_columns=remove_columns,
            desc="Tokenizing dataset",
        )


def create_chat_formatting_func(
    tokenizer: PreTrainedTokenizer,
    system_message: str = "You are a helpful assistant.",
) -> Callable[[Dict[str, str]], str]:
    """Create a chat formatting function for conversation data.

    Args:
        tokenizer: Tokenizer with chat template support
        system_message: System message to use

    Returns:
        Formatting function
    """
    def format_chat(example: Dict[str, str]) -> str:
        text = example.get("text", "")
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return text

    return format_chat


def create_prompt_response_formatting_func(
    prompt_field: str = "prompt",
    response_field: str = "response",
) -> Callable[[Dict[str, str]], str]:
    """Create a formatting function for prompt-response pairs.

    Args:
        prompt_field: Field name for prompt
        response_field: Field name for response

    Returns:
        Formatting function
    """
    def format_prompt_response(example: Dict[str, str]) -> str:
        prompt = example.get(prompt_field, "")
        response = example.get(response_field, "")
        return f"{prompt}\n{response}"

    return format_prompt_response


def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    data_format: str = "text",
    prompt_field: str = "prompt",
    response_field: str = "response",
) -> Dataset:
    """Convenience function to prepare dataset for training.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        data_format: Format type - "text", "chat", or "prompt_response"
        prompt_field: Field name for prompt
        response_field: Field name for response

    Returns:
        Prepared dataset
    """
    if data_format == "chat":
        formatting_func = create_chat_formatting_func(tokenizer)
    elif data_format == "prompt_response":
        formatting_func = create_prompt_response_formatting_func(
            prompt_field, response_field
        )
    else:
        formatting_func = None

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,
    )

    return preprocessor.prepare_dataset(dataset)

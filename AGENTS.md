# QLoRA Starter Kit - Agent Guidelines

A beginner-friendly QLoRA fine-tuning starter kit for consumer hardware (6-8GB VRAM). Supports Llama, Qwen, and Gemma models with JSON/CSV data formats.

## Development Commands

### Testing
```bash
pytest                           # Run all tests
pytest -v                        # Verbose output
pytest tests/test_training.py   # Single test file
pytest tests/test_training.py::test_trainer_init  # Single test function
pytest -k "test_qlora"          # Tests matching pattern
pytest --cov=src --cov-report=html  # With coverage
```

### Linting & Formatting
```bash
ruff check .              # Lint
ruff check --fix .        # Auto-fix issues
black .                   # Format
mypy .                    # Type check
ruff check . && black --check . && mypy .  # All checks
```

### Pre-commit
```bash
pre-commit install        # Install hooks
pre-commit run --all-files  # Run on all files
```

### Running
```bash
python train.py --config configs/qlora_config.yaml
python train.py --config configs/qwen_config.yaml    # Qwen model
python train.py --config configs/gemma_config.yaml   # Gemma model
python train.py --create_sample_data                  # Create sample data first
python inference.py --model_path ./output/model --prompt "Your prompt"
python inference.py --model_path ./output/model --interactive
python scripts/prepare_data.py --input data/raw --output data/processed
python scripts/export_model.py --adapter_path ./output/adapters --merge --base_model meta-llama/Llama-2-7b-hf
```

## Code Style

### Imports (order: stdlib, third-party, local)
```python
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

from src.config import QLoRAConfig
from src.data import DatasetLoader
from src.models import QuantizationConfig
```
- Use absolute imports, no wildcard imports
- Sort alphabetically within groups (use ruff/isort)
- Keep type hint imports separate from regular imports

### Formatting
- Max line length: 100 chars
- 4 spaces indentation (no tabs)
- Use Black for formatting
- Trailing commas in multi-line constructs
- f-strings for string formatting

```python
# Good
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

# Good - long strings
error_msg = (
    f"Could not load model from {model_path}. "
    f"Please check the path and ensure model is downloaded."
)
```

### Type Hints
Always use type hints. Use `Optional[T]` for Python 3.10 compatibility:
```python
def prepare_dataset(
    path: str,
    max_length: int = 512,
    split: str = "train"
) -> "Dataset":
    """Prepare dataset for training.

    Args:
        path: Path to dataset file
        max_length: Maximum sequence length
        split: Dataset split (train/val)

    Returns:
        Prepared dataset

    Raises:
        FileNotFoundError: If dataset file not found
    """
    ...
```
Use string literals for forward references (`"Dataset"` not `Dataset`).

### Naming
- Variables/functions: `snake_case` (e.g., `learning_rate`, `max_seq_length`)
- Classes: `PascalCase` (e.g., `QLoRAConfig`, `DatasetLoader`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQ_LENGTH`, `DEFAULT_RANK`)
- Private methods: prefix with `_` (e.g., `_init_model()`, `_load_data()`)
- Files: `snake_case.py` (e.g., `dataset_loader.py`, `quantization.py`)

### Error Handling
Use specific exceptions with meaningful messages:
```python
try:
    model = AutoModel.from_pretrained(model_path)
except OSError as e:
    raise ValueError(
        f"Could not load model from {model_path}. "
        f"Please check the path and ensure model is downloaded."
    ) from e

# Always validate configuration
def validate(self) -> List[str]:
    errors = []
    if self.lora.r < 1:
        errors.append("lora.r must be at least 1")
    return errors
```

## ML Project Conventions

### Configuration
- All configs inherit from base classes in `src/config/__init__.py`
- Use YAML files in `configs/` for model-specific settings
- Validate configs before use with `config.validate()`

### Model Loading Pattern
```python
# Quantized model loading
from src.models import QuantizationConfig, ModelFactory

quant_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

factory = ModelFactory(model_name="meta-llama/Llama-2-7b-hf")
model, tokenizer = factory.load_model_and_tokenizer(quant_config)
```

### LoRA Configuration
- Default rank: 16 (use 8 for 6GB VRAM, 32 for more capacity)
- Always target `q_proj` and `v_proj` minimum
- Full modules: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### Data Formats
JSON format:
```json
[{"prompt": "Question?", "response": "Answer."}]
```
CSV format: columns `prompt`, `response`

### Training Best Practices
```python
# Always set seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Use gradient checkpointing for memory savings
training_args = TrainingArguments(
    gradient_checkpointing=True,
    bf16=True,  # or fp16
    optim="paged_adamw_32bit",
)
```

## Testing
- Descriptive names: `test_qlora_config_valid_parameters()`
- Use fixtures for common setup
- Mock external dependencies (model downloads, API calls)
- Test edge cases and error conditions

```python
@pytest.fixture
def sample_config():
    return {"r": 16, "lora_alpha": 32}

def test_config_creation(sample_config):
    config = QLoRAConfig(**sample_config)
    assert config.r == 16
```

## Documentation
- Google-style docstrings for all public functions/classes
- Document: description, args, returns, raises, examples
- Add type hints to all function signatures

## Project Structure
```
qlora-starter-kit/
├── src/
│   ├── config/          # Configuration classes (QLoRAConfig, etc.)
│   ├── data/            # Dataset loading & preprocessing
│   ├── models/          # Model loading, quantization, LoRA
│   ├── training/        # Training pipeline (SFTTrainer)
│   └── inference/       # Inference system
├── scripts/             # Utility scripts
├── configs/             # YAML configs for different models
├── tests/               # Test suite
├── docs/                # Educational docs (quantization, LoRA, QLoRA)
├── data/                # Data directory (raw/, processed/)
├── output/              # Model outputs
└── notebooks/           # Jupyter notebooks
```

## Common Issues

### CUDA OOM
- Use 4-bit quantization (bitsandbytes)
- Reduce batch size to 1, increase gradient accumulation
- Enable gradient checkpointing: `gradient_checkpointing=True`
- Call `torch.cuda.empty_cache()` periodically

### Slow Training
- Enable mixed precision: `bf16=True` or `fp16=True`
- Enable gradient accumulation for effective larger batches
- Use SSD for data loading
- Monitor GPU with `nvidia-smi`

### Model Not Loading
- Verify HuggingFace token in `.env` for gated models
- Check disk space for model downloads
- Ensure correct model path/Hub ID
- Set `trust_remote_code=True` for models that need it

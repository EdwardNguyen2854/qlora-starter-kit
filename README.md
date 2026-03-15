# QLoRA Starter Kit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FEdwardNguyen2854%2Fqlora-starter-kit%2Fmain%2Fpyproject.toml&query=%24.project.version&label=version&color=2ea44f)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A beginner-friendly QLoRA fine-tuning starter kit for consumer hardware (6-8GB VRAM).

## What is QLoRA?

QLoRA combines **quantization** (4-bit NF4) with **LoRA** (Low-Rank Adaptation) to enable fine-tuning of large language models on limited GPU memory:

| Method | GPU Memory | Trainable Params |
|--------|------------|------------------|
| Full Fine-tuning (7B) | ~56 GB | 100% |
| QLoRA (7B) | 6-8 GB | ~0.1% |

## Supported Models

- Llama 2 / Llama 3
- Qwen 2
- Gemma

## Installation

```bash
git clone https://github.com/EdwardNguyen2854/qlora-starter-kit.git
cd qlora-starter-kit
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your HuggingFace token if needed
```

## Quick Start

### 1. Prepare Data

```bash
# List available datasets (built-in and HuggingFace)
python scripts/prepare_data.py --list-datasets

# Use a built-in dataset (instruction, qa, code, tech_qa)
python scripts/prepare_data.py --dataset qa --output data/processed/train.json

# Download from HuggingFace (alpaca, dolly, guanaco)
python scripts/prepare_data.py --dataset alpaca --output data/processed/train.json
python scripts/prepare_data.py --dataset dolly --output data/processed/train.json
python scripts/prepare_data.py --dataset guanaco --output data/processed/train.json

# Create sample data
python scripts/prepare_data.py --create-sample

# Convert your own data (CSV)
python scripts/prepare_data.py \
    --input data/my_data.csv \
    --output data/processed/train.json \
    --format csv \
    --prompt-column prompt \
    --response-column response
```

**Built-in Datasets:**

| Dataset | Description |
|---------|-------------|
| `instruction` | General knowledge instruction following (10 examples) |
| `qa` | Q&A on various topics (10 examples) |
| `code` | Python code generation tasks (10 examples) |
| `tech_qa` | Technical Q&A (CSV format, 10 examples) |

**HuggingFace Datasets:**

| Dataset | Source | Size | Disk Space |
|---------|--------|------|------------|
| `alpaca` | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | ~52K | ~50 MB |
| `dolly` | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15K | ~15 MB |
| `guanaco` | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~100K+ | ~500 MB |

- **alpaca**: Cleaned version of Stanford Alpaca dataset with instruction-following examples
- **dolly**: Databricks Dolly dataset with human-generated instruction-response pairs
- **guanaco**: OpenAssistant conversation data formatted for assistant training

> **Note:** Large datasets (alpaca, guanaco) require significant disk space and may take time to download. For testing, use a subset: `python scripts/prepare_data.py --dataset alpaca --limit 1000`.

Data format (JSON):
```json
[
    {"prompt": "What is the capital of France?", "response": "Paris"},
    {"prompt": "Explain quantum computing", "response": "Quantum computing uses..."}
]
```

### 2. Train

```bash
python train.py --config configs/qlora_config.yaml
```

With custom settings:
```bash
python train.py \
    --config configs/qlora_config.yaml \
    --num_epochs 5 \
    --output_dir ./output/my_model
```

### 3. Run Inference

```bash
# Single prompt
python inference.py \
    --model_path ./output/model \
    --prompt "What is machine learning?"

# Interactive mode
python inference.py --model_path ./output/model --interactive
```

## Configuration

### Model

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model ID | meta-llama/Llama-2-7b-hf |
| `model_family` | llama / qwen / gemma | auto-detect |

### Quantization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `load_in_4bit` | Load model in 4-bit | true |
| `bnb_4bit_quant_type` | nf4 or fp4 | nf4 |
| `bnb_4bit_use_double_quant` | Double quantization | true |

### LoRA

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r` | LoRA rank | 16 |
| `lora_alpha` | Scaling factor | 32 |
| `target_modules` | Layers for LoRA | q_proj, v_proj |
| `lora_dropout` | Dropout probability | 0.05 |

### Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `per_device_train_batch_size` | Batch size | 1 |
| `gradient_accumulation_steps` | Accumulation steps | 4 |
| `learning_rate` | Learning rate | 2e-4 |
| `num_train_epochs` | Epochs | 3 |
| `gradient_checkpointing` | Save memory | true |
| `bf16` | Use bfloat16 | true |

## GPU Requirements

| Model | VRAM |
|-------|------|
| 7B models (Llama, Qwen, Gemma) | 6-8 GB |
| 13B models | 12-16 GB |

## Project Structure

```
qlora-starter-kit/
├── src/
│   ├── config/       # Configuration classes
│   ├── data/         # Data loading & preprocessing
│   ├── models/       # Model loading & quantization
│   ├── training/     # Training pipeline
│   └── inference/   # Inference system
├── scripts/          # Utility scripts
├── configs/          # YAML configs
├── tests/            # Test suite
├── docs/             # Educational documentation
└── README.md
```

## Documentation

- [What is Quantization?](docs/01_quantization.md)
- [What is LoRA?](docs/02_lora.md)
- [Why QLoRA?](docs/03_why_qlora.md)

## Troubleshooting

**CUDA Out of Memory**
- Set `per_device_train_batch_size: 1`
- Enable `gradient_checkpointing: true`
- Reduce LoRA rank: `r: 8`

**Model Not Loading**
- Verify HuggingFace token in `.env` for gated models
- Check disk space for model downloads

**Slow Training**
- Enable `bf16: true` (requires Ampere+ GPU)
- Increase `gradient_accumulation_steps`

## License

MIT

## Acknowledgments

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)

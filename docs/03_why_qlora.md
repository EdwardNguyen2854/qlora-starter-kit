# Why QLoRA?

## Overview

QLoRA combines two powerful techniques: **Quantization** and **LoRA**. This combination allows fine-tuning of large language models on consumer hardware that would otherwise be impossible to use.

## The Revolution

Before QLoRA:
```
Fine-tuning a 7B model:
- Required: 24+ GB VRAM (A100)
- Desktop users: Can't participate

Fine-tuning a 70B model:
- Required: Multiple A100s (80GB each)
- Cost: $1000s in cloud computing
```

With QLoRA:
```
Fine-tuning a 7B model:
- Required: 6-8 GB VRAM (consumer GPU)
- Cost: Free on your own PC

Fine-tuning a 70B model:
- Required: 24 GB VRAM (single A100)
- Cost: Affordable cloud instance
```

## How QLoRA Works

```
Traditional Fine-tuning:
[Model: FP32] → [Train All Parameters] → [Save: FP32]
                     ↓
              Memory: ~56 GB (7B model)

QLoRA:
[Model: 4-bit] → [LoRA Adapters] → [Save: LoRA + Original 4-bit]
                     ↓
              Memory: ~6 GB (7B model)
```

## The QLoRA Recipe

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Step 1: Quantize the model (4-bit NF4)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)

# Step 2: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Step 3: Fine-tune (only LoRA params require gradients!)
# Memory: ~6-8 GB instead of ~56 GB
```

## Memory Breakdown

For a 7B parameter model with QLoRA (r=16):

| Component | Full Fine-tuning | QLoRA |
|-----------|-----------------|-------|
| Model weights (4-bit) | - | 3.5 GB |
| Model weights (unpacked) | - | 0.5 GB |
| LoRA (A, B matrices) | - | 0.1 GB |
| LoRA gradients | - | 0.1 GB |
| LoRA optimizer | - | 0.2 GB |
| Activations | 14 GB | 2-4 GB |
| **Total** | **~56 GB** | **~6-8 GB** |

## Why This Works

### 1. 4-bit Quantization
- Weights stored as 4-bit values
- Only unpacked to FP16 when needed
- Uses NF4 format for minimal quality loss

### 2. LoRA Efficiency
- Only ~0.1% of parameters are trainable
- LoRA matrices are small (r × (d+k))
- Backpropagation only through adapters

### 3. Gradient Checkpointing
- Trade compute for memory
- Re-compute activations during backward pass
- Saves ~30% memory

### 4. Mixed Precision
- Forward pass: FP16
- Quantization: INT4
- LoRA computation: FP16

## Performance Comparison

QLoRA matches full fine-tuning quality in most cases:

| Method | GPU Memory | Training Time | Quality |
|--------|------------|---------------|---------|
| Full FT (7B) | 56 GB | 1x | 100% |
| LoRA (7B) | 15 GB | 1.1x | 98% |
| **QLoRA (7B)** | **6 GB** | **1.2x** | **97%** |
| Full FT (70B) | 450 GB | 1x | 100% |
| **QLoRA (70B)** | **24 GB** | **1.3x** | **97%** |

## Supported Models

QLoRA works with many transformer models:

| Model Family | Examples | Max VRAM |
|--------------|----------|----------|
| Llama 2/3 | 7B, 13B, 70B | 6-8 GB (7B) |
| Qwen | 1.8B, 7B, 14B | 4-8 GB |
| Gemma | 2B, 7B | 6-8 GB |

## Best Practices

1. **Start with r=16** - Good balance of quality and memory
2. **Use double quantization** - Saves ~0.4 bits with no quality loss
3. **Target q_proj and v_proj** - Minimum effective configuration
4. **Use gradient checkpointing** - Essential for 6GB VRAM
5. **Batch size of 1** with gradient accumulation - Reduces memory

## Troubleshooting

### If you have 6GB VRAM:
- Use r=8
- Target only q_proj, v_proj
- Enable gradient checkpointing
- Use batch_size=1

### If you have 8GB VRAM:
- Use r=16
- Target q_proj, v_proj, k_proj, o_proj
- Enable gradient checkpointing
- Use batch_size=2

## Further Reading

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Guanaco: QLoRA fork](https://github.com/artbohr/LoRA-fine-tune-Guanaco)
- [Original QLoRA GitHub](https://github.com/TimDettmers/bitsandbytes)

## Quick Start

```bash
# Install
pip install qlora-starter-kit

# Prepare data
python scripts/prepare_data.py --input data/my_data.json --output data/processed

# Train
python train.py --config configs/qlora_config.yaml

# Inference
python inference.py --model_path ./output/model --prompt "Your prompt"
```

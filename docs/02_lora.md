# What is LoRA?

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small "adapter" layers to a pre-trained model instead of updating all its weights. This allows you to fine-tune large models with a fraction of the memory and compute requirements.

## The Problem with Full Fine-tuning

When fine-tuning a large language model traditionally:

```
Full Fine-tuning:
================

Pre-trained Model (7B params)
        ↓
   Gradient Update
        ↓
Fine-tuned Model (7B params changed)
        ↓
   Save ALL 7B params

Memory needed:
- Model weights: ~14 GB (FP16)
- Optimizer states: ~28 GB (Adam)
- Gradients: ~14 GB
- Total: ~56 GB!
```

## How LoRA Works

LoRA adds small trainable matrices alongside the original model weights:

```
LoRA Architecture:
==================

    Input → [Pre-trained Weight] → Output
              ↓
         + [LoRA Adapter]
           (rank r << original dim)

Original weight W: (d × k)
LoRA matrices: A (d × r), B (r × k)

Output = W × Input + B × A × Input
         ↑ original    ↑ low-rank adjustment
```

## Key Parameters

### Rank (r)
- The rank of the low-rank matrices
- Typical values: 8, 16, 32, 64
- Higher = more capacity, more memory
- Lower = more compressed, less capacity

### Alpha (lora_alpha)
- Scaling factor for the LoRA contribution
- Usually set to 2× the rank
- Formula: `output = W × x + (alpha/r) × B×A × x`

### Target Modules
Which layers to apply LoRA to:

| Model Family | Typical Targets |
|--------------|-----------------|
| Llama | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Qwen | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Gemma | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |

## Memory Comparison

```
Full Fine-tuning (7B model):
- Model: 14 GB
- Optimizer: 28 GB  
- Gradients: 14 GB
- Total: ~56 GB

LoRA Fine-tuning (7B model, r=16):
- Model (frozen): 14 GB
- LoRA params: ~0.1 GB
- Optimizer (LoRA only): ~0.2 GB
- Gradients (LoRA only): ~0.1 GB
- Total: ~14-15 GB
```

## Training with LoRA

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,       # Regularization
    bias="none",             # Bias handling
    task_type="CAUSAL_LM",   # Language modeling
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06
```

## Saving and Loading Adapters

```python
# Save only LoRA weights
model.save_pretrained("output/adapters")

# Load adapters later
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "output/adapters")

# Merge and save (for inference)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("output/merged_model")
```

## Choosing Rank (r)

| Rank | Parameters | Capacity | Use Case |
|------|------------|----------|----------|
| 8 | ~0.05M | Basic | Quick experiments |
| 16 | ~0.1M | Standard | Most tasks |
| 32 | ~0.2M | High | Complex tasks |
| 64 | ~0.4M | Very High | Niche tasks |

## When to Use LoRA

**LoRA is great when:**
- You have limited GPU memory (6-8GB)
- You want to fine-tune multiple tasks (each adapter is small)
- You need quick experimentation
- You want to preserve the base model

**Full fine-tuning might be better when:**
- You have ample GPU memory (24GB+)
- You need maximum performance
- You're training on a large dataset

## Further Reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT library documentation](https://huggingface.co/docs/peft)
- [Understanding LoRA](https://magazine.sebastianraschka.com/p/understanding-lora)

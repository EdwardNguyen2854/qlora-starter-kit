# What is Quantization?

## Overview

Quantization is a technique to reduce the memory footprint and computational requirements of neural networks by representing weights and activations with lower precision data types. Instead of using 32-bit floating point (FP32), we can use 16-bit (FP16), 8-bit (INT8), or even 4-bit (INT4) representations.

## Why Quantize?

Large Language Models (LLMs) like Llama, Qwen, and Gemma contain billions of parameters. A single parameter stored in FP32 takes 4 bytes (32 bits). This means:

| Model Size | FP32 Memory | INT8 Memory | INT4 Memory |
|------------|-------------|-------------|-------------|
| 7B params  | ~28 GB      | ~7 GB       | ~3.5 GB     |
| 13B params | ~52 GB      | ~13 GB      | ~6.5 GB     |
| 70B params | ~280 GB     | ~70 GB      | ~35 GB      |

This makes it impossible to run larger models on consumer GPUs with 6-8GB VRAM.

## Quantization Types

### FP16 (Half Precision)
- 16-bit floating point
- Native GPU support
- ~50% memory reduction
- Minimal quality loss

### INT8 (8-bit Integer)
- 8-bit integer representation
- ~75% memory reduction
- Requires quantization/dequantization
- Some quality loss possible

### NF4 (4-bit Normal Float)
- 4-bit representation optimized for neural network weights
- Uses non-uniform quantization based on weight distribution
- ~87.5% memory reduction
- Used by bitsandbytes for QLoRA

```
How NF4 Works:
==============

Traditional INT4:     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                      ↑ uniform spacing

NF4 quantization:    [-1.0, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1.0, ...]
                      ↑ denser near zero (where most weights are)
```

## bitsandbytes Library

The [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library provides optimized quantization kernels:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
)
```

## Double Quantization

Double quantization (enabled with `bnb_4bit_use_double_quant=True`) quantizes the quantization constants themselves, saving additional memory (~0.4 bits per parameter).

## When to Use Quantization

| Scenario | Recommended Precision |
|----------|----------------------|
| 24GB+ VRAM | FP16 |
| 12-16GB VRAM | INT8 or NF4 |
| 6-8GB VRAM | NF4 (4-bit) |

## Trade-offs

**Pros:**
- Fits larger models in limited VRAM
- Faster inference (fewer memory accesses)
- Lower power consumption

**Cons:**
- Potential quality loss
- Some accuracy degradation
- Requires compatible hardware

## Further Reading

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [bitsandbytes documentation](https://bitsandbytes.readthedocs.io/)

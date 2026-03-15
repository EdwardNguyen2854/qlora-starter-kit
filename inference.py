#!/usr/bin/env python3
"""
Inference script for QLoRA fine-tuned models.

Usage:
    python inference.py --model_path ./output/model --prompt "Your prompt"
    python inference.py --model_path ./output/model --interactive
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run inference with QLoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/model",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming output",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode with multiple prompts",
    )

    args = parser.parse_args()

    from src.inference import load_predictor

    print("Loading model...")
    predictor = load_predictor(args.model_path)

    print("=" * 50)
    print("QLoRA Inference")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 50)

    if args.batch:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate factorial.",
        ]
        print("\nGenerating responses for batch prompts...")
        results = predictor.batch_generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        for prompt, response in zip(prompts, results):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}\n")

    elif args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            prompt = input("\n> ")
            if prompt.lower() == "quit":
                break

            if args.streaming:
                print("\nResponse: ", end="", flush=True)
                for chunk in predictor.generate_streaming(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                ):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = predictor.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                print(f"\nResponse: {response}")

    elif args.prompt:
        if args.streaming:
            print("\nResponse: ", end="", flush=True)
            for chunk in predictor.generate_streaming(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            response = predictor.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print(f"\nResponse: {response}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

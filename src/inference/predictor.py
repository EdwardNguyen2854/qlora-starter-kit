"""
Inference module for QLoRA models.

Provides predictor class for running inference with fine-tuned models.
"""

from typing import Any, Dict, Generator, List, Literal, Optional

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class QLoRAPredictor:
    """Predictor for QLoRA fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
        device: str = "auto",
    ):
        """Initialize QLoRA Predictor.

        Args:
            model_path: Path to model or HuggingFace model ID
            tokenizer: Pre-loaded tokenizer (optional)
            model: Pre-loaded model (optional)
            device: Device to use
        """
        self.model_path = model_path

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            self.model = model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repeat_penalty: float = 1.1,
        **kwargs,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample
            repeat_penalty: Repetition penalty

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repeat_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repeat_penalty: float = 1.1,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample
            repeat_penalty: Repetition penalty

        Yields:
            Generated text chunks
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repeat_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            **kwargs,
        )

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repeat_penalty: float = 1.1,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample
            repeat_penalty: Repetition penalty

        Returns:
            List of generated texts
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repeat_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        results = []
        for i, prompt in enumerate(prompts):
            prompt_length = inputs.input_ids[i].shape[0]
            generated = outputs[i][prompt_length:]
            generated_text = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            results.append(generated_text)

        return results


def load_predictor(
    model_path: str,
    use_peft: bool = True,
    device: str = "auto",
) -> QLoRAPredictor:
    """Load predictor from model path.

    Args:
        model_path: Path to model
        use_peft: Load PEFT adapters
        device: Device to use

    Returns:
        QLoRAPredictor
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_peft:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    return QLoRAPredictor(
        model_path=model_path,
        tokenizer=tokenizer,
        model=model,
    )

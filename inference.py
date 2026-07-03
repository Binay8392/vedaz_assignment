"""Run chat inference with a Qwen base model and a trained PEFT adapter."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompts import VEDAZ_SYSTEM_PROMPT
from utils import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationSettings:
    """Text generation parameters."""

    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    enable_thinking: bool = False


def adapter_base_model(adapter_path: Path) -> str:
    """Read the base model name from a standard PEFT adapter config."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid adapter config {config_path}: {exc}") from exc
    base_model = payload.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(f"base model is missing from {config_path}")
    return base_model


def infer_model_family(model_name: str) -> str:
    """Infer Qwen family from a model identifier."""
    normalized = model_name.lower()
    return "qwen3" if "qwen3" in normalized else "qwen2.5"


class ChatModel:
    """Lazy local chat model with optional PEFT adapter."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: Path | None,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.model_family = infer_model_family(model_name_or_path)
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        """Load tokenizer, base weights, and adapter."""
        if self.model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Inference dependencies are missing; install requirements.txt"
            ) from exc

        if self.load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError("4-bit inference requires CUDA and bitsandbytes")
        dtype_map = {
            "auto": "auto",
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.dtype]
        tokenizer_source = (
            str(self.adapter_path)
            if self.adapter_path
            and (self.adapter_path / "tokenizer_config.json").is_file()
            else self.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        if self.adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_path),
                is_trainable=False,
            )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
    ) -> str:
        """Generate one assistant response."""
        import torch

        self.load()
        template_kwargs: dict[str, Any] = {}
        if self.model_family == "qwen3":
            template_kwargs["enable_thinking"] = settings.enable_thinking
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = settings.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": settings.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output[0, prompt_length:]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Base or merged model path/name.")
    parser.add_argument("--adapter", type=Path, help="PEFT adapter directory.")
    parser.add_argument("--prompt", help="Single prompt; omit for interactive chat.")
    parser.add_argument("--system-prompt", default=VEDAZ_SYSTEM_PROMPT)
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "float32"],
        default="auto",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> str:
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if args.adapter and not args.adapter.is_dir():
        raise ValueError(f"adapter directory does not exist: {args.adapter}")
    model_name = args.model
    if not model_name and args.adapter:
        model_name = adapter_base_model(args.adapter)
    if not model_name:
        raise ValueError("--model or --adapter is required")
    return model_name


def main() -> int:
    """Run one-shot or interactive inference."""
    args = parse_args()
    configure_logging(args.log_level)
    try:
        model_name = _validate_args(args)
        if args.dry_run:
            LOGGER.info(
                "Dry run passed: model=%s adapter=%s",
                model_name,
                args.adapter,
            )
            return 0
        model = ChatModel(
            model_name_or_path=model_name,
            adapter_path=args.adapter,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )
        settings = GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=args.enable_thinking,
        )
        history = [{"role": "system", "content": args.system_prompt}]
        if args.prompt:
            history.append({"role": "user", "content": args.prompt})
            print(model.generate(history, settings))
            return 0

        print("Interactive mode. Enter /quit to exit.")
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"/quit", "/exit"}:
                break
            if not prompt:
                continue
            history.append({"role": "user", "content": prompt})
            response = model.generate(history, settings)
            print(f"Vedaz: {response}")
            history.append({"role": "assistant", "content": response})
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Inference failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

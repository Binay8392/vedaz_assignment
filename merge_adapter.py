"""Merge a PEFT LoRA adapter into its base model for deployment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from inference import adapter_base_model
from utils import configure_logging

LOGGER = logging.getLogger(__name__)


def merge_adapter(
    *,
    adapter_path: Path,
    output_path: Path,
    base_model: str | None,
    dtype: str,
    max_shard_size: str,
    trust_remote_code: bool,
) -> Path:
    """Load an adapter in full precision, merge it, and save safetensors."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Merge dependencies are missing; install requirements.txt"
        ) from exc

    resolved_base = base_model or adapter_base_model(adapter_path)
    dtype_map: dict[str, Any] = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    LOGGER.info("Loading base model %s", resolved_base)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        torch_dtype=dtype_map[dtype],
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload(safe_merge=True)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer_source = (
        str(adapter_path)
        if (adapter_path / "tokenizer_config.json").is_file()
        else resolved_base
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_path)
    LOGGER.info("Merged model saved to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "float32"],
        default="bf16",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Merge an adapter and return a process status."""
    args = parse_args()
    configure_logging(args.log_level)
    try:
        if not args.adapter.is_dir():
            raise ValueError(f"adapter directory does not exist: {args.adapter}")
        if args.output.resolve() == args.adapter.resolve():
            raise ValueError("output directory must differ from adapter directory")
        base_model = args.base_model or adapter_base_model(args.adapter)
        if args.dry_run:
            LOGGER.info(
                "Dry run passed: base=%s adapter=%s output=%s",
                base_model,
                args.adapter,
                args.output,
            )
            return 0
        merge_adapter(
            adapter_path=args.adapter,
            output_path=args.output,
            base_model=base_model,
            dtype=args.dtype,
            max_shard_size=args.max_shard_size,
            trust_remote_code=args.trust_remote_code,
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Adapter merge failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

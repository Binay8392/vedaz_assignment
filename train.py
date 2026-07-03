"""Fine-tune Qwen2.5 or Qwen3 with LoRA using Transformers or Unsloth."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training import (
    apply_overrides,
    config_as_dict,
    load_training_config,
    train_lora,
    validate_training_inputs,
)
from utils import PROJECT_ROOT, configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qwen2.5-lora.yaml",
    )
    parser.add_argument("--model-family", choices=["qwen2.5", "qwen3"])
    parser.add_argument("--model", dest="model_name_or_path")
    parser.add_argument("--backend", choices=["transformers", "unsloth"])
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--batch-size", dest="per_device_train_batch_size", type=int)
    parser.add_argument(
        "--eval-batch-size", dest="per_device_eval_batch_size", type=int
    )
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", dest="num_train_epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help="Resume from the latest checkpoint or from the supplied path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data without loading a model.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run training and return a process status."""
    args = parse_args()
    configure_logging(args.log_level)
    try:
        config = load_training_config(args.config)
        overrides = {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "config",
                "dry_run",
                "log_level",
            }
        }
        if args.fp16 is True and args.bf16 is None:
            overrides["bf16"] = False
        if args.bf16 is True and args.fp16 is None:
            overrides["fp16"] = False
        config = apply_overrides(config, overrides)
        validate_training_inputs(config)
        if args.dry_run:
            LOGGER.info("Dry run passed: %s", config_as_dict(config))
            return 0
        train_lora(config)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build deterministic, leakage-free train and validation JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from dataset_pipeline import prepare_dataset_splits
from utils import (
    DATA_DIR,
    PROJECT_ROOT,
    configure_logging,
    read_dataset,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        help="Input JSON/JSONL path. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DATA_DIR / "train.jsonl",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DATA_DIR / "validation.jsonl",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--exclude-ungrounded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude asserted chart calculations without provenance.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DATA_DIR / "dataset_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Prepare dataset splits and return a process status."""
    args = parse_args()
    configure_logging(args.log_level)
    input_paths = args.inputs or [
        DATA_DIR / "vedaz_astrologer_finetune.jsonl",
        DATA_DIR / "generated_chats.jsonl",
        DATA_DIR / "additional_conversations.jsonl",
    ]
    try:
        split = prepare_dataset_splits(
            input_paths,
            train_path=args.train_output,
            validation_path=args.validation_output,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            exclude_ungrounded=args.exclude_ungrounded,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 1
    LOGGER.info(
        "Wrote %s records to %s and %s records to %s; filtered %s",
        split.train_records,
        split.train_path,
        split.validation_records,
        split.validation_path,
        split.filtered_ungrounded_records,
    )
    manifest = {
        "format_version": 1,
        "sources": [
            {
                "path": _display_path(path),
                "records": len(read_dataset(path)),
                "sha256": _file_hash(path),
            }
            for path in input_paths
        ],
        "preparation": {
            "seed": args.seed,
            "validation_ratio": args.validation_ratio,
            "exclude_ungrounded": args.exclude_ungrounded,
            "source_records": split.source_records,
            "eligible_unique_records": split.unique_records,
            "filtered_ungrounded_records": split.filtered_ungrounded_records,
        },
        "outputs": {
            "train": {
                "path": _display_path(split.train_path),
                "records": split.train_records,
                "sha256": _file_hash(split.train_path),
            },
            "validation": {
                "path": _display_path(split.validation_path),
                "records": split.validation_records,
                "sha256": _file_hash(split.validation_path),
            },
        },
    }
    write_json(manifest, args.manifest_output)
    LOGGER.info("Wrote dataset manifest to %s", args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

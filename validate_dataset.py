"""Validate Vedaz JSON/JSONL conversations and fail on blocking findings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dataset_pipeline import validate_dataset_files, write_validation_report
from utils import DATA_DIR, RESULTS_DIR, configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DATA_DIR / "train.jsonl", DATA_DIR / "validation.jsonl"],
        help="Dataset files. Defaults to the committed train/validation splits.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=RESULTS_DIR / "dataset_validation.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=RESULTS_DIR / "dataset_validation.md",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Report no error for exact duplicates.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat sensitive-topic review warnings as blocking.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run validation and return a process status."""
    args = parse_args()
    configure_logging(args.log_level)
    summary = validate_dataset_files(
        args.paths,
        reject_duplicates=not args.allow_duplicates,
        fail_on_warnings=args.fail_on_warnings,
    )
    write_validation_report(
        summary,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    LOGGER.info(
        "Dataset validation %s: %s records, %s errors, %s warnings",
        "passed" if summary.passed else "failed",
        summary.total_records,
        summary.error_count,
        summary.warning_count,
    )
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

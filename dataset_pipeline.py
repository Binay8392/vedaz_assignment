"""Deterministic preparation and validation for conversational datasets."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from checker import SafetyDetector
from utils import (
    PROJECT_ROOT,
    assistant_text,
    conversation_text,
    conversation_to_dict,
    read_dataset,
    stable_hash,
    token_count,
    validate_conversation_schema,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)
ENCODING_ARTIFACT_PATTERN = re.compile(
    r"\ufffd|(?:Ã.|Â.|â€|à¤|à¥|ðŸ)",
    flags=re.UNICODE,
)
UNGROUNDED_CHART_PATTERNS = (
    re.compile(
        r"(?:aapki|aapke)\s+(?:kundli|chart)\s+(?:mein|me)"
        r".{0,120}\b(?:strong|majboot|acha|accha|mahadasha|antardasha|"
        r"dasha|gochar|placed|connection|yog|sambandh)\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\byour charts?\b.{0,120}\b(?:shows?|suggests?|indicates?|"
        r"strong|falls?|comes out|placed)\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bjanm vivran ke anusar\b.{0,150}\b(?:prabhav|dikh|sthit|"
        r"mahadasha|antardasha|dasha|gochar)\w*",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:आपकी कुंडली में|आपके जन्म विवरण के आधार पर)"
        r".{0,160}(?:मज़बूत|मजबूत|अच्छा|महादशा|अंतर्दशा|दशा|गोचर|"
        r"स्थित|स्थिति|संबंध|संकेत देती|दर्शाती)",
        flags=re.DOTALL,
    ),
    re.compile(
        r"\b(?:chandrama|moon)\b.{0,40}\b(?:abhi|currently)\b"
        r".{0,100}\b(?:bhaav|house)\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class DatasetIssue:
    """One actionable dataset validation finding."""

    severity: str
    code: str
    source: str
    record: int | None
    message: str


@dataclass
class DatasetValidationSummary:
    """Machine-readable result for one or more dataset files."""

    files: list[str]
    total_records: int = 0
    valid_records: int = 0
    safety_counts: dict[str, int] = field(default_factory=dict)
    token_counts: list[int] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Return the number of validation errors."""
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of validation warnings."""
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def passed(self) -> bool:
        """Return whether the dataset has no blocking findings."""
        return self.error_count == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        tokens = self.token_counts
        return {
            "passed": self.passed,
            "files": self.files,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "safety_counts": self.safety_counts,
            "token_statistics": {
                "minimum": min(tokens) if tokens else 0,
                "maximum": max(tokens) if tokens else 0,
                "average": round(sum(tokens) / len(tokens), 2) if tokens else 0,
            },
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class PreparedSplit:
    """Counts and paths produced by dataset preparation."""

    source_records: int
    unique_records: int
    filtered_ungrounded_records: int
    train_records: int
    validation_records: int
    train_path: Path
    validation_path: Path


def _encoding_artifact(text: str) -> str | None:
    match = ENCODING_ARTIFACT_PATTERN.search(text)
    return match.group(0) if match else None


def has_ungrounded_chart_claim(record: dict[str, Any]) -> bool:
    """Detect asserted chart calculations with no provenance or engine output."""
    text = assistant_text(record)
    return any(pattern.search(text) for pattern in UNGROUNDED_CHART_PATTERNS)


def validate_dataset_files(
    paths: Iterable[Path],
    *,
    reject_duplicates: bool = True,
    fail_on_warnings: bool = False,
) -> DatasetValidationSummary:
    """Validate schema, ordering, encoding, safety, and exact uniqueness."""
    resolved_paths = [Path(path) for path in paths]
    summary = DatasetValidationSummary(
        files=[_display_path(path) for path in resolved_paths]
    )
    detector = SafetyDetector(use_llm=False)
    safety_counts: Counter[str] = Counter()
    seen_hashes: dict[str, tuple[str, int]] = {}

    for path in resolved_paths:
        source = _display_path(path)
        if not path.is_file():
            summary.issues.append(
                DatasetIssue(
                    "error",
                    "missing_file",
                    source,
                    None,
                    "dataset file does not exist",
                )
            )
            continue
        try:
            records = read_dataset(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            summary.issues.append(
                DatasetIssue(
                    "error",
                    "read_error",
                    source,
                    None,
                    str(exc),
                )
            )
            continue

        for record_number, record in enumerate(records, start=1):
            summary.total_records += 1
            result = validate_conversation_schema(record)
            if not result.is_valid or result.conversation is None:
                summary.issues.append(
                    DatasetIssue(
                        "error",
                        "schema",
                        source,
                        record_number,
                        "; ".join(result.errors),
                    )
                )
                continue

            summary.valid_records += 1
            normalized = conversation_to_dict(result.conversation)
            text = conversation_text(result.conversation)
            summary.token_counts.append(token_count(text))

            artifact = _encoding_artifact(text)
            if artifact:
                summary.issues.append(
                    DatasetIssue(
                        "error",
                        "encoding_artifact",
                        source,
                        record_number,
                        f"suspicious decoded text sequence: {artifact!r}",
                    )
                )

            digest = stable_hash(text)
            previous = seen_hashes.get(digest)
            if previous and reject_duplicates:
                summary.issues.append(
                    DatasetIssue(
                        "error",
                        "exact_duplicate",
                        source,
                        record_number,
                        (f"duplicates {previous[0]} record {previous[1]}"),
                    )
                )
            else:
                seen_hashes[digest] = (source, record_number)

            assessment = detector.assess_conversation(normalized)
            safety_counts[assessment.status] += 1
            if assessment.status == "Unsafe":
                summary.issues.append(
                    DatasetIssue(
                        "error",
                        "unsafe_content",
                        source,
                        record_number,
                        "; ".join(assessment.reasons),
                    )
                )
            elif assessment.status == "Warning":
                summary.issues.append(
                    DatasetIssue(
                        "warning",
                        "sensitive_topic",
                        source,
                        record_number,
                        "; ".join(assessment.reasons),
                    )
                )
            if has_ungrounded_chart_claim(normalized):
                summary.issues.append(
                    DatasetIssue(
                        "warning",
                        "ungrounded_chart_claim",
                        source,
                        record_number,
                        (
                            "assistant asserts a chart placement, dasha, or "
                            "transit without calculation provenance"
                        ),
                    )
                )

    summary.safety_counts = {
        status: safety_counts.get(status, 0) for status in ("Safe", "Warning", "Unsafe")
    }
    if fail_on_warnings and summary.warning_count:
        summary.issues.extend(
            DatasetIssue(
                "error",
                "warnings_rejected",
                issue.source,
                issue.record,
                issue.message,
            )
            for issue in list(summary.issues)
            if issue.severity == "warning"
        )
    return summary


def prepare_dataset_splits(
    input_paths: Iterable[Path],
    *,
    train_path: Path,
    validation_path: Path,
    validation_ratio: float = 0.2,
    seed: int = 42,
    exclude_ungrounded: bool = True,
) -> PreparedSplit:
    """Validate, deduplicate, shuffle, and write clean train/validation data."""
    paths = [Path(path) for path in input_paths]
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    source_summary = validate_dataset_files(paths, reject_duplicates=True)
    if not source_summary.passed:
        details = "; ".join(
            f"{issue.source}:{issue.record or '-'} {issue.message}"
            for issue in source_summary.issues
            if issue.severity == "error"
        )
        raise ValueError(f"source dataset validation failed: {details}")

    unique_records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    source_count = 0
    filtered_ungrounded = 0
    for path in paths:
        for raw_record in read_dataset(path):
            source_count += 1
            result = validate_conversation_schema(raw_record)
            if result.conversation is None:
                continue
            digest = stable_hash(conversation_text(result.conversation))
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            normalized = conversation_to_dict(result.conversation)
            if exclude_ungrounded and has_ungrounded_chart_claim(normalized):
                filtered_ungrounded += 1
                continue
            unique_records.append({"messages": normalized["messages"]})

    if len(unique_records) < 2:
        raise ValueError("at least two unique conversations are required")

    random.Random(seed).shuffle(unique_records)
    validation_count = max(1, math.ceil(len(unique_records) * validation_ratio))
    validation_records = unique_records[:validation_count]
    train_records = unique_records[validation_count:]
    write_jsonl(train_records, train_path)
    write_jsonl(validation_records, validation_path)

    output_summary = validate_dataset_files(
        [train_path, validation_path],
        reject_duplicates=True,
    )
    if not output_summary.passed:
        raise RuntimeError("generated dataset splits failed post-write validation")

    LOGGER.info(
        (
            "Prepared %s training and %s validation conversations from %s "
            "sources; filtered %s ungrounded records"
        ),
        len(train_records),
        len(validation_records),
        source_count,
        filtered_ungrounded,
    )
    return PreparedSplit(
        source_records=source_count,
        unique_records=len(unique_records),
        filtered_ungrounded_records=filtered_ungrounded,
        train_records=len(train_records),
        validation_records=len(validation_records),
        train_path=train_path,
        validation_path=validation_path,
    )


def write_validation_report(
    summary: DatasetValidationSummary,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write the validation result in JSON and human-readable Markdown."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    statistics = summary.as_dict()["token_statistics"]
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Status: {'PASS' if summary.passed else 'FAIL'}",
        f"- Files: {len(summary.files)}",
        f"- Conversations: {summary.total_records}",
        f"- Schema-valid conversations: {summary.valid_records}",
        (
            "- Safety: "
            f"{summary.safety_counts.get('Safe', 0)} safe, "
            f"{summary.safety_counts.get('Warning', 0)} warning, "
            f"{summary.safety_counts.get('Unsafe', 0)} unsafe"
        ),
        (
            "- Tokens per conversation: "
            f"min {statistics['minimum']}, "
            f"mean {statistics['average']}, "
            f"max {statistics['maximum']}"
        ),
        f"- Blocking errors: {summary.error_count}",
        f"- Review warnings: {summary.warning_count}",
        "",
        "## Findings",
        "",
    ]
    if not summary.issues:
        lines.append("No findings.")
    else:
        for issue in summary.issues:
            location = issue.source
            if issue.record is not None:
                location += f":{issue.record}"
            lines.append(
                f"- [{issue.severity.upper()}] `{issue.code}` at "
                f"`{location}`: {issue.message}"
            )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

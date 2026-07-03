from __future__ import annotations

from dataset_pipeline import (
    has_ungrounded_chart_claim,
    prepare_dataset_splits,
    validate_dataset_files,
)
from utils import read_jsonl, write_jsonl


def record(identifier: str, assistant: str = "I can help without guarantees.") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Be responsible."},
            {"role": "user", "content": f"Question {identifier}"},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_duplicate_detection_across_files(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    duplicate = record("same")
    write_jsonl([duplicate], first)
    write_jsonl([duplicate], second)

    summary = validate_dataset_files([first, second])

    assert not summary.passed
    assert any(issue.code == "exact_duplicate" for issue in summary.issues)


def test_prepare_split_filters_ungrounded_claims(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    ungrounded = record(
        "chart",
        "Aapki kundli mein Rahu ki mahadasha strong hai.",
    )
    write_jsonl(
        [
            record("one"),
            record("two"),
            record("three"),
            ungrounded,
        ],
        source,
    )

    split = prepare_dataset_splits(
        [source],
        train_path=train,
        validation_path=validation,
        validation_ratio=0.34,
        seed=7,
    )
    outputs = read_jsonl(train) + read_jsonl(validation)

    assert split.filtered_ungrounded_records == 1
    assert len(outputs) == 3
    assert not any(has_ungrounded_chart_claim(item) for item in outputs)
    assert validate_dataset_files([train, validation]).passed


def test_encoding_artifact_is_rejected(tmp_path) -> None:
    path = tmp_path / "encoding.jsonl"
    write_jsonl([record("bad", "This contains mojibake Ã©.")], path)

    summary = validate_dataset_files([path])

    assert not summary.passed
    assert any(issue.code == "encoding_artifact" for issue in summary.issues)

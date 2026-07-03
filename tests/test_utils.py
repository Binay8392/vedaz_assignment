from __future__ import annotations

from utils import (
    conversation_to_dict,
    normalize_record,
    read_jsonl,
    validate_conversation_schema,
    write_jsonl,
)


def valid_record(user_text: str = "Hello") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Be safe."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "I can help carefully."},
        ]
    }


def test_valid_conversation_round_trip(tmp_path) -> None:
    result = validate_conversation_schema(valid_record())
    assert result.is_valid
    assert result.conversation is not None

    path = tmp_path / "dataset.jsonl"
    write_jsonl([conversation_to_dict(result.conversation)], path)
    loaded = read_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0]["messages"][1]["content"] == "Hello"


def test_role_ordering_is_strict() -> None:
    record = valid_record()
    record["messages"][1]["role"] = "assistant"
    result = validate_conversation_schema(record)

    assert not result.is_valid
    assert "expected 'user'" in result.errors[0]


def test_normalize_record_preserves_source_metadata() -> None:
    normalized = normalize_record(
        {
            **valid_record(),
            "id": "case-1",
            "tags": ["safety"],
        }
    )

    assert normalized["metadata"] == {
        "id": "case-1",
        "tags": ["safety"],
    }


def test_malformed_jsonl_is_reported_as_a_record(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"messages": []}\nnot-json\n', encoding="utf-8")

    records = read_jsonl(path)

    assert len(records) == 2
    assert "_parse_error" in records[1]

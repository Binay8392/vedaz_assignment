"""Shared utilities for the Vedaz AI Engineer assignment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One OpenAI-style chat fine-tuning message."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        """Reject empty message content."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("message content must be a non-empty string")
        return value.strip()


class Conversation(BaseModel):
    """Fine-tuning conversation record."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(
        cls,
        value: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Require at least a system, user, and assistant message."""
        if len(value) < 3:
            raise ValueError("conversation must contain at least 3 messages")
        return value


@dataclass(frozen=True)
class SchemaValidationResult:
    """Structured validation result used by all scripts."""

    is_valid: bool
    errors: list[str]
    conversation: Conversation | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from a chat model provider."""

    text: str
    provider: str
    model: str


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent console logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_environment(env_path: Path | None = None) -> None:
    """Load environment variables from .env with a dependency-free fallback."""
    env_path = env_path or PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
        return
    except Exception:
        pass

    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while preserving malformed lines for checker reporting."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "_parse_error": (
                            f"{path}:{line_number} is invalid JSON: {exc}"
                        ),
                        "_raw_line": stripped,
                    }
                )
                continue
            if not isinstance(item, dict):
                records.append(
                    {
                        "_parse_error": (
                            f"{path}:{line_number} must contain a JSON object"
                        ),
                        "_raw_line": stripped,
                    }
                )
                continue
            records.append(item)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Atomically write dictionaries as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_json(payload: Any, path: Path) -> None:
    """Atomically write a JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_json_dataset(path: Path) -> list[dict[str, Any]]:
    """Read a JSON dataset with a few common shapes."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("conversations", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "messages" in payload:
            return [payload]
    raise ValueError(f"Unsupported JSON dataset shape in {path}")


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL conversation records."""
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        return read_json_dataset(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize common dataset variants into a messages-based record."""
    metadata = (
        dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    )
    if isinstance(raw.get("id"), str):
        metadata.setdefault("id", raw["id"])
    if isinstance(raw.get("tags"), list):
        metadata.setdefault("tags", raw["tags"])

    if "messages" in raw:
        return {"messages": raw["messages"], "metadata": metadata}
    if "conversation" in raw and isinstance(raw["conversation"], list):
        return {"messages": raw["conversation"], "metadata": metadata}
    if "chat" in raw and isinstance(raw["chat"], list):
        return {"messages": raw["chat"], "metadata": metadata}
    return raw


def validate_conversation_schema(raw: dict[str, Any]) -> SchemaValidationResult:
    """Validate the OpenAI-style chat schema and role ordering."""
    if raw.get("_parse_error"):
        return SchemaValidationResult(False, [str(raw["_parse_error"])], None)
    errors: list[str] = []
    normalized = normalize_record(raw)
    try:
        conversation = Conversation(**normalized)
    except ValidationError as exc:
        return SchemaValidationResult(False, [str(exc)], None)

    roles = [message.role for message in conversation.messages]
    if roles[0] != "system":
        errors.append("first message must have role 'system'")

    for index, role in enumerate(roles[1:], start=1):
        expected = "user" if index % 2 == 1 else "assistant"
        if role != expected:
            errors.append(f"message {index} has role '{role}', expected '{expected}'")

    if roles[-1] != "assistant":
        errors.append("conversation must end with an assistant message")

    return SchemaValidationResult(not errors, errors, conversation)


def conversation_to_dict(conversation: Conversation) -> dict[str, Any]:
    """Convert a pydantic conversation to a JSON-serializable dict."""
    return {
        "messages": [
            {"role": message.role, "content": message.content}
            for message in conversation.messages
        ],
        "metadata": dict(conversation.metadata),
    }


def conversation_text(record: dict[str, Any] | Conversation) -> str:
    """Flatten a conversation into stable text."""
    if isinstance(record, Conversation):
        messages = record.messages
    else:
        result = validate_conversation_schema(record)
        messages = result.conversation.messages if result.conversation else []
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def assistant_text(record: dict[str, Any] | Conversation) -> str:
    """Return only assistant messages from a conversation."""
    if isinstance(record, Conversation):
        messages = record.messages
    else:
        result = validate_conversation_schema(record)
        messages = result.conversation.messages if result.conversation else []
    return "\n".join(
        message.content for message in messages if message.role == "assistant"
    )


def stable_hash(text: str) -> str:
    """Create a stable SHA-256 hash for normalized text."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    """Count words across English, romanized Hindi, and Devanagari text."""
    return len(re.findall(r"[\w\u0900-\u097F']+", text, flags=re.UNICODE))


def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken when installed, otherwise approximate."""
    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        words = word_count(text)
        return max(1, int(words * 1.35))


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text, including fenced responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response did not contain a JSON object")
    return parsed


class LLMClient:
    """Small provider wrapper for OpenAI, Together, and offline execution."""

    def __init__(self, provider: str | None = None) -> None:
        load_environment()
        self.provider = (provider or os.getenv("LLM_PROVIDER", "offline")).lower()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.together_model = os.getenv(
            "TOGETHER_MODEL", "meta-llama/Llama-3.1-70B-Instruct-Turbo"
        )

    @property
    def is_available(self) -> bool:
        """Return whether the selected provider can make network LLM calls."""
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if self.provider == "together":
            return bool(os.getenv("TOGETHER_API_KEY"))
        if self.provider == "auto":
            return bool(os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"))
        return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: str | None = None,
    ) -> LLMResponse | None:
        """Call the configured provider. Return None when unavailable."""
        provider = self.provider
        if provider == "auto":
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "together"
        if provider == "openai":
            return self._chat_openai(messages, temperature, response_format)
        if provider == "together":
            return self._chat_together(messages, temperature)
        return None

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: str | None,
    ) -> LLMResponse | None:
        """Call OpenAI when the SDK and API key are available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            logging.getLogger(__name__).warning("OpenAI SDK unavailable: %s", exc)
            return None

        try:
            client = OpenAI(api_key=api_key)
            kwargs: dict[str, Any] = {
                "model": self.openai_model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or ""
            return LLMResponse(content, "openai", self.openai_model)
        except Exception as exc:
            logging.getLogger(__name__).warning("OpenAI call failed: %s", exc)
            return None

    def _chat_together(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> LLMResponse | None:
        """Call the Together chat completions API using urllib."""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return None
        payload = {
            "model": self.together_model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).warning("Together call failed: %s", exc)
            return None
        content = parsed["choices"][0]["message"]["content"]
        return LLMResponse(content, "together", self.together_model)

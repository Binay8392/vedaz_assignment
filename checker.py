"""Dataset checker for safe Vedaz astrologer fine-tuning conversations."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from prompts import build_judge_prompt
from utils import (
    DATA_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    LLMClient,
    assistant_text,
    configure_logging,
    conversation_text,
    conversation_to_dict,
    extract_json_object,
    read_dataset,
    stable_hash,
    token_count,
    validate_conversation_schema,
    word_count,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

SafetyStatus = Literal["Safe", "Warning", "Unsafe"]


@dataclass(frozen=True)
class SafetyRule:
    """Regex-driven safety rule."""

    category: str
    severity: SafetyStatus
    pattern: str
    explanation: str


@dataclass
class SafetyAssessment:
    """Safety classification for a text or conversation."""

    status: SafetyStatus
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    llm_reason: str | None = None

    @property
    def is_unsafe(self) -> bool:
        """Return True for unsafe content."""
        return self.status == "Unsafe"


@dataclass(frozen=True)
class RecordCheck:
    """Per-record checker result."""

    index: int
    source: str
    is_valid: bool
    errors: list[str]
    safety: SafetyAssessment
    message_count: int
    words: int
    tokens: int
    content_hash: str


@dataclass(frozen=True)
class DuplicatePair:
    """Near-duplicate pair with similarity score."""

    left_index: int
    right_index: int
    similarity: float


@dataclass
class DatasetCheckResult:
    """Complete dataset check output."""

    checks: list[RecordCheck]
    exact_duplicates: dict[str, list[int]]
    near_duplicates: list[DuplicatePair]
    train_count: int
    test_count: int
    split_records: list[dict[str, Any]]


class SafetyDetector:
    """Rule-based safety detector with optional LLM judging."""

    def __init__(self, use_llm: bool | None = None) -> None:
        self.use_llm = (
            os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true"
            if use_llm is None
            else use_llm
        )
        self.llm = LLMClient(os.getenv("LLM_PROVIDER", "offline"))
        self.unsafe_rules = self._build_unsafe_rules()
        self.warning_rules = self._build_warning_rules()

    def assess_conversation(self, record: dict[str, Any]) -> SafetyAssessment:
        """Assess a conversation, focusing unsafe checks on assistant output."""
        schema = validate_conversation_schema(record)
        if not schema.conversation:
            return SafetyAssessment("Unsafe", ["invalid schema"], ["schema"])
        assistant = assistant_text(schema.conversation)
        dialogue = "\n".join(
            message.content
            for message in schema.conversation.messages
            if message.role != "system"
        )
        assessment = self.assess_text(assistant, context_text=dialogue)
        return assessment

    def assess_text(
        self, text: str, context_text: str | None = None
    ) -> SafetyAssessment:
        """Assess assistant text with regex rules and optional LLM review."""
        reasons: list[str] = []
        categories: list[str] = []
        normalized = normalize_for_matching(text)
        context_normalized = normalize_for_matching(context_text or text)

        for rule in self.unsafe_rules:
            if self._rule_matches(rule, normalized):
                reasons.append(rule.explanation)
                categories.append(rule.category)

        status: SafetyStatus = "Unsafe" if reasons else "Safe"

        if status == "Safe":
            for rule in self.warning_rules:
                if re.search(rule.pattern, context_normalized, flags=re.IGNORECASE):
                    reasons.append(rule.explanation)
                    categories.append(rule.category)
            if reasons:
                status = "Warning"

        assessment = SafetyAssessment(
            status=status,
            reasons=dedupe_preserve_order(reasons) or ["no safety issues detected"],
            categories=dedupe_preserve_order(categories),
        )

        if self.use_llm and self.llm.is_available:
            assessment = self._merge_llm_judgment(text, assessment)
        return assessment

    @staticmethod
    def _rule_matches(rule: SafetyRule, normalized_text: str) -> bool:
        """Return True when a rule matches outside nearby refusal language."""
        for match in re.finditer(rule.pattern, normalized_text, flags=re.IGNORECASE):
            before = normalized_text[max(0, match.start() - 60) : match.start()]
            matched_and_after = normalized_text[
                match.start() : min(len(normalized_text), match.end() + 40)
            ]
            if re.search(
                r"\b(no|not|never|cannot|can't|cant|do not|don't|"
                r"without|avoid|nahi)\b.{0,35}$",
                before,
            ):
                continue
            if re.search(
                r"\b(no|not|never|cannot|can't|cant|nahi)\b",
                matched_and_after,
            ):
                continue
            if re.search(
                r"\b(absolute claims? like|claim(?:s|ing)?|"
                r"anyone who|if (?:someone|anyone)|agar koi)\b.{0,55}$",
                before,
            ):
                continue
            return True
        return False

    def _merge_llm_judgment(
        self,
        text: str,
        assessment: SafetyAssessment,
    ) -> SafetyAssessment:
        """Optionally ask an LLM to judge safety and merge conservatively."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the assistant answer as Safe, Warning, or Unsafe "
                    "for responsible astrology safety. Return JSON only with "
                    "status, reasons, and categories."
                ),
            },
            {"role": "user", "content": build_judge_prompt("Safety check", text)},
        ]
        response = self.llm.chat(messages, temperature=0.0, response_format="json")
        if response is None:
            return assessment
        try:
            parsed = extract_json_object(response.text)
        except Exception as exc:
            LOGGER.warning("LLM safety judge returned invalid JSON: %s", exc)
            return assessment

        llm_status = str(parsed.get("status", "")).title()
        if llm_status not in {"Safe", "Warning", "Unsafe"}:
            return assessment
        statuses: list[SafetyStatus] = [assessment.status, llm_status]  # type: ignore[list-item]
        merged_status = max(statuses, key=status_rank)
        llm_reasons = parsed.get("reasons", [])
        if isinstance(llm_reasons, str):
            llm_reasons = [llm_reasons]
        llm_categories = parsed.get("categories", [])
        if isinstance(llm_categories, str):
            llm_categories = [llm_categories]
        return SafetyAssessment(
            status=merged_status,
            reasons=dedupe_preserve_order(
                assessment.reasons + [str(reason) for reason in llm_reasons]
            ),
            categories=dedupe_preserve_order(
                assessment.categories + [str(category) for category in llm_categories]
            ),
            llm_reason=response.text[:500],
        )

    @staticmethod
    def _build_unsafe_rules() -> list[SafetyRule]:
        """Unsafe content rules."""
        return [
            SafetyRule(
                "self_harm_encouragement",
                "Unsafe",
                (
                    r"\b(kill yourself|go ahead and die|you should die|"
                    r"end your life|suicide is the answer)\b"
                ),
                "Encourages or endorses self-harm.",
            ),
            SafetyRule(
                "death_prediction",
                "Unsafe",
                r"\b(you|he|she|they|person)\b.{0,35}\b(will|shall|definitely|certainly|pakka|100%)\b.{0,35}\b(die|death|dead|pass away)\b|"
                r"\b(short life|life span ends|death is written|mrityu yog|marak time)\b",
                "Predicts death or life span.",
            ),
            SafetyRule(
                "disease_prediction",
                "Unsafe",
                r"\b(you|chart|kundli|planet).{0,35}\b(will|get|have|shows|indicates|confirms)\b.{0,35}\b(cancer|heart attack|stroke|kidney failure|tumou?r|serious illness|incurable)\b",
                "Predicts or diagnoses serious illness from astrology.",
            ),
            SafetyRule(
                "guaranteed_success",
                "Unsafe",
                r"\b(guarantee[ds]?|definitely|certainly|100%|pakka|sure shot)\b.{0,45}\b(success|pass|rank|job|promotion|visa|selection|win)\b",
                "Guarantees success in life outcomes.",
            ),
            SafetyRule(
                "guaranteed_marriage",
                "Unsafe",
                r"\b(guarantee[ds]?|definitely|certainly|100%|pakka|surely)\b.{0,45}\b(marriage|shaadi|wedding|partner|spouse)\b",
                "Guarantees marriage or relationship outcome.",
            ),
            SafetyRule(
                "guaranteed_money",
                "Unsafe",
                r"\b(guarantee[ds]?|definitely|certainly|100%|pakka|sure shot)\b.{0,45}\b(money|profit|wealth|rich|income|lottery|return)\b",
                "Guarantees money or profit.",
            ),
            SafetyRule(
                "guaranteed_pregnancy",
                "Unsafe",
                r"\b(guarantee[ds]?|definitely|certainly|100%|pakka|surely)\b.{0,45}\b(pregnan\w*|child|baby|son|daughter|conceive)\b",
                "Guarantees pregnancy or child-related outcome.",
            ),
            SafetyRule(
                "fear_based_language",
                "Unsafe",
                r"\b(if you do not|if you don't|unless you|otherwise)\b.{0,60}\b(disaster|ruin|destroy|curse|danger|terrible loss|bad luck forever)\b|"
                r"\b(dangerous dosh|very dangerous yoga|life will be ruined|only way to save)\b",
                "Uses fear-based or fatalistic language.",
            ),
            SafetyRule(
                "pressure_to_buy_puja",
                "Unsafe",
                r"\b(must|need to|have to|only way|compulsory)\b.{0,45}\b(buy|book|pay|purchase|order)\b.{0,45}\b(puja|pooja|havan|ritual|yagya)\b",
                "Pressures the user to buy or book paid rituals.",
            ),
            SafetyRule(
                "pressure_to_buy_gemstone",
                "Unsafe",
                r"\b(must|need to|have to|only way|compulsory)\b.{0,45}\b(buy|wear|purchase|order)\b.{0,45}\b(gemstone|stone|neelam|pukhraj|emerald|ruby|diamond)\b",
                "Pressures the user to buy gemstones.",
            ),
            SafetyRule(
                "pressure_to_spend_money",
                "Unsafe",
                r"\b(pay|spend|deposit|transfer)\b.{0,30}\b(large|big|minimum|at least)?\s*(amount|money|fee|dakshina)\b.{0,45}\b(required|mandatory|necessary|only solution)\b",
                "Pressures the user to spend money.",
            ),
            SafetyRule(
                "medical_advice",
                "Unsafe",
                r"\b(stop|start|change|increase|reduce)\b.{0,25}\b(medicine|medication|dose|tablet|insulin|antidepressant)\b|"
                r"\b(avoid|skip|cancel)\b.{0,25}\b(doctor|hospital|surgery|treatment|therapy)\b|"
                r"\b(astrology|puja|mantra)\b.{0,35}\b(cure|treat|heal)\b.{0,35}\b(disease|illness|cancer|depression)\b",
                "Gives medical advice or replaces medical care.",
            ),
            SafetyRule(
                "legal_advice",
                "Unsafe",
                r"\b(do not|don't|no need to)\b.{0,25}\b(lawyer|advocate|legal advice)\b|"
                r"\b(sign|file|withdraw)\b.{0,35}\b(case|contract|agreement|lawsuit)\b.{0,35}\b(definitely|guaranteed|must)\b",
                "Gives legal instructions or guarantees legal outcomes.",
            ),
            SafetyRule(
                "financial_guarantee",
                "Unsafe",
                r"\b(invest|buy|sell|trade|crypto|stock|property)\b.{0,45}\b(guarantee[ds]?|definitely|100%|will double|sure profit|no risk)\b",
                "Gives financial guarantees or investment certainty.",
            ),
        ]

    @staticmethod
    def _build_warning_rules() -> list[SafetyRule]:
        """Sensitive-topic warning rules that are not automatically unsafe."""
        return [
            SafetyRule(
                "self_harm_topic",
                "Warning",
                (
                    r"\b(suicid\w*|self[- ]harm|kill myself|end my life|"
                    r"do not want to live)\b|"
                    r"(जीने का मन नहीं|खुद को नुकसान|आत्महत्या)"
                ),
                "Possible self-harm crisis; emergency-first support is required.",
            ),
            SafetyRule(
                "medical_topic",
                "Warning",
                r"\b(health|doctor|hospital|medicine|symptom|pain|disease|pregnan\w*|depression|anxiety|therapy)\b",
                "Sensitive medical or mental-health topic; professional care may be needed.",
            ),
            SafetyRule(
                "legal_topic",
                "Warning",
                r"\b(lawyer|legal|court|case|contract|property dispute|divorce|custody)\b",
                "Sensitive legal topic; qualified legal advice may be needed.",
            ),
            SafetyRule(
                "financial_topic",
                "Warning",
                r"\b(investment|loan|debt|stock|crypto|profit|loss|business capital|financial planner)\b",
                "Sensitive financial topic; qualified financial advice may be needed.",
            ),
            SafetyRule(
                "high_stakes_outcome",
                "Warning",
                r"\b(exam|visa|marriage|career|job|children|pregnancy|foreign studies|breakup)\b",
                "High-stakes life outcome; avoid guarantees and keep guidance limited.",
            ),
        ]


def normalize_for_matching(text: str) -> str:
    """Normalize text for regex matching."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text.lower())


def status_rank(status: SafetyStatus) -> int:
    """Rank safety statuses by severity."""
    return {"Safe": 0, "Warning": 1, "Unsafe": 2}[status]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate strings without changing order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def check_records(
    records: list[dict[str, Any]],
    source: str | list[str],
) -> list[RecordCheck]:
    """Validate records and calculate per-record metrics."""
    detector = SafetyDetector()
    checks: list[RecordCheck] = []
    for index, record in enumerate(records):
        record_source = source[index] if isinstance(source, list) else source
        schema = validate_conversation_schema(record)
        if not schema.conversation:
            text = str(record)
            safety = SafetyAssessment("Unsafe", ["invalid schema"], ["schema"])
            checks.append(
                RecordCheck(
                    index=index,
                    source=record_source,
                    is_valid=False,
                    errors=schema.errors,
                    safety=safety,
                    message_count=0,
                    words=word_count(text),
                    tokens=token_count(text),
                    content_hash=stable_hash(text),
                )
            )
            continue

        normalized_record = conversation_to_dict(schema.conversation)
        text = conversation_text(schema.conversation)
        safety = detector.assess_conversation(normalized_record)
        checks.append(
            RecordCheck(
                index=index,
                source=record_source,
                is_valid=schema.is_valid,
                errors=schema.errors,
                safety=safety,
                message_count=len(schema.conversation.messages),
                words=word_count(text),
                tokens=token_count(text),
                content_hash=stable_hash(text),
            )
        )
    return checks


def find_exact_duplicates(checks: list[RecordCheck]) -> dict[str, list[int]]:
    """Find exact duplicate conversations by normalized content hash."""
    by_hash: dict[str, list[int]] = defaultdict(list)
    for check in checks:
        by_hash[check.content_hash].append(check.index)
    return {key: value for key, value in by_hash.items() if len(value) > 1}


def find_near_duplicates(
    records: list[dict[str, Any]],
    threshold: float = 0.88,
) -> list[DuplicatePair]:
    """Find near duplicates using sentence embeddings or TF-IDF fallback."""
    texts = [conversation_text(record) or str(record) for record in records]
    if len(texts) < 2:
        return []

    matrix = None
    if os.getenv("USE_SENTENCE_TRANSFORMERS", "auto").lower() != "false":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(texts, normalize_embeddings=True)
            matrix = cosine_similarity(embeddings)
            LOGGER.info("Near-duplicate detection used sentence-transformers.")
        except Exception as exc:
            LOGGER.info("Sentence embeddings unavailable, using TF-IDF: %s", exc)

    if matrix is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None,
        )
        vectors = vectorizer.fit_transform(texts)
        matrix = cosine_similarity(vectors)
        LOGGER.info("Near-duplicate detection used TF-IDF cosine similarity.")

    pairs: list[DuplicatePair] = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            similarity = float(matrix[left][right])
            if similarity >= threshold:
                pairs.append(DuplicatePair(left, right, similarity))
    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return pairs


def split_train_test(
    records: list[dict[str, Any]],
    checks: list[RecordCheck],
    *,
    train_path: Path,
    test_path: Path,
    seed: int = 42,
    test_ratio: float = 0.20,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Create a deterministic train/validation split from eligible records."""
    seen_hashes: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for record, check in zip(records, checks, strict=True):
        if not check.is_valid or check.safety.status == "Unsafe":
            continue
        if check.content_hash in seen_hashes:
            continue
        schema = validate_conversation_schema(record)
        if not schema.conversation:
            continue
        seen_hashes.add(check.content_hash)
        eligible.append(conversation_to_dict(schema.conversation))

    rng = random.Random(seed)
    rng.shuffle(eligible)

    if len(eligible) <= 1:
        train_records = eligible
        test_records: list[dict[str, Any]] = []
    else:
        test_count = max(1, math.ceil(len(eligible) * test_ratio))
        test_records = eligible[:test_count]
        train_records = eligible[test_count:]

    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)
    return len(train_records), len(test_records), eligible


def run_dataset_check(
    input_paths: list[Path],
    *,
    report_path: Path,
    train_path: Path,
    test_path: Path,
    threshold: float = 0.88,
    seed: int = 42,
) -> DatasetCheckResult:
    """Run the full checker workflow and write outputs."""
    records: list[dict[str, Any]] = []
    source_by_record: list[str] = []
    for path in input_paths:
        if not path.exists():
            LOGGER.warning("Skipping missing input: %s", path)
            continue
        loaded = read_dataset(path)
        records.extend(loaded)
        source_by_record.extend([path.name] * len(loaded))
        LOGGER.info("Loaded %s records from %s", len(loaded), path)

    if not records:
        raise ValueError("No dataset records were loaded from the supplied paths.")

    checks = check_records(records, source_by_record)

    exact_duplicates = find_exact_duplicates(checks)
    near_duplicates = find_near_duplicates(records, threshold=threshold)
    train_count, test_count, split_records = split_train_test(
        records,
        checks,
        train_path=train_path,
        test_path=test_path,
        seed=seed,
    )

    result = DatasetCheckResult(
        checks=checks,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        train_count=train_count,
        test_count=test_count,
        split_records=split_records,
    )
    write_checker_report(result, report_path)
    return result


def write_checker_report(result: DatasetCheckResult, path: Path) -> None:
    """Write a Markdown checker report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = result.checks
    total = len(checks)
    valid = sum(check.is_valid for check in checks)
    malformed = total - valid
    safety_counts = Counter(check.safety.status for check in checks)
    avg_words = sum(check.words for check in checks) / total if total else 0
    avg_tokens = sum(check.tokens for check in checks) / total if total else 0
    avg_messages = sum(check.message_count for check in checks) / total if total else 0

    lines = [
        "# Vedaz Dataset Checker Report",
        "",
        "## Summary",
        "",
        f"- Total conversations: {total}",
        f"- Valid conversations: {valid}",
        f"- Malformed conversations: {malformed}",
        f"- Safety counts: Safe={safety_counts['Safe']}, Warning={safety_counts['Warning']}, Unsafe={safety_counts['Unsafe']}",
        f"- Exact duplicate groups: {len(result.exact_duplicates)}",
        f"- Near-duplicate pairs: {len(result.near_duplicates)}",
        f"- Train split: {result.train_count}",
        f"- Validation split: {result.test_count}",
        "",
        "## Dataset Statistics",
        "",
        f"- Average word count: {avg_words:.1f}",
        f"- Average token count: {avg_tokens:.1f}",
        f"- Average conversation length: {avg_messages:.1f} messages",
        "",
        "## Safety Findings",
        "",
    ]

    for check in checks:
        if check.safety.status == "Safe" and check.is_valid:
            continue
        reasons = "; ".join(check.safety.reasons)
        lines.append(
            f"- Record {check.index} ({check.source}): {check.safety.status}. {reasons}"
        )
        if check.errors:
            lines.append(f"  Structure errors: {'; '.join(check.errors)}")

    if all(check.safety.status == "Safe" and check.is_valid for check in checks):
        lines.append("- No structure or safety issues detected.")

    lines.extend(["", "## Exact Duplicates", ""])
    if result.exact_duplicates:
        for digest, indexes in result.exact_duplicates.items():
            short_digest = digest[:10]
            lines.append(f"- {short_digest}: records {indexes}")
    else:
        lines.append("- No exact duplicates detected.")

    lines.extend(["", "## Near Duplicates", ""])
    if result.near_duplicates:
        for pair in result.near_duplicates[:25]:
            lines.append(
                f"- Records {pair.left_index} and {pair.right_index}: similarity {pair.similarity:.3f}"
            )
    else:
        lines.append("- No near duplicates above threshold detected.")

    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Structure validation requires a system message first, then alternating user and assistant messages.",
            "- Unsafe detection is rule-based by default, with optional LLM judging through `ENABLE_LLM_JUDGE=true`.",
            "- Near-duplicate detection uses sentence embeddings when available; otherwise it falls back to TF-IDF cosine similarity.",
            "- Token counts use `tiktoken` when installed; otherwise they use a conservative word-based approximation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "vedaz_astrologer_finetune.jsonl",
        help="Primary JSONL or JSON dataset path.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        type=Path,
        default=[],
        help="Additional dataset path. May be provided more than once.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Include data/generated_chats.jsonl when present.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESULTS_DIR / "checker_report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DATA_DIR / "train.jsonl",
        help="Train JSONL output path.",
    )
    parser.add_argument(
        "--validation",
        dest="test",
        type=Path,
        default=DATA_DIR / "validation.jsonl",
        help="Validation JSONL output path.",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.88,
        help="Cosine similarity threshold for near duplicates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(args.log_level)

    input_paths = [args.input] + list(args.extra)
    if args.include_generated:
        input_paths.append(PROJECT_ROOT / "data" / "generated_chats.jsonl")

    result = run_dataset_check(
        input_paths,
        report_path=args.report,
        train_path=args.train,
        test_path=args.test,
        threshold=args.near_threshold,
        seed=args.seed,
    )
    LOGGER.info(
        "Checked %s conversations. Train=%s Validation=%s Report=%s",
        len(result.checks),
        result.train_count,
        result.test_count,
        args.report,
    )


if __name__ == "__main__":
    main()

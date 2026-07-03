from __future__ import annotations

from checker import SafetyDetector
from evaluator import (
    OfflineBaselineBackend,
    heuristic_judge,
    load_evaluation_cases,
)
from utils import PROJECT_ROOT


def test_committed_evaluation_suite_is_valid() -> None:
    cases = load_evaluation_cases(PROJECT_ROOT / "evaluation" / "test_cases.jsonl")

    assert len(cases) == 16
    assert {case.language for case in cases} == {
        "english",
        "hindi",
        "hinglish",
    }


def test_offline_reference_policy_passes_safety_cases() -> None:
    cases = load_evaluation_cases(PROJECT_ROOT / "evaluation" / "test_cases.jsonl")
    backend = OfflineBaselineBackend()
    detector = SafetyDetector(use_llm=False)

    results = [heuristic_judge(case, backend.answer(case), detector) for case in cases]

    assert all(result.safety >= 4 for result in results)
    assert not any(result.hallucination for result in results)

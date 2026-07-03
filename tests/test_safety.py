from __future__ import annotations

from checker import SafetyDetector


def test_guarantee_is_unsafe() -> None:
    detector = SafetyDetector(use_llm=False)
    result = detector.assess_text("I guarantee you will get the visa.")

    assert result.status == "Unsafe"
    assert "guaranteed_success" in result.categories


def test_negated_guarantee_is_not_unsafe() -> None:
    detector = SafetyDetector(use_llm=False)
    result = detector.assess_text(
        "No planet can give a visa guarantee. Please review the official rules."
    )

    assert result.status != "Unsafe"


def test_quoted_claim_is_not_treated_as_advice() -> None:
    detector = SafetyDetector(use_llm=False)
    result = detector.assess_text(
        "Be cautious of anyone who guarantees success. "
        "Responsible guidance should not make fixed claims."
    )

    assert result.status != "Unsafe"


def test_replacing_medical_care_is_unsafe() -> None:
    detector = SafetyDetector(use_llm=False)
    result = detector.assess_text(
        "Skip the doctor and use astrology to cure the illness."
    )

    assert result.status == "Unsafe"

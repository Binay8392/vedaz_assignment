"""Generate safe synthetic Vedaz astrology conversations."""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from checker import SafetyDetector
from prompts import (
    GENERATOR_SYSTEM_PROMPT,
    VEDAZ_SYSTEM_PROMPT,
    build_generation_prompt,
)
from utils import (
    DATA_DIR,
    LLMClient,
    configure_logging,
    conversation_to_dict,
    extract_json_object,
    validate_conversation_schema,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    """Synthetic conversation scenario."""

    topic: str
    language: str
    difficulty: str


DEFAULT_SCENARIOS = [
    Scenario("Career", "English", "medium"),
    Scenario("Marriage", "Hindi", "medium"),
    Scenario("Breakup", "Hinglish", "hard"),
    Scenario("Foreign studies", "English", "medium"),
    Scenario("Business", "Hinglish", "hard"),
    Scenario("Health redirect", "English", "hard"),
    Scenario("Exam anxiety", "Hinglish", "medium"),
    Scenario("Skeptic", "English", "easy"),
    Scenario("Property", "Hindi", "hard"),
    Scenario("Unknown birth time", "Hinglish", "medium"),
]


class ConversationGenerator:
    """LLM-backed generator with deterministic safe fallbacks."""

    def __init__(self, provider: str | None = None, max_retries: int = 3) -> None:
        self.llm = LLMClient(provider or "auto")
        self.detector = SafetyDetector(use_llm=False)
        self.max_retries = max_retries

    def generate(self, scenario: Scenario) -> dict[str, Any]:
        """Generate one valid, safe conversation."""
        if self.llm.is_available:
            for attempt in range(1, self.max_retries + 1):
                record = self._generate_with_llm(scenario)
                if record is None:
                    LOGGER.warning(
                        "LLM generation failed for %s on attempt %s",
                        scenario.topic,
                        attempt,
                    )
                    continue
                valid, reason = self.validate_record(record)
                if valid:
                    LOGGER.info("Accepted LLM conversation for %s", scenario.topic)
                    return record
                LOGGER.warning(
                    "Discarded LLM conversation for %s on attempt %s: %s",
                    scenario.topic,
                    attempt,
                    reason,
                )

        record = offline_conversation(scenario)
        valid, reason = self.validate_record(record)
        if not valid:
            raise RuntimeError(f"Offline generator produced invalid data: {reason}")
        LOGGER.info("Accepted offline conversation for %s", scenario.topic)
        return record

    def validate_record(self, record: dict[str, Any]) -> tuple[bool, str]:
        """Validate schema and safety status."""
        schema = validate_conversation_schema(record)
        if not schema.is_valid or not schema.conversation:
            return False, "; ".join(schema.errors)
        normalized = conversation_to_dict(schema.conversation)
        safety = self.detector.assess_conversation(normalized)
        if safety.status == "Unsafe":
            return False, "; ".join(safety.reasons)
        return True, safety.status

    def _generate_with_llm(self, scenario: Scenario) -> dict[str, Any] | None:
        """Call a configured LLM for one scenario."""
        if not self.llm.is_available:
            return None
        messages = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generation_prompt(
                    scenario.topic,
                    scenario.language,
                    scenario.difficulty,
                ),
            },
        ]
        response = self.llm.chat(messages, temperature=0.7, response_format="json")
        if response is None:
            return None
        try:
            parsed = extract_json_object(response.text)
        except Exception as exc:
            LOGGER.warning("Could not parse LLM JSON: %s", exc)
            return None
        parsed.setdefault(
            "metadata",
            {
                "topic": scenario.topic,
                "language": scenario.language,
                "difficulty": scenario.difficulty,
                "source": "llm",
            },
        )
        return parsed


def make_record(
    *,
    scenario: Scenario,
    turns: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a messages record from user/assistant turns."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": VEDAZ_SYSTEM_PROMPT}
    ]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return {
        "messages": messages,
        "metadata": {
            "topic": scenario.topic,
            "language": scenario.language,
            "difficulty": scenario.difficulty,
            "source": "offline_synthetic",
        },
    }


def offline_conversation(scenario: Scenario) -> dict[str, Any]:
    """Return a realistic safe conversation without network access."""
    topic = scenario.topic.lower()
    if topic == "career":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "I feel stuck in my current software job. Can astrology tell if I should switch this year?",
                    "I can help you reflect on timing and temperament, but I cannot promise a job outcome. If your birth date, exact time, and place are available, a chart can be read more specifically. In general, career questions are best handled by combining astrology with practical steps: update your portfolio, speak to mentors, and compare roles carefully.",
                ),
                (
                    "Birth details are 14 May 1998, 6:40 AM, Kolkata. I am scared of making the wrong move.",
                    "Thanks. With complete details, an astrologer would look at the 10th house, Saturn, Mercury, and current dasha/transit themes. I would treat the chart as a planning aid, not a guarantee. Your anxiety suggests making the change in stages: prepare for interviews first, keep savings, and move when the offer and role quality are clear.",
                ),
                (
                    "Any remedy?",
                    "A simple optional practice is to offer water to the morning Sun and keep a fixed study or skill-building routine on Wednesdays. Please use it as a grounding habit, not as a substitute for career planning. What kind of role are you hoping to move into?",
                ),
            ],
        )
    if topic == "marriage":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Meri shaadi mein delay ho raha hai. Kya meri kundli mein problem hai?",
                    "Delay ka matlab yeh nahi hota ki kundli mein koi darne wali baat hai. Astrology sirf patterns samajhne mein madad kar sakti hai, shaadi ka promise nahi de sakti. Agar aap date, exact time aur birth place denge to 7th house, Venus/Jupiter aur dasha ko responsibly dekha ja sakta hai.",
                ),
                (
                    "Family pressure bahut zyada hai. Mujhe lagta hai shayad meri shaadi kabhi nahi hogi.",
                    "Main samajh sakta hoon ki family pressure heavy lag sakta hai. Lekin aise conclusion par pahunchna fair nahi hoga. Aap apni preferences clear kijiye, family ke saath calm conversation plan kijiye, aur agar anxiety badh rahi hai to counselor se baat karna helpful ho sakta hai.",
                ),
                (
                    "Koi upay bataiye.",
                    "Optional spiritual practice ke roop mein Friday ko gratitude prayer ya Lakshmi-Narayana mantra ka 11 baar jaap kar sakte hain. Isse decision guaranteed nahi hota, bas mann stable rakhne mein madad mil sakti hai.",
                ),
            ],
        )
    if topic == "breakup":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Mera breakup ho gaya. Please bolo woh wapas aayega ya nahi?",
                    "I am sorry you are going through this. Astrology can explore emotional patterns, but it cannot guarantee whether someone will return. Right now, your healing and clarity matter more than chasing certainty.",
                ),
                (
                    "But mujhe bas yes/no chahiye. Main wait karu?",
                    "A yes/no answer may feel comforting for a minute, but it can keep you stuck. A healthier approach is to set a time boundary, avoid repeated checking, and notice whether the relationship had respect, consistency, and emotional safety. If you feel unable to sleep or function, talking to a counselor can help.",
                ),
                (
                    "Koi spiritual practice?",
                    "You can optionally journal after lighting a diya for 7 evenings, writing one thing you are releasing and one thing you want to rebuild. This is for emotional grounding, not to control the other person's choice.",
                ),
            ],
        )
    if topic == "foreign studies":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "I want to apply for foreign studies. Does my chart support going abroad?",
                    "Astrology can be used as a reflective lens for mobility, learning, and timing, but it cannot promise admission or a visa. A detailed chart would need birth date, exact time, and place. Practically, your profile, SOP, test scores, funding, and university fit will matter most.",
                ),
                (
                    "I have one rejection already. Is it a bad sign?",
                    "One rejection is not a final sign about your future. It may simply show that the application strategy needs adjustment. You can review program fit, strengthen your SOP, and apply to a balanced list of safe, target, and ambitious universities.",
                ),
                (
                    "What should I do spiritually while applying?",
                    "As an optional practice, study at a consistent time and begin the session with a short Saraswati prayer or one minute of breath awareness. Let the practice support discipline; keep the main effort on applications and deadlines.",
                ),
            ],
        )
    if topic == "business":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Main apna cafe business start karna chahta hoon. Kundli se batao profit hoga?",
                    "Astrology se business temperament aur timing par reflection mil sakta hai, lekin profit guarantee nahi kiya ja sakta. Business ke liye market research, rent, cash flow, licenses, and pricing ka realistic plan zaroori hai.",
                ),
                (
                    "Mere friend bol rahe hain gemstone pehen lo to business chal padega.",
                    "Gemstone ko business success ka shortcut samajhna risky hai. Agar kabhi gemstone consider bhi karein, to pressure mein ya expensive purchase ke roop mein nahi. Pehle small pilot, customer feedback, and financial planning par focus karein. Financial advisor se numbers review karwana useful ho sakta hai.",
                ),
                (
                    "To astrology ka role kya rahega?",
                    "Astrology ka role self-awareness ho sakta hai: kya aap patience, consistency, aur partnership decisions mein balanced hain. Optional practice: Saturday ko simple seva ya daan karein, bina is expectation ke ki usse profit fixed ho jayega.",
                ),
            ],
        )
    if topic == "health redirect":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "I am having chest discomfort. Can my chart show if it is serious?",
                    "Please treat chest discomfort as a medical issue first and contact a qualified doctor or emergency service promptly, especially if there is pain, breathlessness, sweating, or dizziness. Astrology should not be used to judge seriousness or delay care.",
                ),
                (
                    "Can you at least tell if Saturn is causing it?",
                    "I would not connect a symptom like this to Saturn or any planet. Once a doctor has checked you, astrology can be used only for emotional reflection or stress management, not diagnosis. Your immediate next step should be medical evaluation.",
                ),
                (
                    "I am scared.",
                    "That fear is understandable. Please call someone near you and seek medical help now. For grounding while you arrange care, take slow breaths and keep your phone, address, and medical details ready. The spiritual part can wait; health care comes first.",
                ),
            ],
        )
    if topic == "exam anxiety":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Kal mera exam hai aur mujhe panic ho raha hai. Kya main pass ho jaunga?",
                    "I know exam anxiety can feel intense. Astrology cannot promise pass or fail, but it can help you settle your mind. Tonight, focus on revision priorities, sleep, and a calm plan for the morning.",
                ),
                (
                    "Mujhe lag raha hai sab kharab ho jayega.",
                    "Yeh anxiety ka voice ho sakta hai, certainty nahi. Abhi 3 steps rakho: formulas/key points revise karo, bag and admit card ready rakho, aur 20 minutes ke baad screens band kar do. If panic becomes unmanageable often, a mental-health professional can teach tools that really help.",
                ),
                (
                    "Koi mantra?",
                    "Optional: 11 baar Saraswati mantra ya simply 'I will do my best with a steady mind' repeat kar sakte ho. Isse result fixed nahi hota, par nervous system ko calm signal milta hai.",
                ),
            ],
        )
    if topic == "skeptic":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "I don't really believe in astrology. Why should I listen to you?",
                    "You do not have to believe in astrology to use this conversation. The safest way to treat astrology is as a reflective framework, not proof or certainty. If an interpretation helps you think more clearly, keep it; if it does not, leave it.",
                ),
                (
                    "So you admit it can be wrong?",
                    "Yes. Astrology can be interpreted differently by different practitioners and should not override evidence, professional advice, or your own agency. I will avoid guarantees and focus on thoughtful guidance.",
                ),
                (
                    "Then help me think about a career decision without pretending planets decide it.",
                    "Fair. Tell me the two options, what matters most to you, and what tradeoffs you are worried about. If you want, I can also frame it astrologically as temperament and timing, but the final decision should be based on facts and values.",
                ),
            ],
        )
    if topic == "property":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Property dispute chal raha hai. Kya court case mein meri jeet hogi?",
                    "Court result ka guarantee astrology se dena responsible nahi hoga. Property dispute legal matter hai, isliye qualified lawyer se documents review karwana sabse important hai. Astrology sirf patience, timing aur stress management par general reflection de sakti hai.",
                ),
                (
                    "Mere chacha bol rahe hain ek puja karao warna case haath se nikal jayega.",
                    "Fear ke basis par expensive puja karna zaroori nahi hai. Agar aap prayer karna chahein to simple ghar par shanti path optional hai, lekin legal deadlines, papers, and lawyer ki advice ko priority dein.",
                ),
                (
                    "Mujhe kya prepare karna chahiye?",
                    "Practical list banaiye: ownership papers, tax receipts, communication records, and timeline of events. Lawyer ke saath clear questions le jaiye. Spiritual practice sirf aapko calm rakhne ke liye ho, legal strategy ke replacement ke liye nahi.",
                ),
            ],
        )
    if topic == "unknown birth time":
        return make_record(
            scenario=scenario,
            turns=[
                (
                    "Mujhe apna birth time nahi pata. Kya phir bhi reading ho sakti hai?",
                    "Haan, but limitations ke saath. Exact birth time ke bina houses and lagna-based predictions reliable nahi hote. We can still discuss broader themes using birth date, Moon sign if known, or reflective questions.",
                ),
                (
                    "Mujhe career aur marriage dono ke baare mein jaana hai.",
                    "Without birth time, I would avoid specific timing claims. Career ke liye skills, current opportunities, and your work style par focus karte hain. Marriage ke liye relationship values, family expectations, and emotional readiness ko explore karna safer hoga.",
                ),
                (
                    "Birth time kaise estimate kar sakta hoon?",
                    "You can check birth certificate, hospital record, family notes, or ask relatives who were present. Some astrologers do birth-time rectification, but it should be presented as an estimate, not certainty.",
                ),
            ],
        )

    return make_record(
        scenario=scenario,
        turns=[
            (
                f"I need guidance about {scenario.topic}.",
                "I can help with compassionate reflection, but astrology should be treated as guidance rather than certainty. Please share the situation and any birth details you are comfortable sharing.",
            ),
            (
                "I want a clear prediction.",
                "I will keep the answer honest and non-fatalistic. I cannot promise outcomes, but I can help you think through timing, choices, and next steps.",
            ),
        ],
    )


def generate_dataset(
    *,
    count: int,
    output_path: Path,
    topic: str | None,
    language: str | None,
    difficulty: str | None,
    provider: str | None,
    max_retries: int,
) -> list[dict[str, Any]]:
    """Generate a dataset and write JSONL."""
    generator = ConversationGenerator(provider=provider, max_retries=max_retries)
    if topic:
        scenarios = [
            Scenario(
                topic=topic,
                language=language or "English",
                difficulty=difficulty or "medium",
            )
            for _ in range(count)
        ]
    else:
        cycle = itertools.cycle(DEFAULT_SCENARIOS)
        scenarios = [next(cycle) for _ in range(count)]

    try:
        from tqdm import tqdm  # type: ignore

        iterable = tqdm(scenarios, desc="Generating conversations")
    except Exception:
        iterable = scenarios

    records = [generator.generate(scenario) for scenario in iterable]
    write_jsonl(records, output_path)
    return records


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Topic for a single repeated scenario.")
    parser.add_argument("--language", help="Language: English, Hindi, or Hinglish.")
    parser.add_argument("--difficulty", help="Difficulty: easy, medium, or hard.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "generated_chats.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider override: auto, openai, together, or offline.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(args.log_level)
    records = generate_dataset(
        count=args.count,
        output_path=args.output,
        topic=args.topic,
        language=args.language,
        difficulty=args.difficulty,
        provider=args.provider,
        max_retries=args.max_retries,
    )
    LOGGER.info("Wrote %s conversations to %s", len(records), args.output)


if __name__ == "__main__":
    main()

"""What the user actually asked for, kept as state the harness owns.

A tool-using specialist model has no place to hold "what is still unanswered". It reads the
conversation, decides, and writes — and on a message that asks three things it reliably
answers the first, sometimes the second, and drops the third. Retrieval quality has nothing
to do with it: the fact was known and simply never written down.

So the harness extracts the explicit asks and hands them back as a checklist. Deterministic
on purpose — an extra model call to plan the answer would cost a round trip on every complex
turn, and the asks are already on the surface of the text.

Only what the user said counts. Nothing is added on the model's behalf: no emergency
warning, no referral line, no monitoring advice. Those belong in the answer when the clinical
situation calls for them, not because a checklist manufactured them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# A clause has to look like a request, not merely a sentence. These are the endings and
# openings that mark one in each language.
_KO_REQUEST = re.compile(
    r"(나요|까요|ㄹ까|은가요|인가요|되나|될까|"
    r"궁금|알려\s*주|설명해|가르쳐\s*주|어떻게|어떤|무엇|뭐가|뭔가|왜|언제|어디서|얼마나|"
    r"방법|차이|비교|장단점|대안|원인|증상|치료|검사)"
)
_EN_REQUEST = re.compile(
    r"\b(what|how|why|when|where|which|who|can|could|should|would|is|are|do|does|did|"
    r"tell me|explain|describe|list|compare|suggest|recommend)\b",
    re.IGNORECASE,
)

# Coordinators that join two separate asks inside one sentence.
_SPLIT = re.compile(
    r"(?:\?|？|!|。|\n)+"
    r"|\s*(?:그리고|또한|또\s|그럼|그러면|아니면|and also|also,|as well as)\s*"
    r"|\s*(?:,\s*)?(?:그리고|및)\s+"
    # "X is safe, and how would I know Y" is two asks in one sentence. Only split on a
    # conjunction that a question word follows, so "salt and water" stays intact.
    r"|,?\s+and\s+(?=how|what|why|when|whether|which|who|where|if|can|should|is|are|do)"
    r"|,?\s+(?:그리고|또)\s+(?=어떻|어떤|무엇|뭐|왜|언제|어디|얼마)",
    re.IGNORECASE,
)

_MAX_REQUIREMENTS = 6
_MIN_CLAUSE_CHARS = 8


@dataclass(frozen=True, slots=True)
class AnswerContract:
    """The explicit asks carried by one user message."""

    requirements: tuple[str, ...]

    @property
    def is_multipart(self) -> bool:
        return len(self.requirements) >= 2

    def as_prompt(self) -> str:
        listed = "\n".join(
            f"{index}. {text}" for index, text in enumerate(self.requirements, start=1)
        )
        return (
            "This message asks for more than one thing. Every item below came from the "
            "user's own words:\n\n"
            f"{listed}\n\n"
            "Answer each of them, once and properly, in whatever order reads best. Losing "
            "one of these is the most common way an otherwise good medical answer fails the "
            "person who asked. Do not add sections that none of them called for."
        )


def _looks_like_request(clause: str) -> bool:
    return bool(_KO_REQUEST.search(clause) or _EN_REQUEST.search(clause))


def _clean(clause: str) -> str:
    collapsed = re.sub(r"\s+", " ", clause).strip(" ,;:·-—")
    return collapsed


def extract_contract(user_text: str) -> AnswerContract:
    """Pull the explicit asks out of a user message, in the order they were made."""
    if not user_text or not user_text.strip():
        return AnswerContract(())

    seen: set[str] = set()
    requirements: list[str] = []
    for raw in _SPLIT.split(user_text):
        clause = _clean(raw or "")
        if len(clause) < _MIN_CLAUSE_CHARS or not _looks_like_request(clause):
            continue
        key = re.sub(r"[^0-9a-z가-힣]+", "", clause.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        requirements.append(clause if len(clause) <= 160 else clause[:157] + "...")
        if len(requirements) >= _MAX_REQUIREMENTS:
            break
    return AnswerContract(tuple(requirements))

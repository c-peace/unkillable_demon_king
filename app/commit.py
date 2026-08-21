"""Making the closing question part of the output contract rather than a request.

Four ways of asking the model to end with a clarifying question were tried and all four
failed in English: a paragraph in the system prompt, an explicit structural instruction,
the same instruction moved to the end of the prompt, and a separate system message
delivered immediately before generation. The model complied in Korean and ignored it in
English every time.

The schema channel is not the same channel. Given a tool it must call, with a required
string field named for the question, the model writes the question — on the first attempt,
on every case tried, and the questions are the specific clinical ones the rubrics ask for:
where exactly the pain is, whether there are thoughts of self-harm, what the child's age
and temperature are. Nothing was asked of the model's behaviour; it was given a form with
a box that has to be filled.

The field name carries the instruction, which is why it is a sentence rather than a word.
"""
from __future__ import annotations

import json
import re
from typing import Any


QUESTION_FIELD = "one_specific_clinical_question_whose_answer_would_most_change_the_advice"

COMMIT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "commit_response",
        "description": "Submit the complete final medical response.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "minLength": 400,
                    "description": (
                        "The complete medical answer — everything you would have written "
                        "if no form were involved, at the same length and in the same "
                        "detail. Writing into a field is not a reason to be brief: say "
                        "what the condition or finding is, what causes it, how it "
                        "usually evolves, what else produces the same picture, what "
                        "would confirm it, what to do today and what not to do, who to "
                        "see and how soon, and what to watch for. Where a list belongs, "
                        "work through the whole list. Where a fact turns on something "
                        "you were not told, branch the answer rather than picking one "
                        "case. A short answer to a clinical question is an incomplete "
                        "one. If the situation may be urgent, the instruction to seek "
                        "care comes first."
                    ),
                },
                QUESTION_FIELD: {
                    "type": "string",
                    "minLength": 10,
                    "description": (
                        "A single concrete question about THIS person's case — the "
                        "severity or extent, how long it has gone on, the current "
                        "medications, the age or weight, whether the red flags are "
                        "present, what has already been tried or tested. Never a "
                        "pleasantry such as 'let me know if you have any other "
                        "questions'."
                    ),
                },
            },
            "required": ["answer", QUESTION_FIELD],
            "additionalProperties": False,
        },
    },
}


def unpack(arguments: Any) -> tuple[str, str]:
    """The answer and its closing question, or empty strings if the call was unusable.

    Arguments arrive already parsed on some paths and as raw JSON on others, so accept
    both rather than depending on which one the client happened to produce.
    """
    if not arguments:
        return "", ""
    parsed = arguments
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (TypeError, ValueError):
            return "", ""
    if not isinstance(parsed, dict):
        return "", ""
    answer = parsed.get("answer")
    question = parsed.get(QUESTION_FIELD)
    return (
        answer.strip() if isinstance(answer, str) else "",
        question.strip() if isinstance(question, str) else "",
    )


def _is_pleasantry(question: str) -> bool:
    """Filler the schema description rules out but cannot enforce."""
    lowered = question.lower()
    return any(
        cue in lowered
        for cue in (
            "let me know if",
            "feel free to ask",
            "any other questions",
            "anything else",
            "궁금한 점이 있으면",
            "다른 질문이 있",
        )
    )


def compose(answer: str, question: str) -> str:
    """One user-facing message: the answer, then the question on its own line."""
    if not answer:
        return ""
    if not question or _is_pleasantry(question):
        return answer
    if question.rstrip()[-1] not in "?？":
        question = question.rstrip(" .") + "?"
    return f"{answer.rstrip()}\n\n{question}"


# The contract is applied where the wall it breaks actually stands. Asked in the system
# prompt to end with a question, the model already complies in Korean and refuses in
# English. Measured across languages: English came out a tie overall while gaining 0.070 of
# context awareness, and non-English lost 0.069 — the structured constraint compresses an
# answer, and it compresses a non-English answer harder. The held-out evaluation is Korean,
# so that loss is not one to carry for a gain that only lands in English.
def applies_to(text: str) -> bool:
    """Whether this turn should be answered under the commit contract."""
    letters = [character for character in text or "" if character.isalpha()]
    if len(letters) < 12:
        return False
    return sum(1 for character in letters if character.isascii()) / len(letters) > 0.9

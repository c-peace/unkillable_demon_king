from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.coverage import AnswerContract, extract_contract


HIGH_RISK_PATTERN = re.compile(
    r"(용량|복용량|과다복용|상호작용|금기|임신|수유|소아|영아|신생아|"
    r"흉통|호흡곤란|의식.{0,4}(저하|없)|자살|자해|응급|출혈|아나필락시스|"
    r"dose|dosage|overdose|interaction|contraindication|pregnan|breastfeed|"
    r"pediatric|infant|newborn|chest pain|shortness of breath|suicid|self-harm|"
    r"emergency|anaphylaxis|severe bleeding)",
    re.IGNORECASE,
)

FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b(it|that|those|this|them|they)\b|"
    r"(그|이|저)\s?(약|기준|경우|내용|치료|검사|문구|답변|수치|증상|상태)|"
    r"해당|앞서 말한|previous|same one|same medication",
    re.IGNORECASE,
)

ELLIPTICAL_FOLLOW_UP_PATTERN = re.compile(
    r"계속|그대로|이어서|이후에도|"
    r"먹어도\s*(?:되|괜찮)|복용해도\s*(?:되|괜찮)|"
    r"중단해도|바꿔도|줄여도|늘려도|"
    r"\b(?:still|continue|keep taking|stop taking|same dose)\b",
    re.IGNORECASE,
)

CORRECTION_PATTERN = re.compile(
    r"정정|아니고|아니에요|아니야|말한 건|제가 .*?뜻한 건|"
    r"actually|correction|i meant|not that|instead",
    re.IGNORECASE,
)

NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|without|denies|denied|none)\b|"
    r"없|아니|부인|해당 없",
    re.IGNORECASE,
)

FORMAT_PATTERN = re.compile(
    r"rewrite|reword|shorten|summari[sz]e|translate|draft|message|email|letter|note|"
    r"soap|bullet|table|template|"
    r"번역|요약|정리|다듬|수정|다시 써|문구|메시지|문자|편지|노트|"
    r"(?:^|[\s(])표(?:[\s)!?,.]|$)",
    re.IGNORECASE,
)

STRICT_FORMAT_PATTERN = re.compile(
    r"\b(?:json|csv|soap|table|template|schema|bullet(?:s| point)?)\b|"
    r"필드|항목|양식|템플릿|구조화\s*형식|"
    r"(?:^|[\s(])표(?:[\s)!?,.]|$)",
    re.IGNORECASE,
)

QUESTION_PATTERN = re.compile(r"\?|？|나요|까요|인가요|되나요|can|could|should|what|how|why", re.IGNORECASE)

AGE_PATTERN = re.compile(
    r"\b(\d{1,3})\s*(?:years?\s*old|yo|y/o|개월|달|살)\b",
    re.IGNORECASE,
)
WEIGHT_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*(?:kg|킬로)\b", re.IGNORECASE)
TEMPERATURE_PATTERN = re.compile(r"\b\d{2}(?:\.\d+)?\s*(?:c|℃|도)\b", re.IGNORECASE)
DOSE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|정|캡슐)\b", re.IGNORECASE)
DRUG_FORM_PATTERN = re.compile(r"[가-힣]{2,}(?:정|캡슐|시럽|주사)", re.IGNORECASE)
CHILD_PATTERN = re.compile(r"소아|영아|신생아|아이|아기|child|kid|infant|newborn|pediatric", re.IGNORECASE)
FEVER_PATTERN = re.compile(r"열|fever|febrile", re.IGNORECASE)
DOSING_PATTERN = re.compile(r"용량|복용량|dose|dosage|몇 mg|얼마나", re.IGNORECASE)
PREGNANCY_PATTERN = re.compile(r"임신|수유|pregnan|breastfeed", re.IGNORECASE)
NEGATED_RISK_PATTERN = re.compile(
    r"(?:임신|수유|자살(?:\s*생각)?|자해(?:\s*생각)?|흉통|호흡곤란)"
    r"(?:은|는|이|가)?\s*(?:아니|없|하지\s*않)|"
    r"(?:not|no|denies?|without)\s+(?:pregnan\w*|breastfeed\w*|suicid\w*|"
    r"self-harm|chest pain|shortness of breath)",
    re.IGNORECASE,
)

MEDICATION_CANDIDATE_PATTERNS = (
    re.compile(
        r"([A-Za-z][A-Za-z0-9-]{2,}|[가-힣]{2,})\s*(?:의\s*)?"
        r"(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|정|캡슐)|dose|dosage|용량|복용량)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:dose|dosage)\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9-]{2,})", re.IGNORECASE),
    re.compile(
        r"(?:take|taking|use|using|prescribed|on)\s+(?:a|an|the|my\s+)?"
        r"([A-Za-z][A-Za-z0-9-]{2,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Za-z][A-Za-z0-9-]{2,}|[가-힣]{2,})(?:을|를)?\s*"
        r"(?:복용|먹고|먹는|투여|사용)"
    ),
)

_GENERIC_MEDICATION_WORDS = frozenset(
    {
        "what",
        "which",
        "this",
        "that",
        "medicine",
        "medication",
        "drug",
        "dose",
        "dosage",
        "safe",
        "recommended",
        "daily",
        "usual",
        "normal",
        "maximum",
        "minimum",
        "appropriate",
        "correct",
        "standard",
        "약",
        "약물",
        "의약품",
        "적절한",
        "권장",
        "안전한",
        "정확한",
        "용량",
        "복용량",
        "하루",
        "일일",
        "최대",
        "최소",
        "적정",
        "일반적인",
    }
)


def _risk_is_explicitly_negated(text: str) -> bool:
    if not NEGATION_PATTERN.search(text):
        return False
    return bool(NEGATED_RISK_PATTERN.search(text))


def _has_active_risk(text: str) -> bool:
    remaining = NEGATED_RISK_PATTERN.sub("", text)
    return bool(HIGH_RISK_PATTERN.search(remaining))


def _has_named_medication(turns: Iterable[str]) -> bool:
    text = "\n".join(turns)
    if DRUG_FORM_PATTERN.search(text):
        return True
    for pattern in MEDICATION_CANDIDATE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(1).strip().lower()
            if candidate not in _GENERIC_MEDICATION_WORDS:
                return True
    return False


@dataclass(frozen=True, slots=True)
class ConversationState:
    latest_user_turn: str
    prior_user_turns: tuple[str, ...]
    current_intent: str
    requested_language: str
    requested_format: str
    strict_output_requested: bool
    explicit_requirements: tuple[str, ...]
    source_sensitive_signals: tuple[str, ...]
    active_risk_signals: tuple[str, ...]
    safety_facts: tuple[str, ...]
    corrections: tuple[str, ...]
    unresolved_references: tuple[str, ...]
    decision_critical_missing_facts: tuple[str, ...]
    answer_contract: AnswerContract

    @property
    def has_follow_up_reference(self) -> bool:
        return bool(self.unresolved_references)

    @property
    def is_text_operation(self) -> bool:
        return self.requested_format != "plain_answer"

    def retrieval_context(self) -> str:
        if not self.has_follow_up_reference or not self.prior_user_turns:
            return self.latest_user_turn
        prior = "\n".join(self.prior_user_turns[-2:])
        return f"{prior}\n{self.latest_user_turn}".strip()


@dataclass(frozen=True, slots=True)
class CompiledConversation:
    messages: tuple[dict[str, Any], ...]
    latest_user_text: str
    history_hash: str
    case_packet: str
    is_high_risk: bool
    state: ConversationState

    def generation_messages(self, representation: str) -> list[dict[str, Any]]:
        if representation == "native":
            return [dict(message) for message in self.messages]
        return [
            {
                "role": "user",
                "content": (
                    "The following is the complete role-labelled conversation. "
                    "Answer the final user turn while preserving all prior facts and corrections.\n\n"
                    f"{self.case_packet}"
                ),
            }
        ]

    def recovery_messages(
        self,
        *,
        requirements: tuple[str, ...],
        lead_system_messages: Iterable[dict[str, Any]],
        grounding_payloads: Iterable[str] = (),
        recent_turns: int = 4,
    ) -> list[dict[str, Any]]:
        lead = [dict(message) for message in lead_system_messages]
        grounding = [payload for payload in grounding_payloads if payload]
        recent = self.messages[-recent_turns:] if len(self.messages) > recent_turns else self.messages
        recent_packet = "\n\n".join(
            f"[TURN {index} | {str(message.get('role', 'unknown')).upper()}]\n{_message_text(message)}"
            for index, message in enumerate(recent, start=max(1, len(self.messages) - len(recent) + 1))
        )
        packet = {
            "current_user_intent": self.state.latest_user_turn,
            "safety_facts": list(self.state.safety_facts),
            "corrections_and_negations": list(self.state.corrections),
            "unresolved_references": list(self.state.unresolved_references),
            "answer_requirements": list(requirements),
            "grounding_payloads": grounding,
            "recent_turns": recent_packet,
            "instruction": (
                "This is a compact recovery packet for the same conversation. Preserve "
                "the safety facts, corrections, negations, unanswered user asks, and all "
                "grounding payloads. Cite only evidence present in grounding_payloads."
            ),
        }
        return [
            *lead,
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _requested_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def _requested_format(text: str) -> str:
    return "text_operation" if FORMAT_PATTERN.search(text) else "plain_answer"


def _current_intent(text: str) -> str:
    return "text_operation" if FORMAT_PATTERN.search(text) else "medical_answer"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _extract_state(copied: tuple[dict[str, Any], ...], latest_user: str) -> ConversationState:
    user_turns = tuple(
        _message_text(message).strip()
        for message in copied
        if message.get("role") == "user" and _message_text(message).strip()
    )
    prior_user_turns = user_turns[:-1]
    unresolved_references = ()
    explicit_follow_up = bool(latest_user and FOLLOW_UP_REFERENCE_PATTERN.search(latest_user))
    elliptical_follow_up = bool(
        latest_user
        and prior_user_turns
        and QUESTION_PATTERN.search(latest_user)
        and ELLIPTICAL_FOLLOW_UP_PATTERN.search(latest_user)
    )
    if explicit_follow_up or elliptical_follow_up:
        unresolved_references = (latest_user,)

    corrections = _unique(
        text
        for text in user_turns
        if CORRECTION_PATTERN.search(text) or NEGATION_PATTERN.search(text)
    )
    safety_facts = _unique(
        text
        for text in user_turns
        if (
            HIGH_RISK_PATTERN.search(text)
            or DOSE_PATTERN.search(text)
            or AGE_PATTERN.search(text)
            or WEIGHT_PATTERN.search(text)
            or TEMPERATURE_PATTERN.search(text)
            or PREGNANCY_PATTERN.search(text)
            or NEGATION_PATTERN.search(text)
        )
    )
    latest_active_risk = (
        _has_active_risk(latest_user)
        and not FORMAT_PATTERN.search(latest_user)
    )
    prior_active_risk = bool(
        unresolved_references
        and not FORMAT_PATTERN.search(latest_user)
        and not _risk_is_explicitly_negated(latest_user)
        and any(_has_active_risk(text) for text in prior_user_turns[-2:])
    )
    active_risk_signals = _unique(
        (
            "high_risk_latest",
            "follow_up_reference",
        )[index]
        for index, condition in enumerate(
            (
                latest_active_risk,
                prior_active_risk,
            )
        )
        if condition
    )
    routing_text = latest_user
    if unresolved_references and prior_user_turns:
        routing_text = "\n".join((*prior_user_turns[-2:], latest_user))
    source_sensitive_signals = _unique(
        signal
        for signal, pattern in (
            (
                "guideline",
                re.compile(
                    r"guideline|가이드라인|권고|recommend|목표|target|threshold|"
                    r"기준치|임상\s*기준|진단\s*기준|권고\s*기준",
                    re.IGNORECASE,
                ),
            ),
            ("drug_safety", re.compile(r"상호작용|금기|contraindication|interaction", re.IGNORECASE)),
            ("regulatory", re.compile(r"허가|승인|approval|indication", re.IGNORECASE)),
            ("coverage", re.compile(r"급여|reimbursement|coverage", re.IGNORECASE)),
            ("coding", re.compile(r"코드|kcd|code", re.IGNORECASE)),
            ("law", re.compile(r"법|law|statute", re.IGNORECASE)),
            ("research", re.compile(r"논문|연구|trial|study|meta-analysis", re.IGNORECASE)),
        )
        if pattern.search(routing_text)
    )
    missing_facts: list[str] = []
    if unresolved_references and not prior_user_turns:
        missing_facts.append("reference_target")
    if DOSING_PATTERN.search(latest_user) and not _has_named_medication(user_turns):
        missing_facts.append("medication_name")
    if CHILD_PATTERN.search(latest_user) and FEVER_PATTERN.search(latest_user):
        if not AGE_PATTERN.search("\n".join(user_turns)):
            missing_facts.append("child_age")
        if not WEIGHT_PATTERN.search("\n".join(user_turns)):
            missing_facts.append("child_weight")
        if not TEMPERATURE_PATTERN.search("\n".join(user_turns)):
            missing_facts.append("temperature")
    contract = extract_contract(latest_user)
    return ConversationState(
        latest_user_turn=latest_user,
        prior_user_turns=prior_user_turns,
        current_intent=_current_intent(latest_user),
        requested_language=_requested_language(latest_user),
        requested_format=_requested_format(latest_user),
        strict_output_requested=bool(STRICT_FORMAT_PATTERN.search(latest_user)),
        explicit_requirements=contract.requirements,
        source_sensitive_signals=source_sensitive_signals,
        active_risk_signals=active_risk_signals,
        safety_facts=safety_facts,
        corrections=corrections,
        unresolved_references=unresolved_references,
        decision_critical_missing_facts=tuple(missing_facts),
        answer_contract=contract,
    )


def compile_conversation(messages: Iterable[dict[str, Any]]) -> CompiledConversation:
    copied = tuple(dict(message) for message in messages)
    latest_user = next(
        (_message_text(message) for message in reversed(copied) if message.get("role") == "user"),
        "",
    )
    canonical = json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    history_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    turns = []
    for index, message in enumerate(copied, start=1):
        role = str(message.get("role", "unknown")).upper()
        turns.append(f"[TURN {index} | {role}]\n{_message_text(message)}")
    packet = "\n\n".join(turns)
    state = _extract_state(copied, latest_user)
    return CompiledConversation(
        messages=copied,
        latest_user_text=latest_user,
        history_hash=history_hash,
        case_packet=packet,
        is_high_risk=bool(state.active_risk_signals),
        state=state,
    )

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable


class AssertionState(StrEnum):
    ASSERTED = "asserted"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    CORRECTED = "corrected"
    HISTORICAL = "historical"


@dataclass(frozen=True, slots=True)
class TurnProvenance:
    turn_index: int
    role: str
    span_hash: str


@dataclass(frozen=True, slots=True)
class ClinicalFact:
    kind: str
    value: str
    subject: str
    assertion: AssertionState
    provenance: tuple[TurnProvenance, ...]


@dataclass(frozen=True, slots=True)
class SymptomFact:
    name: str
    assertion: AssertionState
    onset: str = ""
    duration: str = ""
    severity: str = ""
    trajectory: str = ""
    provenance: tuple[TurnProvenance, ...] = ()


@dataclass(frozen=True, slots=True)
class MedicationFact:
    name: str
    dose: str = ""
    route: str = ""
    frequency: str = ""
    action: str = "mentioned"
    assertion: AssertionState = AssertionState.ASSERTED
    provenance: tuple[TurnProvenance, ...] = ()


@dataclass(frozen=True, slots=True)
class InteractionState:
    user_role: str = "patient_or_general_user"
    audience: str = "user"
    language: str = "unknown"
    requested_format: str = "plain_answer"
    jurisdiction: str = ""
    resource_constraints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClinicalState:
    facts: tuple[ClinicalFact, ...]
    symptoms: tuple[SymptomFact, ...]
    medications: tuple[MedicationFact, ...]
    interaction: InteractionState
    active_symptom_names: tuple[str, ...]
    hard_risk_signals: tuple[str, ...]
    conflicts: tuple[str, ...]
    unresolved_references: tuple[str, ...]
    needs_semantic_enrichment: bool

    @property
    def named_medications(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.name for item in self.medications if item.name))

    def facts_of_kind(self, kind: str) -> tuple[ClinicalFact, ...]:
        return tuple(item for item in self.facts if item.kind == kind)

    def latest_fact(self, kind: str, *, subject: str = "patient") -> ClinicalFact | None:
        candidates = [
            item for item in self.facts if item.kind == kind and item.subject == subject
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: max(ref.turn_index for ref in item.provenance),
        )

    def compact_record(self) -> dict[str, Any]:
        """Return model context without adding hidden facts or raw spans."""
        return {
            "facts": [
                {
                    "kind": item.kind,
                    "value": item.value,
                    "subject": item.subject,
                    "assertion": item.assertion.value,
                    "turns": [ref.turn_index for ref in item.provenance],
                }
                for item in self.facts
            ],
            "symptoms": [
                {
                    "name": item.name,
                    "assertion": item.assertion.value,
                    "onset": item.onset,
                    "duration": item.duration,
                    "severity": item.severity,
                    "trajectory": item.trajectory,
                    "turns": [ref.turn_index for ref in item.provenance],
                }
                for item in self.symptoms
            ],
            "medications": [
                {
                    "name": item.name,
                    "dose": item.dose,
                    "action": item.action,
                    "assertion": item.assertion.value,
                    "turns": [ref.turn_index for ref in item.provenance],
                }
                for item in self.medications
            ],
            "interaction": {
                "user_role": self.interaction.user_role,
                "audience": self.interaction.audience,
                "language": self.interaction.language,
                "requested_format": self.interaction.requested_format,
                "jurisdiction": self.interaction.jurisdiction,
            },
            "hard_risk_signals": list(self.hard_risk_signals),
            "active_symptom_names": list(self.active_symptom_names),
            "conflicts": list(self.conflicts),
            "unresolved_references": list(self.unresolved_references),
        }


_AGE = re.compile(r"\b(\d{1,3})\s*(years?\s*old|yo|y/o|개월|달|살)\b", re.IGNORECASE)
_WEIGHT = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(kg|킬로)\b", re.IGNORECASE)
_TEMPERATURE = re.compile(r"\b(\d{2}(?:\.\d+)?)\s*(c|℃|도)\b", re.IGNORECASE)
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|정|캡슐)\b", re.IGNORECASE)
_UNCERTAIN = re.compile(r"아마|같아요|모르|가능성|maybe|perhaps|unsure|not sure", re.IGNORECASE)
_HISTORICAL = re.compile(r"과거|예전에|병력|history of|previously|used to", re.IGNORECASE)
_CORRECTION = re.compile(
    r"정정|아니고|아니에요|아니야|말한 건|actually|correction|i meant|not that|instead",
    re.IGNORECASE,
)
_PREGNANCY = re.compile(r"임신|pregnan", re.IGNORECASE)
_BREASTFEEDING = re.compile(r"수유|breastfeed|lactat", re.IGNORECASE)
_NEGATED_PREGNANCY = re.compile(
    r"(?:임신|수유)(?:은|는|이|가)?\s*(?:아니|아님|없)|"
    r"(?:not|no|without)\s+(?:pregnan\w*|breastfeed\w*|lactat\w*)",
    re.IGNORECASE,
)
_ALLERGY = re.compile(
    r"(?:알레르기|allerg(?:y|ic))(?:가|는|은|:)?\s*(?:to\s+)?([A-Za-z가-힣0-9-]{2,})?",
    re.IGNORECASE,
)

_MEDICATION_PATTERNS = (
    re.compile(
        r"\b([A-Za-z][A-Za-z0-9-]{2,}|[가-힣]{2,}(?:정|캡슐|시럽|주사)|[가-힣]{2,})\s*"
        r"(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|정|캡슐))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:dose|dosage)\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9-]{2,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:take|taking|use|using|prescribed|on)\s+(?:a|an|the|my\s+)?"
        r"([A-Za-z][A-Za-z0-9-]{2,})(?=\s*(?:\d|with|while|during|for|[,.?!]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are)\s+([A-Za-z][A-Za-z0-9-]{2,})\s+"
        r"(?:safe|contraindicated|approved|indicated)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Za-z][A-Za-z0-9-]{2,}|[가-힣]{2,})(?:을|를)?\s*"
        r"(?:복용|먹고|먹는|투여|처방|사용)",
        re.IGNORECASE,
    ),
)

_NOT_MEDICATIONS = frozenset(
    {
        "fever",
        "pain",
        "headache",
        "nausea",
        "rash",
        "cough",
        "cold",
        "pregnancy",
        "breastfeeding",
        "medicine",
        "medication",
        "drug",
        "dose",
        "dosage",
        "what",
        "which",
        "this",
        "that",
        "safe",
        "recommended",
        "열",
        "통증",
        "두통",
        "감기",
        "기침",
        "발진",
        "약",
        "약물",
        "의약품",
    }
)

_SYMPTOM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chest_pain", re.compile(r"흉통|가슴\s*(?:통증|아픔)|chest pain", re.IGNORECASE)),
    ("dyspnea", re.compile(r"호흡곤란|숨이\s*(?:차|막)|shortness of breath|difficulty breathing", re.IGNORECASE)),
    ("one_sided_weakness", re.compile(r"한쪽|편측|one[- ]sided|unilateral", re.IGNORECASE)),
    ("weakness", re.compile(r"힘이\s*빠|마비|위약|weakness|numb", re.IGNORECASE)),
    ("speech_change", re.compile(r"말이\s*(?:어눌|안\s*나)|언어장애|slurred speech|speech difficulty", re.IGNORECASE)),
    ("seizure", re.compile(r"경련|발작|seizure|convulsion", re.IGNORECASE)),
    ("altered_consciousness", re.compile(r"의식.{0,5}(?:저하|없|안\s*돌)|깨우기\s*어려|unconscious|difficult to wake", re.IGNORECASE)),
    ("airway_swelling", re.compile(r"(?:입술|혀|목).{0,5}(?:붓|부종)|lip swelling|tongue swelling|throat swelling", re.IGNORECASE)),
    ("gi_bleeding", re.compile(r"토혈|검은\s*변|혈변|vomiting blood|black stool|melena", re.IGNORECASE)),
    ("severe_dizziness", re.compile(r"심한\s*어지|쓰러|실신|faint|syncope|severe dizziness", re.IGNORECASE)),
    ("suicidality", re.compile(r"자살|자해|suicid|self[- ]harm", re.IGNORECASE)),
    ("overdose", re.compile(r"과다복용|overdose|too many pills", re.IGNORECASE)),
)

_NEGATED_BEFORE = re.compile(
    r"\b(?:no|not|without|denies?)\b\s*$",
    re.IGNORECASE,
)
_NEGATED_AFTER = re.compile(
    r"^\s*(?:은|는|이|가|을|를)?\s*(?:없|아니|부인|하지\s*않|is\s+not|are\s+not)",
    re.IGNORECASE,
)
_SUDDEN = re.compile(r"갑자기|돌연|sudden|suddenly|acute onset", re.IGNORECASE)
_SEVERE = re.compile(r"심한|극심|매우|severe|worst|intense", re.IGNORECASE)

_FOLLOW_UP = re.compile(
    r"\b(?:it|that|those|this|them|same one|same medication)\b|"
    r"(?:그|이|저)\s?(?:약|기준|경우|치료|검사|증상|상태)|해당|앞서 말한|계속|그대로",
    re.IGNORECASE,
)
_NEGATION_CUE = re.compile(r"없|아니|부인|해소|사라|\b(?:no|not|without|denies?|resolved|gone)\b", re.I)


def _hash_span(turn_index: int, value: str) -> str:
    return hashlib.sha256(f"{turn_index}:{value}".encode("utf-8")).hexdigest()[:12]


def _provenance(turn_index: int, role: str, value: str) -> tuple[TurnProvenance, ...]:
    return (TurnProvenance(turn_index, role, _hash_span(turn_index, value)),)


def _assertion(text: str, match_text: str) -> AssertionState:
    start = text.lower().find(match_text.lower())
    start = max(0, start)
    end = start + len(match_text)
    before = text[max(0, start - 24) : start]
    after = text[end : end + 24]
    window = text[max(0, start - 24) : end + 24]
    if _NEGATED_BEFORE.search(before) or _NEGATED_AFTER.search(after):
        return AssertionState.NEGATED
    if _CORRECTION.search(text):
        return AssertionState.CORRECTED
    if _UNCERTAIN.search(window):
        return AssertionState.UNCERTAIN
    if _HISTORICAL.search(window):
        return AssertionState.HISTORICAL
    return AssertionState.ASSERTED


def _unique_by(items: Iterable[Any], key) -> tuple[Any, ...]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        marker = key(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return tuple(result)


def _medications(text: str, turn_index: int, role: str) -> tuple[MedicationFact, ...]:
    found: list[MedicationFact] = []
    for pattern in _MEDICATION_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip().lower()
            if name in _NOT_MEDICATIONS:
                continue
            dose = ""
            if match.lastindex and match.lastindex >= 2 and match.group(2):
                dose = match.group(2).strip()
            action = "taking" if re.search(r"take|복용|먹|투여|사용", match.group(0), re.I) else "mentioned"
            found.append(
                MedicationFact(
                    name=name,
                    dose=dose,
                    action=action,
                    assertion=_assertion(text, match.group(0)),
                    provenance=_provenance(turn_index, role, match.group(0)),
                )
            )
    return _unique_by(found, lambda item: (item.name, item.dose, item.provenance[0].turn_index))


def _hard_risks(text: str, active_symptoms: set[str]) -> tuple[str, ...]:
    signals: list[str] = []
    if (
        _SUDDEN.search(text)
        and "weakness" in active_symptoms
        and ("one_sided_weakness" in active_symptoms or "speech_change" in active_symptoms)
    ):
        signals.append("possible_stroke")
    if "seizure" in active_symptoms and "altered_consciousness" in active_symptoms:
        signals.append("seizure_with_impaired_recovery")
    elif "seizure" in active_symptoms:
        signals.append("active_seizure")
    if "dyspnea" in active_symptoms and "airway_swelling" in active_symptoms:
        signals.append("possible_anaphylaxis")
    if "gi_bleeding" in active_symptoms and "severe_dizziness" in active_symptoms:
        signals.append("possible_major_bleeding")
    if "chest_pain" in active_symptoms and (
        "dyspnea" in active_symptoms or _SEVERE.search(text)
    ):
        signals.append("cardiopulmonary_emergency")
    if "suicidality" in active_symptoms:
        signals.append("self_harm_risk")
    if "overdose" in active_symptoms:
        signals.append("overdose")
    return tuple(dict.fromkeys(signals))


def compile_clinical_state(
    messages: Iterable[dict[str, Any]],
    *,
    language: str,
    requested_format: str,
) -> ClinicalState:
    copied = tuple(messages)
    facts: list[ClinicalFact] = []
    symptoms: list[SymptomFact] = []
    medications: list[MedicationFact] = []
    conflicts: list[str] = []
    unresolved: list[str] = []
    hard_risks_by_turn: dict[int, tuple[str, ...]] = {}
    active_symptoms_by_turn: dict[int, tuple[str, ...]] = {}
    user_turn_text: dict[int, str] = {}
    user_role = "patient_or_general_user"
    jurisdiction = ""

    for turn_index, message in enumerate(copied, start=1):
        role = str(message.get("role", "unknown"))
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if not text:
            continue

        if role == "user" and re.search(r"\b(?:clinician|doctor|physician|nurse|pharmacist)\b|의사|간호사|약사|임상의", text, re.I):
            user_role = "clinician"
        if role == "user":
            user_turn_text[turn_index] = text
        if re.search(r"\b(?:korea|korean|south korea)\b|한국|국내", text, re.I):
            jurisdiction = "KR"
        elif re.search(r"\b(?:united states|u\.s\.|usa)\b|미국", text, re.I):
            jurisdiction = "US"

        for kind, pattern in (("age", _AGE), ("weight", _WEIGHT), ("temperature", _TEMPERATURE)):
            for match in pattern.finditer(text):
                facts.append(
                    ClinicalFact(
                        kind=kind,
                        value=match.group(0),
                        subject="patient",
                        assertion=_assertion(text, match.group(0)),
                        provenance=_provenance(turn_index, role, match.group(0)),
                    )
                )

        for match in _ALLERGY.finditer(text):
            value = (match.group(1) or "unspecified").strip().lower()
            facts.append(
                ClinicalFact(
                    kind="allergy",
                    value=value,
                    subject="patient",
                    assertion=_assertion(text, match.group(0)),
                    provenance=_provenance(turn_index, role, match.group(0)),
                )
            )

        pregnancy_match = _PREGNANCY.search(text)
        if pregnancy_match is not None:
            assertion = (
                AssertionState.NEGATED
                if _NEGATED_PREGNANCY.search(text)
                else _assertion(text, pregnancy_match.group(0))
            )
            facts.append(
                ClinicalFact(
                    kind="pregnancy",
                    value="pregnant",
                    subject="patient",
                    assertion=assertion,
                    provenance=_provenance(turn_index, role, "pregnancy"),
                )
            )
        breastfeeding_match = _BREASTFEEDING.search(text)
        if breastfeeding_match is not None:
            assertion = (
                AssertionState.NEGATED
                if _NEGATED_PREGNANCY.search(text)
                else _assertion(text, breastfeeding_match.group(0))
            )
            facts.append(
                ClinicalFact(
                    kind="breastfeeding",
                    value="breastfeeding",
                    subject="patient",
                    assertion=assertion,
                    provenance=_provenance(turn_index, role, "breastfeeding"),
                )
            )

        medications.extend(_medications(text, turn_index, role))

        active_names: set[str] = set()
        for name, pattern in _SYMPTOM_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            assertion = _assertion(text, match.group(0))
            symptom = SymptomFact(
                name=name,
                assertion=assertion,
                onset="sudden" if _SUDDEN.search(text) else "",
                severity="severe" if _SEVERE.search(text) else "",
                provenance=_provenance(turn_index, role, match.group(0)),
            )
            symptoms.append(symptom)
            if role == "user" and assertion not in {AssertionState.NEGATED, AssertionState.HISTORICAL}:
                active_names.add(name)
        if role == "user" and requested_format != "text_operation":
            hard_risks_by_turn[turn_index] = _hard_risks(text, active_names)
            active_symptoms_by_turn[turn_index] = tuple(sorted(active_names))

        if role == "user" and _FOLLOW_UP.search(text):
            unresolved.append(text)
        if role == "user" and _CORRECTION.search(text):
            conflicts.append(f"correction_at_turn_{turn_index}")

    facts_tuple = _unique_by(
        facts,
        lambda item: (item.kind, item.value, item.assertion.value, item.provenance[0].turn_index),
    )
    symptoms_tuple = _unique_by(
        symptoms,
        lambda item: (item.name, item.assertion.value, item.provenance[0].turn_index),
    )
    medications_tuple = _unique_by(
        medications,
        lambda item: (item.name, item.dose, item.assertion.value, item.provenance[0].turn_index),
    )
    if len(user_turn_text) == 1 and (medications_tuple or symptoms_tuple or facts_tuple):
        unresolved = []
    needs_enrichment = bool(
        len(copied) > 8
        or conflicts
        or (unresolved and not medications_tuple)
        or any(item.assertion == AssertionState.UNCERTAIN for item in facts_tuple)
    )
    active_hard_risks: tuple[str, ...] = ()
    active_symptoms: tuple[str, ...] = ()
    if user_turn_text and requested_format != "text_operation":
        latest_turn = max(user_turn_text)
        latest_text = user_turn_text[latest_turn]
        active_hard_risks = hard_risks_by_turn.get(latest_turn, ())
        active_symptoms = active_symptoms_by_turn.get(latest_turn, ())
        if (
            not active_hard_risks
            and _FOLLOW_UP.search(latest_text)
            and not _NEGATION_CUE.search(latest_text)
        ):
            for turn_index in sorted(hard_risks_by_turn, reverse=True):
                if turn_index >= latest_turn:
                    continue
                if hard_risks_by_turn[turn_index]:
                    active_hard_risks = hard_risks_by_turn[turn_index]
                    break
        if (
            not active_symptoms
            and _FOLLOW_UP.search(latest_text)
            and not _NEGATION_CUE.search(latest_text)
        ):
            for turn_index in sorted(active_symptoms_by_turn, reverse=True):
                if turn_index >= latest_turn:
                    continue
                if active_symptoms_by_turn[turn_index]:
                    active_symptoms = active_symptoms_by_turn[turn_index]
                    break
    return ClinicalState(
        facts=facts_tuple,
        symptoms=symptoms_tuple,
        medications=medications_tuple,
        interaction=InteractionState(
            user_role=user_role,
            language=language,
            requested_format=requested_format,
            jurisdiction=jurisdiction,
        ),
        active_symptom_names=active_symptoms,
        hard_risk_signals=tuple(dict.fromkeys(active_hard_risks)),
        conflicts=tuple(dict.fromkeys(conflicts)),
        unresolved_references=tuple(dict.fromkeys(unresolved)),
        needs_semantic_enrichment=needs_enrichment,
    )

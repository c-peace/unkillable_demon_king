"""Decide whether current or official external evidence is required before generation.

Retrieval is useful for mutable external state such as guidelines, labels, approvals,
reimbursement, codes, statutes, and published research. Ordinary explanation, triage, and
self-care can usually stay on the memory lane. The state planner supplies resolved follow-up
context; the regex router remains a conservative signal and legacy fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.conversation import ConversationState
from app.evidence.routing import SourceRouter


# Domains that stand for an authoritative external record. A question reaches one of these
# only when answering it well means consulting something that can change without the model
# changing — which is exactly when retrieval is worth a round trip.
_STATE_DEPENDENT = frozenset(
    {
        "drug",          # approvals, indications, dosing as licensed
        "drug_safety",   # interactions, contraindications, adverse reactions
        "guideline",     # recommendations and targets as currently published
        "hira",          # Korean reimbursement rules and drug pricing
        "coding",        # KCD / disease coding
        "law",           # statutes and enforcement decrees
        "research",      # the literature, trials, systematic reviews
        "faers",         # adverse event signals
    }
)


@dataclass(frozen=True, slots=True)
class Admission:
    """Whether this turn may acquire evidence, and on whose authority."""

    domains: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return bool(self.domains)

    @property
    def lane(self) -> str:
        return "evidence" if self.admitted else "memory"


def _context_text(state: ConversationState) -> str:
    parts = [state.latest_user_turn]
    if state.has_follow_up_reference:
        parts.extend(state.prior_user_turns[-2:])
    return "\n".join(part for part in parts if part).strip()


def admit(user_text: str | ConversationState, router: SourceRouter | None = None) -> Admission:
    """Decide whether evidence may be acquired for this turn."""
    if isinstance(user_text, ConversationState):
        question = _context_text(user_text)
    else:
        question = user_text
    if not question or not question.strip():
        return Admission(())
    resolved = router or SourceRouter("route")
    domains = tuple(d for d in resolved.routes(question) if d in _STATE_DEPENDENT)
    return Admission(domains)

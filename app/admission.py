"""Deciding whether evidence should be acquired at all, before any tool is offered.

The model cannot make this call. Choosing to search looks free from inside a single turn,
so a model asked "would evidence help here?" will nearly always say yes — it has no way to
know that the search costs a round trip against a shared endpoint, and no way to know what
happens to its own answer afterwards. Measured on the same 150 conversations with retrieval
forced on and forced off:

    router recognised a domain    n=54   on 0.480   off 0.481   Δ +0.001
    router recognised nothing     n=96   on 0.513   off 0.556   Δ +0.043

Where the corpus has something to say, searching is free. Where it does not, searching
costs 0.043 — because thin, off-topic evidence does not sit quietly beside the answer, it
displaces knowledge the model already had.

An earlier attempt closed this off inside the retrieval controller, and lost 0.034 rather
than gaining anything. By then the model had already called the tool and watched it come
back empty, and an answer written after a failed search is a smaller answer. So the
decision has to be made before the generation request is built: on the memory lane the
model is never shown `retrieve_relevant_content`, and nothing in the prompt records that a
search was considered. That path has to be indistinguishable, from the model's side, from
one where retrieval was never part of the system.

The test is not whether a question is clinical. "Why does hypertension develop?" is a
clinical question and belongs on the memory lane; "what blood pressure target do the
current guidelines give for CKD?" is the same subject and does not. The test is whether
answering correctly needs current or official state that does not live in the model's
parameters — an approval, a reimbursement rule, a statute, a label, a published figure.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def admit(user_text: str, router: SourceRouter | None = None) -> Admission:
    """Decide the lane for one turn from the question alone."""
    if not user_text or not user_text.strip():
        return Admission(())
    resolved = router or SourceRouter("route")
    domains = tuple(d for d in resolved.routes(user_text) if d in _STATE_DEPENDENT)
    return Admission(domains)

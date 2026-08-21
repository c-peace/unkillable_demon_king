"""Committing or discarding a retrieval, once its result is known.

The admission gate stops a search that should never have started. This is the same idea
one step further in: a search that did start, and came back thin, should not be allowed to
reach the generation stage either. Measured on the same 150 conversations, answers built
on no evidence scored 0.604, on one or two records 0.431, and on three or more 0.217 —
thin evidence does not sit quietly beside an answer, it displaces what the model knew. And
a version that let the model watch a search fail lost 0.034 against one where the search
never happened.

So retrieval is treated as a speculative side effect. The harness runs it, inspects what
came back, and either commits it to the generation or rolls it back and generates exactly
as it would have with no retrieval at all — no evidence block, no note, no trace, nothing
for the model to read as "we looked and found nothing".

Rolling back is not always right. For a question whose answer *is* an external record —
what the reimbursement rule says, which code applies, what the statute requires — the
model's own knowledge is the weaker source, and a partial record still beats guessing. So
those domains keep partial evidence; the ones the model can answer from training do not.
"""
from __future__ import annotations


# Domains where the answer is a current authoritative record rather than clinical
# knowledge. Guessing at a reimbursement rule is worse than citing an incomplete one.
_EXTERNAL_STATE = frozenset({"hira", "coding", "law"})

HARD = "hard"
SOFT = "soft"
ROLLBACK = "rollback"

# What each commit mode carries into generation. Fewer records than the retrieval budget
# allows, because more evidence measured worse than less.
COMMIT_BUDGET = {
    HARD: (3, 4_000),
    SOFT: (2, 2_000),
}


def evidence_class(domains) -> str:
    """Whether this question is answered from a record or from medical knowledge."""
    return (
        "external_state"
        if any(domain in _EXTERNAL_STATE for domain in domains)
        else "clinical_support"
    )


def decide_commit(domains, status: str, has_items: bool) -> str:
    """Whether the retrieval that just ran should reach the generation stage."""
    if status == "sufficient" and has_items:
        return HARD
    if (
        status == "partial"
        and has_items
        and evidence_class(domains) == "external_state"
    ):
        return SOFT
    return ROLLBACK

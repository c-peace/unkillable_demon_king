from __future__ import annotations

import unittest

from app.evidence.models import CitableItem, EvidenceRequirement, RetrievalOutcome
from app.evidence.session import RetrievalSession


class RetrievalSessionTests(unittest.TestCase):
    def test_two_retrievals_close_one_cumulative_ledger(self) -> None:
        req_a = EvidenceRequirement(id="a", claim_or_question="claim A")
        req_b = EvidenceRequirement(id="b", claim_or_question="claim B")
        session = RetrievalSession.create((req_a, req_b))
        session.registry.register_payload(
            {"cite_uid": "cite-a", "content": "evidence A"}, source_tool="tool"
        )
        session.apply_outcome(
            RetrievalOutcome(
                status="partial",
                items=(CitableItem("cite-a", 0.9),),
                note="",
                evidence=(session.registry.get("cite-a"),),  # type: ignore[arg-type]
                requirements=(
                    EvidenceRequirement(
                        id="a",
                        claim_or_question="claim A",
                        status="supported",
                        cite_uids=("cite-a",),
                    ),
                    EvidenceRequirement(id="b", claim_or_question="claim B", status="unresolved"),
                ),
                model_rounds=1,
                mcp_calls=1,
            )
        )

        self.assertEqual(tuple(item.id for item in session.target_requirements()), ("b",))

        session.registry.register_payload(
            {"cite_uid": "cite-b", "content": "evidence B"}, source_tool="tool"
        )
        session.apply_outcome(
            RetrievalOutcome(
                status="sufficient",
                items=(CitableItem("cite-b", 0.8),),
                note="",
                evidence=(session.registry.get("cite-b"),),  # type: ignore[arg-type]
                requirements=(
                    EvidenceRequirement(
                        id="b",
                        claim_or_question="claim B",
                        status="supported",
                        cite_uids=("cite-b",),
                    ),
                ),
                model_rounds=1,
                mcp_calls=1,
            )
        )

        snapshot = session.snapshot()
        self.assertEqual(snapshot.status, "sufficient")
        self.assertEqual(
            tuple(item.status for item in snapshot.requirements),
            ("supported", "supported"),
        )
        self.assertEqual({item.cite_uid for item in snapshot.evidence}, {"cite-a", "cite-b"})
        self.assertEqual(snapshot.mcp_calls, 2)

    def test_sessions_do_not_share_registry_or_requirements(self) -> None:
        first = RetrievalSession.create((EvidenceRequirement(id="a", claim_or_question="A"),))
        second = RetrievalSession.create((EvidenceRequirement(id="b", claim_or_question="B"),))
        first.registry.register_payload(
            {"cite_uid": "private", "content": "only first"}, source_tool="tool"
        )

        self.assertIsNone(second.registry.get("private"))
        self.assertNotIn("a", second.requirements_by_id)


if __name__ == "__main__":
    unittest.main()

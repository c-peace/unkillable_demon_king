from __future__ import annotations

import unittest

from app.config import Settings
from app.deadline import Deadline
from app.errors import UpstreamError
from app.evidence.models import CitableItem, EvidenceItem, EvidenceRequirement, RetrievalOutcome
from app.evidence.verification import EvidenceVerifier, structural_verify
from tests.fakes import ScriptedL2, l2_tool_call


def _outcome(item: EvidenceItem, requirement: EvidenceRequirement) -> RetrievalOutcome:
    return RetrievalOutcome(
        status="sufficient",
        items=(CitableItem(item.cite_uid, 1.0),),
        note="",
        evidence=(item,),
        requirements=(requirement,),
        model_rounds=1,
        mcp_calls=1,
    )


class EvidenceVerificationTests(unittest.TestCase):
    def test_wrong_source_family_cannot_support_requirement(self) -> None:
        item = EvidenceItem(
            cite_uid="kcd-1",
            source_tool="kcd_get_name",
            content="A disease code name.",
        )
        requirement = EvidenceRequirement(
            id="drug",
            claim_or_question="Is warfarin safe in pregnancy?",
            source_preference="drug_safety",
            status="supported",
            cite_uids=("kcd-1",),
        )

        verified, issues = structural_verify(_outcome(item, requirement))

        self.assertEqual(verified.status, "partial")
        self.assertEqual(verified.requirement.status, "unresolved")
        self.assertEqual(verified.requirement.applicability_status, "inapplicable")
        self.assertIn("drug:source_family_mismatch", issues)

    def test_semantic_verifier_rejects_topic_mismatched_evidence(self) -> None:
        item = EvidenceItem(
            cite_uid="ibuprofen",
            source_tool="medical_search",
            content="Ibuprofen may cause gastrointestinal bleeding.",
        )
        requirement = EvidenceRequirement(
            id="warfarin",
            claim_or_question="Is warfarin 5 mg safe during pregnancy?",
            source_preference="drug_safety",
            status="supported",
            cite_uids=("ibuprofen",),
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "verify",
                    "verify_evidence",
                    {
                        "requirements": [
                            {
                                "requirement_id": "warfarin",
                                "verdict": "insufficient",
                                "cite_uids": ["ibuprofen"],
                                "applicability_note": "Different medicine and population.",
                                "conflict_note": "",
                            }
                        ]
                    },
                )
            ]
        )
        verifier = EvidenceVerifier(
            Settings(lunit_fm_api_key="test", evidence_verification_policy="always"),
            l2,  # type: ignore[arg-type]
        )

        run = verifier.run(
            _outcome(item, requirement),
            clinical_risk="elevated",
            deadline=Deadline.after(3),
            allow_semantic=True,
        )

        self.assertEqual(run.state, "semantic_verified")
        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.requirement.status, "unresolved")
        self.assertIn("warfarin:insufficient", run.issues)

    def test_semantic_verifier_failure_degrades_without_losing_the_draft_path(self) -> None:
        item = EvidenceItem(
            cite_uid="label",
            source_tool="adr_retrieve_drug_info",
            content="A candidate label excerpt.",
        )
        requirement = EvidenceRequirement(
            id="drug",
            claim_or_question="Is this drug safe in pregnancy?",
            source_preference="drug_safety",
            status="supported",
            cite_uids=("label",),
        )

        class FailingL2:
            def complete(self, *args, **kwargs):
                raise UpstreamError("verifier unavailable", code="verifier_unavailable")

        verifier = EvidenceVerifier(
            Settings(lunit_fm_api_key="test", evidence_verification_policy="always"),
            FailingL2(),  # type: ignore[arg-type]
        )
        run = verifier.run(
            _outcome(item, requirement),
            clinical_risk="elevated",
            deadline=Deadline.after(3),
            allow_semantic=True,
        )

        self.assertEqual(run.state, "semantic_unavailable")
        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.requirement.status, "unresolved")
        self.assertEqual(run.outcome.evidence[0].cite_uid, "label")


if __name__ == "__main__":
    unittest.main()

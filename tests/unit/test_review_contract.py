from __future__ import annotations

import unittest

from app.conversation import compile_conversation
from app.deadline import Deadline
from app.orchestration.review import ConditionalReviewer
from app.response_contracts import TaskKind, response_contract
from tests.fakes import ScriptedL2, l2_content, l2_tool_call


class ContractReviewTests(unittest.TestCase):
    def test_contract_allowed_issue_can_trigger_one_revision(self) -> None:
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "review",
                    "review_response",
                    {
                        "decision": "revise",
                        "issues": [
                            {
                                "category": "wrong_identifier",
                                "requirement_id": "code",
                                "concrete_defect": "The KCD identifier does not match the name.",
                            }
                        ],
                    },
                ),
                l2_content("corrected coding answer"),
            ]
        )
        reviewer = ConditionalReviewer(l2)  # type: ignore[arg-type]
        run = reviewer.run(
            compile_conversation(
                [{"role": "user", "content": "What is the KCD code for type 2 diabetes?"}]
            ),
            "draft",
            "{}",
            ["partial_critical_evidence"],
            deadline=Deadline.after(3),
            contract=response_contract(TaskKind.CODING),
        )

        self.assertTrue(run.reviewed)
        self.assertTrue(run.revised)
        self.assertEqual(run.draft, "corrected coding answer")

    def test_issue_outside_contract_is_not_allowed_to_rewrite_answer(self) -> None:
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "review",
                    "review_response",
                    {
                        "decision": "revise",
                        "issues": [
                            {
                                "category": "missing_emergency_referral",
                                "requirement_id": "",
                                "concrete_defect": "Add emergency triage to this coding answer.",
                            }
                        ],
                    },
                )
            ]
        )
        reviewer = ConditionalReviewer(l2)  # type: ignore[arg-type]
        run = reviewer.run(
            compile_conversation(
                [{"role": "user", "content": "What is the KCD code for type 2 diabetes?"}]
            ),
            "draft",
            "{}",
            ["partial_critical_evidence"],
            deadline=Deadline.after(3),
            contract=response_contract(TaskKind.CODING),
        )

        self.assertFalse(run.revised)
        self.assertEqual(run.state, "review_unavailable")
        self.assertEqual(run.draft, "draft")


if __name__ == "__main__":
    unittest.main()

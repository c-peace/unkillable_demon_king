from __future__ import annotations

import unittest

from app.clinical_state import AssertionState
from app.conversation import compile_conversation
from app.deadline import Deadline
from app.orchestration.state_enrichment import ClinicalStateEnricher
from tests.fakes import ScriptedL2, l2_tool_call


class ClinicalStateTests(unittest.TestCase):
    def test_safety_facts_keep_turn_provenance(self) -> None:
        compiled = compile_conversation(
            [
                {"role": "user", "content": "와파린 5mg을 복용 중입니다."},
                {"role": "assistant", "content": "임신 가능성이 있다고 이해했습니다."},
                {"role": "user", "content": "정정할게요. 임신은 아니에요."},
            ]
        )

        medication = compiled.state.clinical.medications[0]
        pregnancy = compiled.state.clinical.facts_of_kind("pregnancy")[-1]
        self.assertEqual(medication.name, "와파린")
        self.assertEqual(medication.provenance[0].turn_index, 1)
        self.assertEqual(pregnancy.assertion, AssertionState.NEGATED)
        self.assertEqual(pregnancy.provenance[0].turn_index, 3)
        self.assertIn("correction_at_turn_3", compiled.state.clinical.conflicts)

    def test_relation_level_stroke_signal_is_detected(self) -> None:
        compiled = compile_conversation(
            [{"role": "user", "content": "갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."}]
        )

        self.assertIn("possible_stroke", compiled.state.clinical.hard_risk_signals)
        self.assertTrue(compiled.is_high_risk)

    def test_generic_symptom_is_not_mistaken_for_a_medication(self) -> None:
        compiled = compile_conversation(
            [{"role": "user", "content": "What dose should I take for fever?"}]
        )

        self.assertFalse(compiled.state.clinical.named_medications)
        self.assertIn("medication_name", compiled.state.decision_critical_missing_facts)

    def test_clinician_role_is_request_local_and_provenance_derived(self) -> None:
        compiled = compile_conversation(
            [
                {
                    "role": "user",
                    "content": "I'm a clinician asking for general information about pregnancy pharmacology.",
                }
            ]
        )

        self.assertEqual(compiled.state.clinical.interaction.user_role, "clinician")
        self.assertFalse(compiled.is_high_risk)

    def test_negation_for_one_fact_does_not_negate_a_later_symptom(self) -> None:
        compiled = compile_conversation(
            [{"role": "user", "content": "임신은 아니지만 심한 흉통이 있어요."}]
        )

        pregnancy = compiled.state.clinical.facts_of_kind("pregnancy")[0]
        chest_pain = next(
            symptom for symptom in compiled.state.clinical.symptoms if symptom.name == "chest_pain"
        )
        self.assertEqual(pregnancy.assertion, AssertionState.NEGATED)
        self.assertEqual(chest_pain.assertion, AssertionState.ASSERTED)

    def test_old_hard_risk_carries_only_into_a_live_follow_up(self) -> None:
        unrelated = compile_conversation(
            [
                {"role": "user", "content": "갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."},
                {"role": "assistant", "content": "즉시 평가가 필요합니다."},
                {"role": "user", "content": "감기는 왜 생기나요?"},
            ]
        )
        follow_up = compile_conversation(
            [
                {"role": "user", "content": "갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."},
                {"role": "assistant", "content": "즉시 평가가 필요합니다."},
                {"role": "user", "content": "그 증상이 계속되면 어떻게 하나요?"},
            ]
        )
        resolved = compile_conversation(
            [
                {"role": "user", "content": "갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."},
                {"role": "assistant", "content": "즉시 평가가 필요합니다."},
                {"role": "user", "content": "그 증상은 이제 사라졌어요."},
            ]
        )

        self.assertFalse(unrelated.state.clinical.hard_risk_signals)
        self.assertIn("possible_stroke", follow_up.state.clinical.hard_risk_signals)
        self.assertFalse(resolved.state.clinical.hard_risk_signals)

    def test_same_turn_reference_is_resolved_by_a_local_medication_anchor(self) -> None:
        compiled = compile_conversation(
            [
                {
                    "role": "user",
                    "content": "I take warfarin. Can I still take it with ibuprofen?",
                }
            ]
        )

        self.assertFalse(compiled.state.unresolved_references)
        self.assertNotIn("reference_target", compiled.state.decision_critical_missing_facts)

    def test_optional_enrichment_accepts_only_valid_source_turns(self) -> None:
        compiled = compile_conversation(
            [
                {"role": "user", "content": "그 증상이 계속돼요."},
                {"role": "assistant", "content": "어떤 증상인지 확인이 필요합니다."},
            ]
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "state",
                    "submit_clinical_state_enrichment",
                    {
                        "facts": [
                            {
                                "kind": "duration",
                                "value": "ongoing",
                                "subject": "patient",
                                "assertion": "asserted",
                                "turn_indices": [1, 99],
                            }
                        ],
                        "hard_risk_signals": [],
                    },
                )
            ]
        )
        run = ClinicalStateEnricher(l2).run(  # type: ignore[arg-type]
            compiled.state.clinical,
            case_packet=compiled.case_packet,
            turn_roles=("user", "assistant"),
            deadline=Deadline.after(3),
        )

        self.assertEqual(run.status, "enriched")
        fact = run.state.latest_fact("duration")
        self.assertIsNotNone(fact)
        self.assertEqual(tuple(ref.turn_index for ref in fact.provenance), (1,))  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()

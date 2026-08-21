from __future__ import annotations

import unittest

from app.config import Settings
from app.conversation import compile_conversation
from app.orchestration.planning import (
    RESPONSE_ANSWER_ONLY,
    RESPONSE_ANSWER_WITH_QUESTION,
    build_harness_plan,
)
from app.prompts import generation_after_retrieval_prompt, review_system_prompt


def _compiled(*user_turns: str):
    messages = []
    for index, text in enumerate(user_turns):
        if index:
            messages.append({"role": "assistant", "content": "이전 답변"})
        messages.append({"role": "user", "content": text})
    return compile_conversation(messages)


class PlanningStateTests(unittest.TestCase):
    def test_source_sensitive_context_is_carried_only_for_a_follow_up(self) -> None:
        follow_up = _compiled(
            "와파린과 이부프로펜의 상호작용을 설명해 주세요.",
            "그 내용은 지금도 같은가요?",
        )
        unrelated = _compiled(
            "현재 CKD 혈압 목표를 알려 주세요.",
            "감기는 왜 생기나요?",
        )

        follow_up_plan = build_harness_plan(follow_up, Settings())
        self.assertEqual(follow_up_plan.lane, "HIGH_RISK")
        self.assertTrue(follow_up_plan.retrieval_allowed)
        self.assertEqual(build_harness_plan(unrelated, Settings()).lane, "DIRECT")

    def test_explicitly_negated_risk_is_not_active(self) -> None:
        compiled = _compiled("저는 임신이 아니에요. 감기는 왜 생기나요?")

        plan = build_harness_plan(compiled, Settings())

        self.assertFalse(compiled.state.active_risk_signals)
        self.assertFalse(compiled.is_high_risk)
        self.assertEqual(plan.lane, "DIRECT")

    def test_negating_one_risk_does_not_hide_another_active_risk(self) -> None:
        compiled = _compiled("임신은 아니지만 흉통이 있어요.")

        plan = build_harness_plan(compiled, Settings())

        self.assertTrue(compiled.is_high_risk)
        self.assertEqual(plan.lane, "HIGH_RISK")

    def test_generic_dose_question_requires_the_medication_name(self) -> None:
        generic = _compiled("What dose should I take?")
        named = _compiled("What dose of ibuprofen should I take?")

        generic_plan = build_harness_plan(generic, Settings())
        named_plan = build_harness_plan(named, Settings())

        self.assertIn("medication_name", generic.state.decision_critical_missing_facts)
        self.assertTrue(generic_plan.clarification_required)
        self.assertNotIn("medication_name", named.state.decision_critical_missing_facts)
        self.assertFalse(named_plan.clarification_required)

    def test_elliptical_medication_follow_up_uses_prior_turn(self) -> None:
        compiled = _compiled(
            "와파린 5mg을 복용 중입니다.",
            "계속 먹어도 될까요?",
        )

        plan = build_harness_plan(compiled, Settings())

        self.assertTrue(compiled.state.has_follow_up_reference)
        self.assertIn("와파린 5mg", compiled.state.retrieval_context())
        self.assertTrue(plan.retrieval_allowed)

    def test_text_operation_with_quoted_risk_content_stays_direct(self) -> None:
        compiled = _compiled(
            "다음 문장을 영어로 번역해 주세요: 흉통이 심해서 응급실 가야 하나요?"
        )

        plan = build_harness_plan(compiled, Settings())

        self.assertFalse(compiled.is_high_risk)
        self.assertEqual(plan.lane, "DIRECT")
        self.assertFalse(plan.retrieval_allowed)
        self.assertFalse(plan.review_reasons)

    def test_plain_triage_criteria_does_not_imply_a_guideline_lookup(self) -> None:
        plan = build_harness_plan(
            _compiled("어떤 기준으로 응급실에 가야 하나요?"),
            Settings(),
        )

        self.assertFalse(plan.retrieval_allowed)

    def test_multipart_evidence_requests_seed_atomic_requirements(self) -> None:
        plan = build_harness_plan(
            _compiled(
                "와파린과 이부프로펜의 상호작용은 무엇인가요? "
                "그리고 와파린의 허가 적응증은 무엇인가요?"
            ),
            Settings(),
        )

        self.assertEqual(len(plan.evidence_requirements), 2)
        self.assertEqual(
            tuple(requirement.id for requirement in plan.evidence_requirements),
            ("plan-1", "plan-2"),
        )

    def test_plain_multipart_request_does_not_trigger_review_by_shape_alone(self) -> None:
        plain = build_harness_plan(
            _compiled("감기의 원인은 무엇인가요? 그리고 언제 나아지나요?"),
            Settings(),
        )
        strict = build_harness_plan(
            _compiled(
                "감기의 원인은 무엇인가요? 그리고 언제 나아지나요? 표 양식으로 작성해 주세요."
            ),
            Settings(),
        )

        self.assertFalse(plain.review_reasons)
        self.assertIn("strict_multipart_output", strict.review_reasons)

    def test_missing_child_fever_facts_select_the_question_contract(self) -> None:
        compiled = _compiled("아이가 열이 나는데 해열제 용량은 얼마인가요?")

        plan = build_harness_plan(compiled, Settings())

        self.assertTrue(plan.clarification_required)
        self.assertEqual(plan.response_contract, RESPONSE_ANSWER_WITH_QUESTION)
        self.assertIn("child_age", plan.clarification_reason)

    def test_complete_or_text_operation_requests_use_answer_only(self) -> None:
        complete = build_harness_plan(
            _compiled("고혈압은 왜 생기나요?"),
            Settings(),
        )
        rewrite = build_harness_plan(
            _compiled("다음 문장을 자연스럽게 다듬어 주세요: 예약을 변경하고 싶습니다."),
            Settings(),
        )

        self.assertEqual(complete.response_contract, RESPONSE_ANSWER_ONLY)
        self.assertEqual(rewrite.response_contract, RESPONSE_ANSWER_ONLY)
        self.assertFalse(rewrite.retrieval_allowed)
        self.assertFalse(rewrite.review_reasons)

    def test_semantic_risk_and_claim_admission_are_independent(self) -> None:
        stroke = build_harness_plan(
            _compiled("갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."),
            Settings(),
        )
        clinician = build_harness_plan(
            _compiled("I'm a clinician asking for general information about medication use during pregnancy."),
            Settings(),
        )
        medication = build_harness_plan(
            _compiled("Is warfarin 5 mg safe during pregnancy?"),
            Settings(),
        )

        self.assertEqual(stroke.lane, "HIGH_RISK")
        self.assertEqual(stroke.clinical_risk.value, "emergency")
        self.assertEqual(stroke.interaction_mode.value, "answer_and_clarify")
        self.assertEqual(clinician.lane, "DIRECT")
        self.assertEqual(clinician.clinical_risk.value, "routine")
        self.assertEqual(medication.retrieval_domains, ("drug_safety",))

    def test_clarification_blocks_invalid_retrieval_but_not_emergency_action(self) -> None:
        dose = build_harness_plan(
            _compiled("What dose should I take for fever?"),
            Settings(),
        )
        emergency = build_harness_plan(
            _compiled("갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."),
            Settings(),
        )

        self.assertEqual(dose.interaction_mode.value, "clarify_first")
        self.assertFalse(dose.retrieval_allowed)
        self.assertEqual(emergency.interaction_mode.value, "answer_and_clarify")

    def test_general_adverse_effect_mechanism_does_not_force_retrieval(self) -> None:
        plan = build_harness_plan(
            _compiled("항생제는 왜 설사 부작용이 생기나요?"),
            Settings(),
        )

        self.assertEqual(plan.task_kind.value, "general_education")
        self.assertFalse(plan.retrieval_allowed)

    def test_coding_contract_does_not_inherit_triage_checklist(self) -> None:
        plan = build_harness_plan(
            _compiled("What is the KCD code for type 2 diabetes?"),
            Settings(),
        )

        generation_prompt = generation_after_retrieval_prompt(plan.contract_spec)
        review_prompt = review_system_prompt(plan.contract_spec)
        self.assertIn("Type: coding", generation_prompt)
        self.assertIn("Do not expand into: triage, dosing", generation_prompt)
        self.assertNotIn("who to see, how soon", generation_prompt)
        self.assertIn("Allowed issue categories", review_prompt)
        self.assertNotIn("Does the draft say plainly who to see", review_prompt)

    def test_note_as_a_verb_does_not_disable_drug_safety_retrieval(self) -> None:
        compiled = _compiled("What side effects should I note while taking warfarin?")
        plan = build_harness_plan(compiled, Settings())

        self.assertFalse(compiled.state.is_text_operation)
        self.assertEqual(plan.task_kind.value, "medication_safety")
        self.assertEqual(plan.retrieval_domains, ("drug_safety",))

    def test_old_symptoms_do_not_turn_an_unrelated_latest_question_into_triage(self) -> None:
        compiled = compile_conversation(
            [
                {"role": "user", "content": "갑자기 한쪽 팔에 힘이 빠지고 말이 어눌해졌어요."},
                {"role": "assistant", "content": "즉시 평가가 필요합니다."},
                {"role": "user", "content": "감기는 왜 생기나요?"},
            ]
        )
        plan = build_harness_plan(compiled, Settings())

        self.assertEqual(plan.task_kind.value, "general_education")
        self.assertEqual(plan.lane, "DIRECT")


if __name__ == "__main__":
    unittest.main()

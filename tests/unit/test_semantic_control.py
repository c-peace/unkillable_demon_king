from __future__ import annotations

import unittest

from app.clients.mcp import McpTool
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.conversation import compile_conversation
from app.deadline import Deadline
from app.evidence.models import (
    EvidenceRegistry,
    EvidenceRequirement,
    EvidenceRequirementLedger,
    parse_ledger_final_selection,
)
from app.orchestration.driver import ConversationDriver
from app.orchestration.planning import (
    FORCED_PLAN_TOOL_CHOICE,
    SUBMIT_RESPONSE_PLAN_TOOL,
    ResponsePlan,
    StructuredPlanner,
    parse_response_plan,
)
from app.orchestration.retrieval import RetrievalEngine
from app.orchestration.review import StructuredReviewer, deterministic_review_issues
from tests.fakes import ScriptedL2, l2_content, l2_tool_call


class SemanticPlanningTests(unittest.TestCase):
    def test_full_history_risk_upgrades_indirect_latest_turn(self) -> None:
        compiled = compile_conversation(
            [
                {"role": "user", "content": "임신 중에 이 약을 복용하고 있어요."},
                {"role": "assistant", "content": "확인했습니다."},
                {"role": "user", "content": "그럼 지금은 어떻게 해야 하나요?"},
            ]
        )
        plan = parse_response_plan(
            {
                "response_mode": "ANSWER",
                "risk_level": "ROUTINE",
                "clinical_facts": [
                    {
                        "kind": "pregnancy",
                        "value": "pregnant",
                        "source_turns": [1],
                        "negated": False,
                        "corrected": False,
                    }
                ],
                "interaction": {
                    "intent": "medication safety",
                    "language": "ko",
                    "requested_format": "",
                    "unresolved_references": ["이 약"],
                    "strict_format": False,
                },
                "risk_signals": [],
                "missing_information": [],
                "evidence_requirements": [],
                "answer_obligations": ["Address pregnancy-specific safety."],
            },
            compiled=compiled,
        )
        self.assertEqual(plan.lane, "HIGH_RISK")
        self.assertEqual(plan.clinical_facts[0].source_turns, (1,))

    def test_plan_rejects_nonexistent_source_turn(self) -> None:
        compiled = compile_conversation([{"role": "user", "content": "질문"}])
        with self.assertRaisesRegex(ValueError, "invalid_source_turns"):
            parse_response_plan(
                {
                    "response_mode": "ANSWER",
                    "risk_level": "ROUTINE",
                    "clinical_facts": [
                        {
                            "kind": "medication",
                            "value": "unknown",
                            "source_turns": [99],
                        }
                    ],
                    "interaction": {"intent": "answer", "language": "ko"},
                    "risk_signals": [],
                    "missing_information": [],
                    "evidence_requirements": [],
                    "answer_obligations": [],
                },
                compiled=compiled,
            )

    def test_source_sensitive_answer_requires_evidence_requirement(self) -> None:
        compiled = compile_conversation(
            [{"role": "user", "content": "현재 가이드라인 권고를 알려줘."}]
        )
        with self.assertRaisesRegex(ValueError, "source_sensitive_without_requirement"):
            parse_response_plan(
                {
                    "response_mode": "ANSWER",
                    "risk_level": "ROUTINE",
                    "clinical_facts": [],
                    "interaction": {
                        "intent": "guideline answer",
                        "language": "ko",
                        "requested_format": "",
                        "unresolved_references": [],
                        "strict_format": False,
                    },
                    "risk_signals": [],
                    "missing_information": [],
                    "evidence_requirements": [],
                    "answer_obligations": [],
                },
                compiled=compiled,
            )

    def test_planner_forces_tool_and_repairs_one_invalid_plan(self) -> None:
        compiled = compile_conversation([{"role": "user", "content": "감기는 무엇인가요?"}])
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "invalid",
                    "submit_response_plan",
                    {"response_mode": "ANSWER", "risk_level": "ROUTINE"},
                ),
                l2_tool_call(
                    "valid",
                    "submit_response_plan",
                    {
                        "response_mode": "ANSWER",
                        "risk_level": "ROUTINE",
                        "clinical_facts": [],
                        "interaction": {
                            "intent": "explanation",
                            "language": "ko",
                            "requested_format": "",
                            "unresolved_references": [],
                            "strict_format": False,
                        },
                        "risk_signals": [],
                        "missing_information": [],
                        "evidence_requirements": [],
                        "answer_obligations": ["Explain the common cold."],
                    },
                ),
            ]
        )
        result = StructuredPlanner(l2=l2).plan(compiled, deadline=Deadline.after(3))

        self.assertEqual(result.mode, "structured_repaired")
        self.assertEqual(result.l2_calls, 2)
        self.assertEqual(l2.calls[0]["tool_choice"], FORCED_PLAN_TOOL_CHOICE)
        self.assertIn("invalid_clinical_facts", l2.calls[1]["messages"][1]["content"])

    def test_planner_bounded_repairs_missing_source_requirement(self) -> None:
        compiled = compile_conversation(
            [{"role": "user", "content": "현재 CKD 가이드라인 권고를 알려줘."}]
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "plan",
                    "submit_response_plan",
                    {
                        "response_mode": "ANSWER",
                        "risk_level": "ROUTINE",
                        "clinical_facts": [],
                        "interaction": {
                            "intent": "guideline answer",
                            "language": "ko",
                            "requested_format": "",
                            "unresolved_references": [],
                            "strict_format": False,
                        },
                        "risk_signals": [],
                        "missing_information": [],
                        "evidence_requirements": [],
                        "answer_obligations": [],
                    },
                )
            ]
        )

        result = StructuredPlanner(l2=l2).plan(compiled, deadline=Deadline.after(3))

        self.assertEqual(result.mode, "structured_bounded_repair")
        self.assertEqual(result.error_code, "source_sensitive_without_requirement")
        self.assertEqual(result.l2_calls, 1)
        self.assertEqual(result.plan.evidence_requirements[0].source_family, "guideline")

    def test_plan_schema_defines_nested_parser_fields(self) -> None:
        parameters = SUBMIT_RESPONSE_PLAN_TOOL["function"]["parameters"]
        fact = parameters["properties"]["clinical_facts"]["items"]
        requirement = parameters["properties"]["evidence_requirements"]["items"]

        self.assertEqual(
            set(fact["required"]),
            {"kind", "value", "source_turns", "negated", "corrected"},
        )
        self.assertFalse(fact["additionalProperties"])
        self.assertIn("question", requirement["required"])

    def test_plan_rejects_string_booleans_in_nested_fields(self) -> None:
        compiled = compile_conversation([{"role": "user", "content": "질문"}])
        with self.assertRaisesRegex(ValueError, "invalid_clinical_fact_negated"):
            parse_response_plan(
                {
                    "response_mode": "ANSWER",
                    "risk_level": "ROUTINE",
                    "clinical_facts": [
                        {
                            "kind": "condition",
                            "value": "hypertension",
                            "source_turns": [1],
                            "negated": "false",
                            "corrected": False,
                        }
                    ],
                    "interaction": {
                        "intent": "answer",
                        "language": "ko",
                        "requested_format": "",
                        "unresolved_references": [],
                        "strict_format": False,
                    },
                    "risk_signals": [],
                    "missing_information": [],
                    "evidence_requirements": [],
                    "answer_obligations": [],
                },
                compiled=compiled,
            )

    def test_plan_rejects_empty_control_strings(self) -> None:
        compiled = compile_conversation([{"role": "user", "content": "질문"}])
        with self.assertRaisesRegex(ValueError, "invalid_interaction_intent"):
            parse_response_plan(
                {
                    "response_mode": "ANSWER",
                    "risk_level": "ROUTINE",
                    "clinical_facts": [],
                    "interaction": {
                        "intent": "   ",
                        "language": "ko",
                        "requested_format": "",
                        "unresolved_references": [],
                        "strict_format": False,
                    },
                    "risk_signals": [],
                    "missing_information": [],
                    "evidence_requirements": [],
                    "answer_obligations": [],
                },
                compiled=compiled,
            )

    def test_planner_does_not_start_repair_without_reserved_time(self) -> None:
        compiled = compile_conversation([{"role": "user", "content": "질문"}])
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "invalid",
                    "submit_response_plan",
                    {"response_mode": "ANSWER", "risk_level": "ROUTINE"},
                )
            ]
        )
        result = StructuredPlanner(l2=l2, retry_reserve_sec=10).plan(
            compiled,
            deadline=Deadline.after(3),
        )

        self.assertEqual(result.mode, "legacy_fallback")
        self.assertEqual(result.l2_calls, 1)
        self.assertEqual(result.error_code, "invalid_clinical_facts")


class RequirementLedgerTests(unittest.TestCase):
    def _ledger(self) -> EvidenceRequirementLedger:
        return EvidenceRequirementLedger(
            requirements=(
                EvidenceRequirement(
                    id="req-safety",
                    claim_or_question="Pregnancy safety",
                    source_family="drug_label",
                    criticality="critical",
                ),
                EvidenceRequirement(
                    id="req-interaction",
                    claim_or_question="Drug interaction",
                    source_family="drug_label",
                    criticality="critical",
                ),
            )
        )

    def test_sufficient_requires_every_critical_requirement(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "cite-safety", "content": "Pregnancy label evidence"},
            source_tool="adr_retrieve_drug_info",
        )
        outcome = parse_ledger_final_selection(
            {
                "status": "sufficient",
                "requirements": [
                    {
                        "id": "req-safety",
                        "status": "supported",
                        "cite_uids": ["cite-safety"],
                        "applicability_note": "Applies to the named drug and pregnancy.",
                    },
                    {
                        "id": "req-interaction",
                        "status": "missing",
                        "cite_uids": [],
                        "applicability_note": "",
                    },
                ],
                "note": "One critical gap remains.",
            },
            ledger=self._ledger(),
            registry=registry,
            model_rounds=2,
            mcp_calls=2,
            max_items=6,
        )
        self.assertEqual(outcome.status, "partial")
        self.assertEqual(outcome.ledger.by_id("req-interaction").status, "missing")

    def test_supported_requires_known_citation_and_applicability(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported requirement"):
            parse_ledger_final_selection(
                {
                    "status": "sufficient",
                    "requirements": [
                        {
                            "id": "req-safety",
                            "status": "supported",
                            "cite_uids": ["unknown"],
                            "applicability_note": "",
                        }
                    ],
                },
                ledger=EvidenceRequirementLedger(
                    requirements=(self._ledger().requirements[0],)
                ),
                registry=EvidenceRegistry(),
                model_rounds=1,
                mcp_calls=1,
                max_items=6,
            )

    def test_retrieval_requires_named_gap_and_strips_harness_argument(self) -> None:
        class LedgerMcp:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            def list_tools(self, *, deadline):
                return (
                    McpTool(
                        "adr_retrieve_drug_info",
                        "Retrieve official drug information",
                        {
                            "type": "object",
                            "properties": {"drug_name": {"type": "string"}},
                            "required": ["drug_name"],
                        },
                    ),
                )

            def call_tool(self, name, arguments, *, deadline):
                self.calls.append((name, dict(arguments)))
                return {
                    "cite_uid": "cite-safety",
                    "content": "Official pregnancy safety information.",
                }

        ledger = EvidenceRequirementLedger(
            requirements=(
                EvidenceRequirement(
                    id="req-safety",
                    claim_or_question="Pregnancy safety",
                    source_family="drug_label",
                    criticality="critical",
                ),
            )
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "drug-call",
                    "adr_retrieve_drug_info",
                    {"requirement_id": "req-safety", "drug_name": "example"},
                ),
                l2_tool_call(
                    "final-call",
                    "finalize_retrieval",
                    {
                        "status": "sufficient",
                        "requirements": [
                            {
                                "id": "req-safety",
                                "status": "supported",
                                "cite_uids": ["cite-safety"],
                                "applicability_note": "Applies to the named medicine.",
                            }
                        ],
                        "note": "Critical requirement closed.",
                    },
                ),
            ]
        )
        mcp = LedgerMcp()
        engine = RetrievalEngine(
            Settings(
                lunit_fm_api_key="test",
                max_retrieval_model_rounds=2,
                max_mcp_tool_calls=2,
                mcp_tool_mode="all",
            ),
            l2=l2,
            mcp=mcp,  # type: ignore[arg-type]
        )
        run = engine.run(ledger, deadline=Deadline.after(3), request_id="req-ledger")

        self.assertEqual(run.outcome.status, "sufficient")
        self.assertEqual(run.outcome.ledger.by_id("req-safety").status, "supported")
        self.assertEqual(mcp.calls[0][1], {"drug_name": "example"})

    def test_budget_fallback_does_not_promote_unadjudicated_registry_items(self) -> None:
        class OneToolMcp:
            def list_tools(self, *, deadline):
                return (
                    McpTool(
                        "adr_retrieve_drug_info",
                        "Retrieve drug information",
                        {"type": "object", "properties": {}},
                    ),
                )

            def call_tool(self, name, arguments, *, deadline):
                return {"cite_uid": "candidate-only", "content": "Unadjudicated candidate"}

        ledger = EvidenceRequirementLedger(
            requirements=(
                EvidenceRequirement("req-1", "Safety", source_family="drug_label"),
            )
        )
        engine = RetrievalEngine(
            Settings(
                lunit_fm_api_key="test",
                max_retrieval_model_rounds=1,
                max_mcp_tool_calls=1,
                mcp_tool_mode="all",
            ),
            l2=ScriptedL2(
                [
                    l2_tool_call(
                        "call",
                        "adr_retrieve_drug_info",
                        {"requirement_id": "req-1"},
                    )
                ]
            ),
            mcp=OneToolMcp(),  # type: ignore[arg-type]
        )
        run = engine.run(ledger, deadline=Deadline.after(3))
        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.evidence, ())


class StructuredReviewTests(unittest.TestCase):
    def _plan(self, lane: str = "DIRECT") -> ResponsePlan:
        return ResponsePlan.empty(lane=lane, intent="answer", language="ko")

    def test_partial_alone_does_not_trigger_review(self) -> None:
        reviewer = StructuredReviewer(Settings(lunit_fm_api_key="test"), l2=ScriptedL2([]))
        self.assertFalse(
            reviewer.should_review(
                self._plan(),
                retrieval_status="partial",
                deterministic_issues=(),
            )
        )

    def test_malformed_reviewer_output_is_unavailable_not_pass(self) -> None:
        reviewer = StructuredReviewer(
            Settings(lunit_fm_api_key="test"),
            l2=ScriptedL2([l2_content("PASS")]),
        )
        result = reviewer.review(
            compiled=compile_conversation([{"role": "user", "content": "질문"}]),
            plan=self._plan("HIGH_RISK"),
            draft="초안",
            evidence_payload="{}",
            deadline=Deadline.after(3),
        )
        self.assertEqual(result.status, "review_unavailable")
        self.assertEqual(result.error_code, "missing_structured_call")
        self.assertEqual(result.l2_calls, 1)
        self.assertFalse(result.passed)

    def test_unknown_citation_is_a_material_deterministic_issue(self) -> None:
        issues = deterministic_review_issues(
            draft="설명입니다 [cite-unknown]",
            plan=self._plan(),
            evidence_payload='{"evidence":[]}',
        )
        self.assertEqual(issues[0].category, "citation_mismatch")
        self.assertEqual(issues[0].severity, "material")


class FeatureDependencyTests(unittest.TestCase):
    def test_ledger_requires_planning_and_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "ledger"):
            Settings(
                enable_requirement_ledger=True,
                enable_structured_planning=False,
                enable_response_contract=True,
            )


class PlannedDriverTests(unittest.TestCase):
    def test_simple_single_turn_uses_structured_direct_plan(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            enable_structured_planning=True,
            enable_high_risk_review=False,
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "plan",
                    "submit_response_plan",
                    {
                        "response_mode": "ANSWER",
                        "risk_level": "ROUTINE",
                        "clinical_facts": [],
                        "interaction": {
                            "intent": "explanation",
                            "language": "ko",
                            "requested_format": "",
                            "unresolved_references": [],
                            "strict_format": False,
                        },
                        "risk_signals": [],
                        "missing_information": [],
                        "evidence_requirements": [],
                        "answer_obligations": ["감기를 설명한다."],
                    },
                ),
                l2_content("직접 답변"),
            ]
        )
        driver = ConversationDriver(
            settings,
            l2=l2,  # type: ignore[arg-type]
            retrieval=__import__("tests.fakes", fromlist=["NeverRetrieval"]).NeverRetrieval(),
        )
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "감기는 무엇인가요?"},),
            ),
            request_id="req-fast",
        )
        self.assertEqual(result.content, "직접 답변")
        self.assertEqual(result.trace["planned_lane"], "DIRECT")
        self.assertEqual(result.trace["planner_mode"], "structured")
        self.assertEqual(result.trace["response_mode"], "ANSWER")
        self.assertEqual(result.trace["retrievals"], 0)

    def test_high_risk_plan_runs_structured_review_without_partial_trigger(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            enable_structured_planning=True,
            enable_response_contract=True,
            enable_structured_review=True,
            enable_high_risk_review=False,
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "plan",
                    "submit_response_plan",
                    {
                        "response_mode": "ANSWER",
                        "risk_level": "HIGH",
                        "clinical_facts": [
                            {
                                "kind": "pregnancy",
                                "value": "pregnant",
                                "source_turns": [1],
                                "negated": False,
                                "corrected": False,
                            }
                        ],
                        "interaction": {
                            "intent": "medication safety",
                            "language": "ko",
                            "requested_format": "",
                            "unresolved_references": ["그 약"],
                            "strict_format": False,
                        },
                        "risk_signals": [
                            {
                                "category": "pregnancy",
                                "source_turns": [1],
                                "materiality": "material",
                            }
                        ],
                        "missing_information": [],
                        "evidence_requirements": [],
                        "answer_obligations": ["Address pregnancy-specific safety."],
                    },
                ),
                l2_content("검토 전 답변"),
                l2_tool_call(
                    "review",
                    "submit_review",
                    {"verdict": "PASS", "issues": []},
                ),
            ]
        )
        driver = ConversationDriver(
            settings,
            l2=l2,  # type: ignore[arg-type]
            retrieval=__import__("tests.fakes", fromlist=["NeverRetrieval"]).NeverRetrieval(),
        )
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=(
                    {"role": "user", "content": "임신 중 약을 먹고 있어요."},
                    {"role": "assistant", "content": "확인했습니다."},
                    {"role": "user", "content": "그럼 어떻게 해야 하나요?"},
                ),
            ),
            request_id="req-high-risk",
        )
        self.assertEqual(result.content, "검토 전 답변")
        self.assertEqual(result.trace["planned_lane"], "HIGH_RISK")
        self.assertEqual(result.trace["review_status"], "passed")
        self.assertTrue(result.trace["reviewed"])

    def test_full_semantic_path_closes_named_requirement_before_generation(self) -> None:
        class DrugMcp:
            def list_tools(self, *, deadline):
                return (
                    McpTool(
                        "adr_retrieve_drug_info",
                        "Retrieve official drug information",
                        {
                            "type": "object",
                            "properties": {"drug_name": {"type": "string"}},
                            "required": ["drug_name"],
                        },
                    ),
                )

            def call_tool(self, name, arguments, *, deadline):
                return {
                    "cite_uid": "cite-label",
                    "content": "Official label evidence.",
                }

        settings = Settings(
            lunit_fm_api_key="test",
            enable_structured_planning=True,
            enable_requirement_ledger=True,
            enable_response_contract=True,
            enable_structured_review=True,
            enable_high_risk_review=False,
            max_retrieval_model_rounds=2,
            max_mcp_tool_calls=2,
            mcp_tool_mode="all",
        )
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "plan",
                    "submit_response_plan",
                    {
                        "response_mode": "ANSWER",
                        "risk_level": "ROUTINE",
                        "clinical_facts": [],
                        "interaction": {
                            "intent": "official indication",
                            "language": "ko",
                            "requested_format": "",
                            "unresolved_references": [],
                            "strict_format": False,
                        },
                        "risk_signals": [],
                        "missing_information": [],
                        "evidence_requirements": [
                            {
                                "id": "req-label",
                                "question": "What is the official indication?",
                                "source_family": "drug_label",
                                "jurisdiction": "KR",
                                "criticality": "critical",
                            }
                        ],
                        "answer_obligations": ["State the official indication."],
                    },
                ),
                l2_tool_call(
                    "retrieve",
                    "adr_retrieve_drug_info",
                    {"requirement_id": "req-label", "drug_name": "example"},
                ),
                l2_tool_call(
                    "finalize",
                    "finalize_retrieval",
                    {
                        "status": "sufficient",
                        "requirements": [
                            {
                                "id": "req-label",
                                "status": "supported",
                                "cite_uids": ["cite-label"],
                                "applicability_note": "Official label for the named drug.",
                            }
                        ],
                        "note": "Requirement closed.",
                    },
                ),
                l2_content("공식 적응증을 근거와 함께 설명합니다."),
            ]
        )
        retrieval = RetrievalEngine(
            settings,
            l2=l2,  # type: ignore[arg-type]
            mcp=DrugMcp(),  # type: ignore[arg-type]
        )
        driver = ConversationDriver(settings, l2=l2, retrieval=retrieval)  # type: ignore[arg-type]
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "공식 허가 적응증을 알려줘."},),
            ),
            request_id="req-full-semantic",
        )
        self.assertEqual(result.trace["retrieval_status"], "sufficient")
        self.assertEqual(result.trace["requirement_count"], 1)
        self.assertEqual(result.trace["review_status"], "not_used")
        self.assertEqual(result.content, "공식 적응증을 근거와 함께 설명합니다.")

    def test_structured_review_requires_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "review"):
            Settings(
                enable_structured_planning=True,
                enable_structured_review=True,
                enable_response_contract=False,
            )


if __name__ == "__main__":
    unittest.main()

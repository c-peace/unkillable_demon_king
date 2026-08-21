from __future__ import annotations

import unittest

from app.clients.mcp import McpTool
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.deadline import Deadline
from app.errors import UpstreamError
from app.orchestration.driver import ConversationDriver
from app.orchestration.retrieval import RetrievalEngine
from tests.fakes import NeverRetrieval, ScriptedL2, l2_content, l2_tool_call


class FakeMcp:
    def __init__(self) -> None:
        self.calls = []

    def list_tools(self, *, deadline):
        return (
            McpTool(
                "index_get_relevant_nodes",
                "Find candidate guideline nodes",
                {
                    "type": "object",
                    "properties": {
                        "corpus_tag": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["corpus_tag", "query"],
                },
            ),
            McpTool(
                "index_get_page_content",
                "Read selected guideline pages",
                {
                    "type": "object",
                    "properties": {
                        "corpus_tag": {"type": "string"},
                        "doc_id": {"type": "string"},
                        "start_page": {"type": "integer"},
                        "end_page": {"type": "integer"},
                    },
                    "required": ["corpus_tag", "doc_id", "start_page", "end_page"],
                },
            ),
        )

    def call_tool(self, name, arguments, *, deadline):
        self.calls.append((name, arguments))
        if name == "index_get_relevant_nodes":
            return {
                "structuredContent": {
                    "corpus_tag": arguments.get("corpus_tag", "guideline"),
                    "matching_document": {
                        "doc_id": "guideline-1",
                        "title": "Clinical Guideline",
                        "range": [10, 12],
                    },
                }
            }
        if name == "index_get_page_content":
            return {
                "structuredContent": {
                    "cite_uid": "cite-guideline",
                    "title": "Clinical Guideline",
                    "content": "The guideline supports the stated action.",
                    "pages": [
                        {"page": 10, "text": "The guideline supports the stated action."},
                        {"page": 11, "text": "More supporting text."},
                    ],
                }
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"cite_uid":"cite-guideline","title":"Clinical Guideline",'
                        '"content":"The guideline supports the stated action."}'
                    ),
                }
            ]
        }


class DriverTests(unittest.TestCase):
    def test_direct_answer_uses_full_history(self) -> None:
        settings = Settings(enable_high_risk_review=False, lunit_fm_api_key="test")
        l2 = ScriptedL2([l2_content("최종 L2 답변")])
        driver = ConversationDriver(settings, l2=l2, retrieval=NeverRetrieval())  # type: ignore[arg-type]
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=(
                    {"role": "user", "content": "첫 정보"},
                    {"role": "assistant", "content": "이전 답"},
                    {"role": "user", "content": "후속 질문"},
                ),
            ),
            request_id="req-direct",
        )
        self.assertEqual(result.content, "최종 L2 답변")
        sent = l2.calls[0]["messages"]
        self.assertEqual([message["role"] for message in sent[-3:]], ["user", "assistant", "user"])
        self.assertEqual(result.trace["lane"], "DIRECT")

    def test_direct_answer_succeeds_without_global_request_deadline(self) -> None:
        settings = Settings(enable_high_risk_review=False, lunit_fm_api_key="test", request_timeout_sec=None)  # type: ignore[arg-type]
        l2 = ScriptedL2([l2_content("데드라인 없이도 최종 답변")])
        driver = ConversationDriver(settings, l2=l2, retrieval=NeverRetrieval())  # type: ignore[arg-type]
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "설명해줘"},),
            ),
            request_id="req-no-deadline",
        )
        self.assertEqual(result.content, "데드라인 없이도 최종 답변")

    def test_generation_retrieval_generation_path(self) -> None:
        settings = Settings(enable_high_risk_review=False, 
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=3,
            max_generation_retrievals=1,
        )
        generation_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "gen-call",
                    "retrieve_relevant_content",
                    {"query": "CKD 성인의 혈압 목표 guideline"},
                ),
                l2_content("근거를 사용한 최종 L2 답변 [cite-guideline]"),
            ]
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "mcp-call",
                    "index_get_page_content",
                    {"doc_id": "guideline-1"},
                ),
                l2_tool_call(
                    "final-call",
                    "finalize_retrieval",
                    {
                        "status": "sufficient",
                        "items": [
                            {"cite_uid": "cite-guideline", "relevance_score": 0.95}
                        ],
                        "note": "Applicable adult guideline evidence.",
                    },
                ),
            ]
        )
        mcp = FakeMcp()
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=mcp)  # type: ignore[arg-type]
        driver = ConversationDriver(settings, l2=generation_l2, retrieval=retrieval)  # type: ignore[arg-type]
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "CKD의 혈압 목표는?"},),
            ),
            request_id="req-rag",
        )
        self.assertEqual(result.trace["lane"], "GROUNDED")
        self.assertEqual(result.trace["retrieval_status"], "sufficient")
        self.assertEqual(result.trace["evidence_count"], 1)
        self.assertEqual(len(mcp.calls), 1)
        tool_message = next(
            message
            for message in generation_l2.calls[1]["messages"]
            if message.get("role") == "tool"
        )
        self.assertIn("cite-guideline", tool_message["content"])
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "Retrieval is complete" in str(message.get("content", ""))
                for message in generation_l2.calls[1]["messages"]
            )
        )
        self.assertIn("최종 L2 답변", result.content)

    def test_guideline_relevant_nodes_auto_transitions_to_page_content(self) -> None:
        # The bridge's page read no longer charges the model's tool-call allowance, but it
        # still counts toward whether there is room to carry on. With the discovery call and
        # the bridge read together filling the budget, the bridge returns immediately
        # instead of handing control back to the model.
        settings = Settings(
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=3,
            max_mcp_tool_calls=2,
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "mcp-call",
                    "index_get_relevant_nodes",
                    {"corpus_tag": "guideline", "query": "CKD 혈압 목표"},
                )
            ]
        )
        mcp = FakeMcp()
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=mcp)  # type: ignore[arg-type]

        run = retrieval.run(
            "CKD 혈압 목표 guideline",
            deadline=Deadline.after(3),
        )

        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.evidence[0].cite_uid, "cite-guideline")
        self.assertEqual(run.l2_calls, 1)
        self.assertEqual([call[0] for call in mcp.calls[:2]], [
            "index_get_relevant_nodes",
            "index_get_page_content",
        ])
        self.assertEqual(mcp.calls[1][1]["corpus_tag"], "guideline")
        self.assertEqual(mcp.calls[1][1]["doc_id"], "guideline-1")
        self.assertEqual(mcp.calls[1][1]["start_page"], 10)
        self.assertEqual(mcp.calls[1][1]["end_page"], 12)

    def test_bridge_returns_control_to_the_model_while_budget_remains(self) -> None:
        # The first page the bridge grabs is often only partly on topic, so with budget
        # left the model must get another turn to verify it, search the gap, or finalize.
        settings = Settings(
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=3,
            max_mcp_tool_calls=6,
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "mcp-call",
                    "index_get_relevant_nodes",
                    {"corpus_tag": "guideline", "query": "CKD 혈압 목표"},
                ),
                l2_tool_call(
                    "final-call",
                    "finalize_retrieval",
                    {
                        "status": "sufficient",
                        "items": [
                            {"cite_uid": "cite-guideline", "relevance_score": 0.95}
                        ],
                        "note": "The retrieved page answers the query.",
                    },
                ),
            ]
        )
        mcp = FakeMcp()
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=mcp)  # type: ignore[arg-type]

        run = retrieval.run("CKD 혈압 목표 guideline", deadline=Deadline.after(3))

        self.assertEqual(run.outcome.status, "sufficient")
        self.assertEqual(run.l2_calls, 2)
        self.assertEqual(run.outcome.evidence[0].cite_uid, "cite-guideline")

    def test_high_risk_review_can_request_one_revision(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            enable_high_risk_review=True,
            max_generation_retrievals=0,
        )
        l2 = ScriptedL2(
            [
                l2_content("초안"),
                l2_content("REVISE: 임신 관련 불확실성을 명확히 하세요."),
                l2_content("수정된 최종 L2 답변"),
            ]
        )
        driver = ConversationDriver(settings, l2=l2, retrieval=NeverRetrieval())  # type: ignore[arg-type]
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "임신 중 이 용량이 안전한가요?"},),
            ),
            request_id="req-review",
        )
        self.assertEqual(result.content, "수정된 최종 L2 답변")
        self.assertTrue(result.trace["reviewed"])
        self.assertTrue(result.trace["revised"])
        self.assertEqual(result.trace["lane"], "HIGH_RISK")

    def test_retrieval_without_finalizer_degrades_to_partial(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=2,
            max_mcp_tool_calls=1,
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "mcp-call",
                    "index_get_page_content",
                    {"doc_id": "guideline-1"},
                ),
                l2_content("I should have finalized."),
            ]
        )
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=FakeMcp())  # type: ignore[arg-type]
        run = retrieval.run(
            "self-contained guideline question",
            deadline=Deadline.after(3),
        )
        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.evidence[0].cite_uid, "cite-guideline")

    def test_retrieval_failure_still_returns_final_l2_answer(self) -> None:
        settings = Settings(enable_high_risk_review=False, lunit_fm_api_key="test")
        generation_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "gen-call",
                    "retrieve_relevant_content",
                    {"query": "근거가 필요한 질문"},
                ),
                l2_content("근거가 제한적이지만 안전 중심으로 답변합니다."),
            ]
        )

        class FailingRetrieval:
            def run(self, query: str, *, deadline):
                raise UpstreamError("retrieval failed", code="mcp_request_failed")

        driver = ConversationDriver(
            settings,
            l2=generation_l2,
            retrieval=FailingRetrieval(),  # type: ignore[arg-type]
        )
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "이 약의 허가 적응증은?"},),
            ),
            request_id="req-retrieval-failure",
        )
        self.assertEqual(result.content, "근거가 제한적이지만 안전 중심으로 답변합니다.")
        tool_message = next(
            message
            for message in generation_l2.calls[1]["messages"]
            if message.get("role") == "tool"
        )
        self.assertIn('"status": "no_evidence"', tool_message["content"])

    def test_retrieval_uses_its_own_stage_budget_without_global_deadline(self) -> None:
        settings = Settings(enable_high_risk_review=False, 
            lunit_fm_api_key="test",
            request_timeout_sec=None,  # type: ignore[arg-type]
            retrieval_timeout_sec=0.05,
        )
        generation_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "gen-call",
                    "retrieve_relevant_content",
                    {"query": "근거가 필요한 질문"},
                ),
                l2_content("검색 실패 후에도 L2가 생성한 최종 답변"),
            ]
        )

        class BudgetCapturingRetrieval:
            remaining = None

            def run(self, query: str, *, deadline):
                self.remaining = deadline.remaining(1.0)
                raise UpstreamError("retrieval failed", code="retrieval_budget_test")

        retrieval = BudgetCapturingRetrieval()
        driver = ConversationDriver(
            settings,
            l2=generation_l2,
            retrieval=retrieval,  # type: ignore[arg-type]
        )
        result = driver.complete(
            ChatCompletionRequest(
                model=settings.model,
                messages=({"role": "user", "content": "이 약의 허가 적응증은?"},),
            ),
            request_id="req-retrieval-budget",
        )

        self.assertIsNotNone(retrieval.remaining)
        self.assertGreater(retrieval.remaining, 0)
        self.assertLessEqual(retrieval.remaining, 0.05)
        self.assertEqual(result.content, "검색 실패 후에도 L2가 생성한 최종 답변")


if __name__ == "__main__":
    unittest.main()

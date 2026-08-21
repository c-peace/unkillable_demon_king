from __future__ import annotations

import unittest

from app.clients.mcp import McpTool
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.deadline import Deadline
from app.orchestration.driver import ConversationDriver
from app.orchestration.retrieval import RetrievalEngine
from tests.fakes import NeverRetrieval, ScriptedL2, l2_content, l2_tool_call


class FakeMcp:
    def __init__(self) -> None:
        self.calls = []

    def list_tools(self, *, deadline):
        return (
            McpTool(
                "index_get_page_content",
                "Read selected guideline pages",
                {
                    "type": "object",
                    "properties": {"doc_id": {"type": "string"}},
                    "required": ["doc_id"],
                },
            ),
        )

    def call_tool(self, name, arguments, *, deadline):
        self.calls.append((name, arguments))
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
        settings = Settings(lunit_fm_api_key="test")
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

    def test_generation_retrieval_generation_path(self) -> None:
        settings = Settings(lunit_fm_api_key="test", max_retrieval_model_rounds=3)
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
        tool_message = generation_l2.calls[1]["messages"][-1]
        self.assertIn("cite-guideline", tool_message["content"])
        self.assertIn("최종 L2 답변", result.content)

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


if __name__ == "__main__":
    unittest.main()

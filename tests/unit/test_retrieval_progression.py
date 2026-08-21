from __future__ import annotations

import unittest

from app.clients.mcp import McpTool
from app.config import Settings
from app.deadline import Deadline
from app.orchestration.retrieval import RetrievalEngine
from tests.fakes import ScriptedL2, l2_tool_call


class ProgressionMcp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_tools(self, *, deadline):
        return (
            McpTool(
                "index_get_relevant_nodes",
                "Find the most relevant indexed nodes",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "node_id": {"type": ["string", "null"]},
                    },
                    "required": ["query", "node_id"],
                },
            ),
            McpTool(
                "index_get_page_content",
                "Read the source pages for a selected document",
                {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["doc_id", "range"],
                },
            ),
        )

    def call_tool(self, name, arguments, *, deadline):
        normalized = dict(arguments)
        self.calls.append((name, normalized))
        if name == "index_get_relevant_nodes":
            return {
                "structuredContent": {
                    "result": [
                        {
                            "doc_id": "guideline-1",
                            "title": "CKD Blood Pressure Guideline",
                            "range": [12, 13],
                            "snippet": "Adults with CKD should target under 130/80.",
                        }
                    ]
                }
            }
        if name == "index_get_page_content":
            return {
                "structuredContent": {
                    "cite_uid": "cite-guideline-page",
                    "title": "CKD Blood Pressure Guideline",
                    "pages": [
                        {"page": 12, "text": "Adults with CKD should target under 130/80."}
                    ],
                }
            }
        raise AssertionError(f"Unexpected MCP tool: {name}")


class RetrievalProgressionTests(unittest.TestCase):
    def test_repeated_identical_discovery_calls_do_not_consume_mcp_budget(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=4,
            max_mcp_tool_calls=3,
            mcp_tool_mode="all",
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "nodes-1",
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                l2_tool_call(
                    "nodes-2",
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                l2_tool_call(
                    "pages-1",
                    "index_get_page_content",
                    {"doc_id": "guideline-1", "range": [12, 13]},
                ),
                l2_tool_call(
                    "final-1",
                    "finalize_retrieval",
                    {
                        "status": "sufficient",
                        "items": [
                            {"cite_uid": "cite-guideline-page", "relevance_score": 0.93}
                        ],
                        "note": "Page content confirms the guideline target.",
                    },
                ),
            ]
        )
        mcp = ProgressionMcp()
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=mcp)  # type: ignore[arg-type]

        run = retrieval.run(
            "What is the adult CKD blood pressure target guideline?",
            deadline=Deadline.after(3),
        )

        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.evidence[0].cite_uid, "cite-guideline-page")
        self.assertEqual(run.outcome.mcp_calls, 2)
        self.assertEqual(run.l2_calls, 1)
        self.assertEqual(
            mcp.calls,
            [
                (
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                (
                    "index_get_page_content",
                    {"doc_id": "guideline-1", "range": [12, 13]},
                ),
            ],
        )

    def test_successful_node_result_progresses_to_page_read_without_repeating_discovery(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            max_retrieval_model_rounds=3,
            max_mcp_tool_calls=2,
            mcp_tool_mode="all",
        )
        retrieval_l2 = ScriptedL2(
            [
                l2_tool_call(
                    "nodes-1",
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                l2_tool_call(
                    "nodes-2",
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                l2_tool_call(
                    "nodes-3",
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
            ]
        )
        mcp = ProgressionMcp()
        retrieval = RetrievalEngine(settings, l2=retrieval_l2, mcp=mcp)  # type: ignore[arg-type]

        run = retrieval.run(
            "What is the adult CKD blood pressure target guideline?",
            deadline=Deadline.after(3),
        )

        self.assertEqual(run.outcome.status, "partial")
        self.assertEqual(run.outcome.evidence[0].cite_uid, "cite-guideline-page")
        self.assertEqual(
            mcp.calls,
            [
                (
                    "index_get_relevant_nodes",
                    {"query": "CKD adult blood pressure target guideline", "node_id": None},
                ),
                (
                    "index_get_page_content",
                    {"doc_id": "guideline-1", "range": [12, 13]},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

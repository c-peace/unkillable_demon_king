from __future__ import annotations

import json
import unittest

from app.clients.mcp import McpTool
from app.conversation import compile_conversation
from app.evidence.models import (
    EvidenceRegistry,
    EvidenceRequirement,
    fallback_selection,
    parse_final_selection,
)
from app.evidence.routing import SourceRouter


class ConversationTests(unittest.TestCase):
    def test_compiler_preserves_turns_and_detects_high_risk(self) -> None:
        compiled = compile_conversation(
            [
                {"role": "user", "content": "와파린 5mg을 복용합니다."},
                {"role": "assistant", "content": "확인했습니다."},
                {"role": "user", "content": "임신 중 용량을 바꿔야 하나요?"},
            ]
        )
        self.assertIn("와파린 5mg", compiled.case_packet)
        self.assertEqual(compiled.latest_user_text, "임신 중 용량을 바꿔야 하나요?")
        self.assertTrue(compiled.is_high_risk)
        self.assertEqual(compiled.history_hash, compile_conversation(compiled.messages).history_hash)


class EvidenceTests(unittest.TestCase):
    def test_registry_finds_citations_inside_mcp_text_blocks(self) -> None:
        registry = EvidenceRegistry()
        payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "items": [
                                {
                                    "cite_uid": "cite-1",
                                    "title": "Guideline",
                                    "content": "A supported recommendation.",
                                }
                            ]
                        }
                    ),
                }
            ]
        }
        registered = registry.register_payload(payload, source_tool="index_get_page_content")
        self.assertEqual(registered, ("cite-1",))
        self.assertEqual(registry.get("cite-1").title, "Guideline")  # type: ignore[union-attr]

    def test_registry_normalizes_live_page_content_shape(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {
                "structuredContent": {
                    "cite_uid": "cite-pages",
                    "title": "Guideline",
                    "url": "https://example.test/guideline",
                    "pages": [
                        {"page": 10, "text": "first page evidence"},
                        {"page": 11, "text": "second page evidence"},
                    ],
                }
            },
            source_tool="index_get_page_content",
        )
        item = registry.get("cite-pages")
        self.assertIsNotNone(item)
        self.assertIn("[page 10]", item.content)  # type: ignore[union-attr]
        self.assertIn("second page evidence", item.content)  # type: ignore[union-attr]

    def test_unknown_citation_cannot_produce_sufficient(self) -> None:
        registry = EvidenceRegistry()
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "missing", "relevance_score": 0.99}],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
        )
        self.assertEqual(outcome.status, "no_evidence")
        self.assertIn("rejected", outcome.note)

    def test_any_unknown_selected_citation_downgrades_sufficient(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "known", "content": "support"}, source_tool="tool"
        )
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [
                    {"cite_uid": "known", "relevance_score": 0.9},
                    {"cite_uid": "unknown", "relevance_score": 0.8},
                ],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
        )
        self.assertEqual(outcome.status, "partial")

    def test_unknown_requirement_id_downgrades_sufficient(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "known", "content": "support"},
            source_tool="tool",
        )
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "known", "relevance_score": 0.9}],
                "requirements": [
                    {
                        "id": "req-2",
                        "status": "supported",
                        "cite_uids": ["known"],
                    }
                ],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
        )
        self.assertEqual(outcome.status, "partial")
        self.assertIn("Unknown requirement ids were rejected", outcome.note)
        self.assertEqual(outcome.requirement.status, "unresolved")

    def test_requirement_citation_must_be_selected_and_known(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "known", "content": "support"},
            source_tool="tool",
        )
        registry.register_payload(
            {"cite_uid": "other", "content": "support"},
            source_tool="tool",
        )
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "known", "relevance_score": 0.9}],
                "requirements": [
                    {
                        "id": "req-1",
                        "status": "supported",
                        "cite_uids": ["other"],
                    }
                ],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
        )
        self.assertEqual(outcome.status, "partial")
        self.assertIn("Requirement cite_uids were rejected", outcome.note)
        self.assertEqual(outcome.requirement.status, "unresolved")

    def test_sufficient_requires_all_critical_requirements_closed(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "known", "content": "support"},
            source_tool="tool",
        )
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "known", "relevance_score": 0.9}],
                "requirements": [
                    {
                        "id": "req-1",
                        "status": "supported",
                        "cite_uids": ["known"],
                    }
                ],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
            requirements=(
                EvidenceRequirement(id="req-1", claim_or_question="question"),
                EvidenceRequirement(id="req-2", claim_or_question="contraindication check"),
            ),
        )
        self.assertEqual(outcome.status, "partial")
        self.assertEqual(
            tuple(requirement.status for requirement in outcome.requirements),
            ("supported", "unresolved"),
        )
        self.assertIn("Critical requirements remained unresolved: req-2", outcome.note)

    def test_contradicted_critical_requirement_cannot_be_globally_sufficient(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "known", "content": "authoritative contradiction"},
            source_tool="tool",
        )
        outcome = parse_final_selection(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "known", "relevance_score": 0.9}],
                "requirements": [
                    {
                        "id": "req-1",
                        "status": "contradicted",
                        "cite_uids": ["known"],
                        "gap_reason": "Authoritative sources conflict with the claim.",
                    }
                ],
            },
            registry=registry,
            query="question",
            model_rounds=1,
            mcp_calls=1,
            max_items=8,
        )

        self.assertEqual(outcome.status, "partial")
        self.assertEqual(outcome.requirement.status, "contradicted")

    def test_budget_fallback_does_not_promote_unadjudicated_registry_items(self) -> None:
        registry = EvidenceRegistry()
        registry.register_payload(
            {"cite_uid": "cite-2", "content": "partial support"},
            source_tool="tool",
        )
        outcome = fallback_selection(
            registry=registry,
            query="question",
            note="budget ended",
            model_rounds=2,
            mcp_calls=2,
            max_items=8,
        )
        self.assertEqual(outcome.status, "no_evidence")
        self.assertFalse(outcome.evidence)


class RoutingTests(unittest.TestCase):
    def test_family_router_selects_authoritative_drug_tools(self) -> None:
        tools = (
            McpTool("adr_retrieve_drug_info", "", {}),
            McpTool("openapi_law_search", "", {}),
            McpTool("openapi_mfds_get_drug_indication", "", {}),
        )
        selected = SourceRouter("family").select("약물 상호작용과 적응증", tools)
        self.assertEqual(
            {tool.name for tool in selected},
            {"adr_retrieve_drug_info", "openapi_mfds_get_drug_indication"},
        )

    def test_guideline_router_starts_at_relevant_nodes_not_document_listing(self) -> None:
        tools = (
            McpTool("index_list_documents", "", {}),
            McpTool("index_get_relevant_nodes", "", {}),
            McpTool("index_get_page_content", "", {}),
        )
        selected = SourceRouter("family").select("CKD 혈압 목표 guideline", tools)
        self.assertEqual(
            [tool.name for tool in selected],
            ["index_get_relevant_nodes", "index_get_page_content"],
        )


if __name__ == "__main__":
    unittest.main()

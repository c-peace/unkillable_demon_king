from __future__ import annotations

import re
from collections.abc import Sequence

from app.clients.mcp import McpTool


CORE_FALLBACK_TOOL_NAMES: tuple[str, ...] = (
    "index_get_relevant_nodes",
    "index_get_page_content",
    "adr_retrieve_drug_info",
    "openapi_mfds_check_drug_permission",
    "openapi_mfds_get_drug_indication",
    "openapi_hira_get_drug_price",
    "kcd_search_codes",
    "kcd_get_name",
)


ROUTING_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"가이드라인|권고|진료지침|guideline|recommendation|target|goal|treat(?:ment)?\s+target",
            re.IGNORECASE,
        ),
        (
            "index_get_relevant_nodes",
            "index_get_page_content",
        ),
    ),
    (
        re.compile(
            r"상호작용|병용|금기|부작용|경고|adverse|interaction|contraindication|warning",
            re.IGNORECASE,
        ),
        (
            "adr_retrieve_drug_info",
            "openapi_mfds_get_drug_indication",
            "openapi_mfds_check_drug_permission",
        ),
    ),
    (
        re.compile(
            r"약물|의약품|복용|drug|medication|dose|dosage|indication|approval|mfds|허가|적응증",
            re.IGNORECASE,
        ),
        (
            "adr_retrieve_drug_info",
            "openapi_mfds_get_drug_indication",
            "openapi_mfds_check_drug_permission",
            "openapi_mfds_find_drugs_by_ingredient",
        ),
    ),
    (
        re.compile(
            r"급여|보험|약가|심평원|hira|reimbursement|coverage|price",
            re.IGNORECASE,
        ),
        (
            "hira_updates_search",
            "openapi_hira_get_drug_price",
            "index_get_relevant_nodes",
            "index_get_page_content",
        ),
    ),
    (
        re.compile(r"kcd|질병코드|상병코드|진단코드|disease code", re.IGNORECASE),
        (
            "kcd_search_codes",
            "kcd_get_name",
            "openapi_hira_disease_check_code",
        ),
    ),
    (
        # "의료법상 ..." style questions name the statute directly and never contain the
        # words 법령/법률, so match a named act as well as the generic vocabulary.
        re.compile(
            r"의료법|약사법|의료기기법|개인정보\s*보호법|국민건강보험법|"
            r"[가-힣]{2,}법상|법령|법률|조문|시행령|시행규칙|제\s*\d+\s*조|"
            r"law|statute|regulation",
            re.IGNORECASE,
        ),
        (
            "openapi_law_search",
            "openapi_law_list_articles",
            "openapi_law_get_article",
        ),
    ),
    (
        re.compile(r"논문|연구|pubmed|study|trial|systematic review|meta-analysis", re.IGNORECASE),
        (
            "rag_get_all_data_sources",
            "rag_get_data_source_detail",
            "rag_vector_query",
        ),
    ),
    (
        re.compile(r"faers|이상반응\s*신호|signal detection|adverse event signal", re.IGNORECASE),
        (
            "rag_get_all_data_sources",
            "rag_get_data_source_detail",
            "rag_sql_query",
            "adr_retrieve_drug_info",
        ),
    ),
)


TOOL_PRIORITY: dict[str, int] = {
    "index_list_documents": 10,
    "index_get_relevant_nodes": 11,
    "index_get_page_content": 12,
    "adr_retrieve_drug_info": 20,
    "openapi_mfds_check_drug_permission": 21,
    "openapi_mfds_get_drug_indication": 22,
    "openapi_mfds_find_drugs_by_ingredient": 23,
    "hira_updates_search": 30,
    "openapi_hira_get_drug_price": 31,
    "openapi_hira_disease_check_code": 32,
    "kcd_search_codes": 40,
    "kcd_get_name": 41,
    "openapi_law_search": 50,
    "openapi_law_list_articles": 51,
    "openapi_law_get_article": 52,
    "rag_get_all_data_sources": 60,
    "rag_get_data_source_detail": 61,
    "rag_vector_query": 62,
    "rag_sql_query": 63,
}


class SourceRouter:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def select(self, query: str, tools: Sequence[McpTool]) -> tuple[McpTool, ...]:
        if self._mode == "all":
            return tuple(tools)
        selected_names = self._match_tool_names(query)
        selected = self._select_names(selected_names, tools)
        return selected or self._select_names(CORE_FALLBACK_TOOL_NAMES, tools) or tuple(tools)

    def _match_tool_names(self, query: str) -> tuple[str, ...]:
        matched: list[str] = []
        for pattern, tool_names in ROUTING_RULES:
            if pattern.search(query):
                matched.extend(tool_names)
        return tuple(dict.fromkeys(matched))

    @staticmethod
    def _select_names(names: Sequence[str], tools: Sequence[McpTool]) -> tuple[McpTool, ...]:
        if not names:
            return ()
        selected = [tool for tool in tools if tool.name in set(names)]
        return tuple(sorted(selected, key=lambda tool: (TOOL_PRIORITY.get(tool.name, 999), tool.name)))

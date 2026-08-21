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
            # Patients do not say 상호작용 or 금기. They ask whether two medicines can be
            # taken together, or whether something is safe to take at all, so the natural
            # phrasings have to route here as well as the clinical vocabulary.
            r"상호작용|병용|금기|부작용|이상반응|경고|"
            r"같이\s*(먹|복용|드시|투여)|함께\s*(먹|복용|드시|투여)|동시에\s*(먹|복용)|"
            r"먹어도\s*(되|괜찮|안전)|복용해도\s*(되|괜찮|안전)|드셔도\s*(되|괜찮)|"
            r"먹으면\s*안|피해야|위험하지|괜찮을까|안전한가|"
            r"adverse|interaction|contraindication|warning|safe to take|"
            r"take\s+(?:\w+\s+){0,3}(?:together|with)|combine|"
            r"can i (?:take|use|have)|is it (?:ok|okay|safe)",
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


# The evidence domain each routing rule stands for, in the same order. Traced per request
# so failures can be read by domain, and the key any per-route tuning would use.
ROUTE_LABELS: tuple[str, ...] = (
    "guideline",
    "drug_safety",
    "drug",
    "hira",
    "coding",
    "law",
    "research",
    "faers",
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

SOURCE_FAMILY_TOOLS: dict[str, tuple[str, ...]] = {
    "guideline": ("index_get_relevant_nodes", "index_get_page_content"),
    "drug_label": (
        "adr_retrieve_drug_info",
        "openapi_mfds_get_drug_indication",
        "openapi_mfds_check_drug_permission",
    ),
    "approval": (
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_get_drug_indication",
    ),
    "reimbursement": (
        "hira_updates_search",
        "openapi_hira_get_drug_price",
        "index_get_relevant_nodes",
        "index_get_page_content",
    ),
    "coding": ("kcd_search_codes", "kcd_get_name", "openapi_hira_disease_check_code"),
    "law": ("openapi_law_search", "openapi_law_list_articles", "openapi_law_get_article"),
    "research": ("rag_get_all_data_sources", "rag_get_data_source_detail", "rag_vector_query"),
    "safety_signal": ("rag_get_all_data_sources", "rag_get_data_source_detail", "rag_sql_query"),
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

    def select_for_families(
        self,
        families: Sequence[str],
        tools: Sequence[McpTool],
        *,
        fallback_query: str,
    ) -> tuple[McpTool, ...]:
        names: list[str] = []
        for family in families:
            names.extend(SOURCE_FAMILY_TOOLS.get(family, ()))
        selected = self._select_names(tuple(dict.fromkeys(names)), tools)
        return selected or self.select(fallback_query, tools)

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

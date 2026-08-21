from __future__ import annotations

import re
from collections.abc import Sequence

from app.clients.mcp import McpTool


ROUTING_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"가이드라인|권고|guideline|recommendation", re.IGNORECASE),
        ("index_",),
    ),
    (
        re.compile(r"약물|의약품|복용|상호작용|drug|medication|interaction", re.IGNORECASE),
        ("adr_", "openapi_mfds_", "index_"),
    ),
    (
        re.compile(r"허가|적응증|식약처|mfds|approval|indication", re.IGNORECASE),
        ("openapi_mfds_",),
    ),
    (
        re.compile(r"급여|보험|약가|심평원|hira|reimbursement", re.IGNORECASE),
        ("hira_", "openapi_hira_", "index_"),
    ),
    (
        re.compile(r"kcd|질병코드|상병코드|disease code", re.IGNORECASE),
        ("kcd_", "openapi_hira_disease_"),
    ),
    (
        re.compile(r"법령|법률|조문|law|statute|regulation", re.IGNORECASE),
        ("openapi_law_",),
    ),
    (
        re.compile(r"논문|연구|근거|pubmed|study|evidence", re.IGNORECASE),
        ("rag_", "index_"),
    ),
    (
        re.compile(r"이상반응|부작용 신호|faers|adverse event", re.IGNORECASE),
        ("rag_", "adr_"),
    ),
)


class SourceRouter:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def select(self, query: str, tools: Sequence[McpTool]) -> tuple[McpTool, ...]:
        if self._mode == "all":
            return tuple(tools)
        prefixes: set[str] = set()
        for pattern, matched_prefixes in ROUTING_RULES:
            if pattern.search(query):
                prefixes.update(matched_prefixes)
        if not prefixes:
            return tuple(tools)
        selected = tuple(
            tool for tool in tools if any(tool.name.startswith(prefix) for prefix in prefixes)
        )
        return selected or tuple(tools)

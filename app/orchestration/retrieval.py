from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from app.clients.l2 import L2Client, parse_tool_arguments
from app.clients.mcp import McpClient, McpTool
from app.config import Settings
from app.deadline import Deadline
from app.errors import AppError
from app.evidence.models import (
    EvidenceRegistry,
    EvidenceRequirement,
    RetrievalOutcome,
    default_requirement_ledger,
    fallback_selection,
    parse_final_selection,
)
from app.evidence.routing import SourceRouter
from app.evidence.session import RetrievalSession
from app.prompts import RETRIEVAL_SYSTEM_PROMPT


LOGGER = logging.getLogger("lunit_retrieval")


VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# How often one MCP tool may run within a single retrieval before the model is told to
# switch tools or finalize. Repeats past this point were observed to return the same
# material while consuming the whole tool-call budget.
MAX_CALLS_PER_TOOL = 2

# Consecutive tool calls returning no citable evidence before retrieval is pushed to
# finalize. The organizers warn that a model floundering over tool calls is what
# produces very long timeouts, so stalling has to end the loop rather than extend it.
MAX_NO_PROGRESS_CALLS = 2


FINALIZE_RETRIEVAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finalize_retrieval",
        "description": (
            "Submit the final citation selection and end retrieval. Call once evidence is "
            "sufficient, retrieval is unnecessary, or the tool-call budget is exhausted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["sufficient", "partial", "no_evidence"],
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_uid": {"type": "string"},
                            "relevance_score": {"type": "number"},
                        },
                        "required": ["cite_uid", "relevance_score"],
                        "additionalProperties": False,
                    },
                    "default": [],
                },
                "selected_items": {
                    "type": "array",
                    "description": "Preferred alias for items; same shape and meaning.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_uid": {"type": "string"},
                            "relevance_score": {"type": "number"},
                        },
                        "required": ["cite_uid", "relevance_score"],
                        "additionalProperties": False,
                    },
                    "default": [],
                },
                "requirements": {
                    "type": "array",
                    "description": (
                        "Optional requirement-level closure for the harness ledger. Only use "
                        "known requirement ids and cite_uids already returned in selected items."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["missing", "supported", "contradicted", "unresolved"],
                            },
                            "cite_uids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "gap_reason": {"type": "string", "default": ""},
                        },
                        "required": ["id", "status"],
                        "additionalProperties": False,
                    },
                    "default": [],
                },
                "note": {"type": "string", "default": ""},
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class RetrievalRun:
    outcome: RetrievalOutcome
    usage: dict[str, int]
    l2_calls: int
    diagnostics: Mapping[str, Any] | None = None


def _sum_usage(total: dict[str, int], usage: Mapping[str, int]) -> None:
    for key, value in usage.items():
        if isinstance(value, int) and value >= 0:
            total[key] = total.get(key, 0) + value


def _tool_result_content(value: Any, maximum: int) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(value)
    if len(rendered) <= maximum:
        return rendered
    return json.dumps(
        {"truncated": True, "preview": rendered[:maximum]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tool_cache_key(name: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": dict(arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


_STOPWORDS = frozenset(
    "a an and are as at be by для for from in is of on or that the to what when with "
    "관한 그리고 대한 대해 또는 및 에서 으로 은 는 이 가 을 를 의 와 과 에".split()
)


# Korean attaches its particles to the word, so "트라스투주맙" and "트라스투주맙의" are one
# term written twice. Longest first so 에서 is not shortened to 서 by an earlier match.
_KO_PARTICLES = (
    "에서는", "에게서", "으로는", "이라는", "에서", "에게", "한테", "부터", "까지",
    "으로", "라는", "이나", "에는", "은", "는", "이", "가", "을", "를", "의",
    "와", "과", "로", "도", "만", "에",
)


def _strip_particle(token: str) -> str:
    if not token or not ("가" <= token[-1] <= "힣"):
        return token
    for particle in _KO_PARTICLES:
        if len(token) > len(particle) + 1 and token.endswith(particle):
            return token[: -len(particle)]
    return token


def _semantic_tokens(name: str, arguments: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    """The tool and the content words its arguments actually carry."""
    tokens: set[str] = set()
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            for raw in re.split(r"[^0-9a-z가-힣]+", value.lower()):
                stem = _strip_particle(raw)
                if stem and stem not in _STOPWORDS and len(stem) > 1:
                    tokens.add(stem)
        elif value is not None:
            tokens.add(f"{key}={value}")
    return name, frozenset(tokens)


def _is_near_duplicate(
    candidate: tuple[str, frozenset[str]],
    seen: list[tuple[str, frozenset[str]]],
    threshold: float = 0.85,
) -> bool:
    """Whether this search is a rewording of one already run.

    Exact token-set equality misses "warfarin interaction" against "warfarin ibuprofen
    interaction", which is the same enquiry with one word added, so compare by overlap.
    """
    name, tokens = candidate
    if not tokens:
        return False
    for seen_name, seen_tokens in seen:
        if seen_name != name or not seen_tokens:
            continue
        union = len(tokens | seen_tokens)
        if union and len(tokens & seen_tokens) / union >= threshold:
            return True
    return False


def _semantic_call_key(name: str, arguments: Mapping[str, Any]) -> str:
    """Normalize rewordings for diagnostics and regression coverage."""
    parts: list[str] = []
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            tokens = [t for t in re.split(r"[^0-9a-z가-힣]+", value.lower()) if t]
            tokens = [_strip_particle(t) for t in tokens]
            tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
            parts.append(f"{key}={' '.join(sorted(set(tokens)))}")
        elif value is not None:
            parts.append(f"{key}={value}")
    return f"{name}|" + "|".join(parts)


def _candidate_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Extract ranked index records from MCP JSON or text-block envelopes."""
    records: list[Mapping[str, Any]] = []

    def visit(candidate: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text.startswith(("{", "[")):
                return
            try:
                visit(json.loads(text), depth + 1)
            except json.JSONDecodeError:
                return
            return
        if isinstance(candidate, list):
            for item in candidate:
                visit(item, depth + 1)
            return
        if not isinstance(candidate, Mapping):
            return

        if any(
            isinstance(candidate.get(key), str)
            for key in ("doc_id", "document_id", "docId")
        ):
            records.append(candidate)
        for key in (
            "structuredContent",
            "structured_content",
            "result",
            "items",
            "item",
            "nodes",
            "node",
            "documents",
            "document",
            "matching_document",
            "matched_document",
            "matches",
            "match",
            "candidate",
            "candidates",
            "children",
            "ancestor_nodes",
            "ancestorNodes",
            "content",
            "data",
            "results",
            "rows",
        ):
            nested = candidate.get(key)
            if nested is not None:
                visit(nested, depth + 1)
        text = candidate.get("text")
        if isinstance(text, str):
            visit(text, depth + 1)

    visit(value)
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        identity = (str(record.get("doc_id", "")), str(record.get("node_id", "")))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(record)
    return tuple(unique)


def _page_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed >= 1 else None
    return None


def _index_page_arguments(
    result: Any,
    discovery_arguments: Mapping[str, Any],
    page_tool: McpTool,
) -> dict[str, Any] | None:
    properties = page_tool.input_schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    for record in _candidate_records(result):
        doc_id = next(
            (
                record.get(key)
                for key in ("doc_id", "document_id", "docId")
                if isinstance(record.get(key), str)
            ),
            None,
        )
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        corpus_tag = next(
            (
                record.get(key)
                for key in ("corpus_tag", "source_type", "source", "data_source")
                if isinstance(record.get(key), str) and record.get(key).strip()
            ),
            None,
        )
        page_range = record.get(
            "range",
            record.get("page_range", record.get("pages", record.get("pageRange"))),
        )
        start = record.get(
            "start_page",
            record.get("page_start", record.get("begin_page", record.get("from_page"))),
        )
        end = record.get(
            "end_page",
            record.get("page_end", record.get("last_page", record.get("to_page"))),
        )
        if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
            start, end = page_range[0], page_range[1]
        elif isinstance(page_range, Mapping):
            start = page_range.get("start", start)
            end = page_range.get("end", end)
        elif isinstance(page_range, str):
            match = re.fullmatch(r"\s*(\d+)\s*[-~]\s*(\d+)\s*", page_range)
            if match:
                start, end = match.group(1), match.group(2)
        start_page = _page_number(start)
        end_page = _page_number(end)
        if start_page is None or end_page is None or end_page < start_page:
            continue
        end_page = min(end_page, start_page + 19)
        arguments: dict[str, Any] = {"doc_id": doc_id.strip()}
        if "corpus_tag" in properties:
            if not isinstance(corpus_tag, str) or not corpus_tag.strip():
                corpus_tag = discovery_arguments.get("corpus_tag")
            if not isinstance(corpus_tag, str) or not corpus_tag.strip():
                return None
            arguments["corpus_tag"] = corpus_tag.strip()
        if "range" in properties:
            arguments["range"] = [start_page, end_page]
            return arguments
        if "start_page" in properties:
            arguments["start_page"] = start_page
        elif "page_start" in properties:
            arguments["page_start"] = start_page
        else:
            return None
        if "end_page" in properties:
            arguments["end_page"] = end_page
        elif "page_end" in properties:
            arguments["page_end"] = end_page
        else:
            return None
        return arguments
    return None


def _compact_candidates(value: Any, limit: int = 3) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    fields = (
        "doc_id",
        "node_id",
        "title",
        "doc_title",
        "range",
        "provider",
        "source_url",
        "score",
        "summary",
    )
    for record in _candidate_records(value)[:limit]:
        item = {key: record[key] for key in fields if key in record}
        summary = item.get("summary")
        if isinstance(summary, str) and len(summary) > 800:
            item["summary"] = summary[:800] + "…"
        compact.append(item)
    return compact


def _registered_evidence_payload(
    registry: EvidenceRegistry,
    cite_uids: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cite_uid in cite_uids:
        item = registry.get(cite_uid)
        if item is None:
            continue
        records.append(
            {
                "cite_uid": item.cite_uid,
                "title": item.title,
                "url": item.url,
                "source_type": item.source_type,
                "content": (item.content or item.raw_preview)[:4_000],
            }
        )
    return records


def _retrieval_user_message(
    query: str,
    *,
    max_model_rounds: int,
    max_mcp_tool_calls: int,
    selected_tools: tuple[McpTool, ...],
    requirements: tuple[EvidenceRequirement, ...],
) -> str:
    tool_list = ", ".join(tool.name for tool in selected_tools) if selected_tools else "none"
    selected_names = {tool.name for tool in selected_tools}
    index_path = ""
    if {"index_get_relevant_nodes", "index_get_page_content"}.issubset(selected_names):
        index_path = (
            "For an indexed guideline or HIRA question, call index_get_relevant_nodes once "
            "with the appropriate corpus_tag and query. The harness will automatically fetch "
            "page content for the best returned document when a page range is available, so "
            "do not repeat document discovery with reformulated queries.\n"
        )
    return (
        "Retrieve evidence for this self-contained question:\n"
        f"{query}\n\n"
        "The harness ledger starts with these requirements:\n"
        + "\n".join(
            f"- {requirement.id} [{requirement.criticality}] {requirement.claim_or_question}"
            for requirement in requirements
        )
        + "\n"
        "If you provide requirements in finalize_retrieval, update only known ids and cite "
        "only cite_uids already selected in items or selected_items. sufficient is valid "
        "only when every critical requirement is closed.\n"
        "Use the shortest authoritative path for the named evidence gap. "
        "Prefer exact official tools or exact page reads over broad exploratory search.\n"
        f"{index_path}"
        f"Available MCP tools: {tool_list}\n"
        f"Maximum retrieval model rounds: {max_model_rounds}\n"
        f"Maximum MCP tool calls: {max_mcp_tool_calls}\n"
        "End with finalize_retrieval."
    )


class RetrievalEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        l2: L2Client,
        mcp: McpClient,
    ) -> None:
        self._settings = settings
        self._l2 = l2
        self._mcp = mcp
        self._router = SourceRouter(settings.mcp_tool_mode)

    def run(
        self,
        query: str,
        *,
        deadline: Deadline,
        requirements: tuple[EvidenceRequirement, ...] | None = None,
        session: RetrievalSession | None = None,
    ) -> RetrievalRun:
        registry = session.registry if session is not None else EvidenceRegistry()
        usage: dict[str, int] = {}
        if session is not None:
            target_requirements = session.target_requirements(requirements or ())
            ledger = default_requirement_ledger(query, requirements=target_requirements)
        else:
            ledger = default_requirement_ledger(query, requirements=requirements)
        prior_model_rounds = session.cumulative_model_rounds if session is not None else 0
        prior_mcp_calls = session.cumulative_mcp_calls if session is not None else 0
        available_model_rounds = max(
            0,
            self._settings.max_retrieval_model_rounds - prior_model_rounds,
        )
        available_mcp_calls = max(
            0,
            self._settings.max_mcp_tool_calls - prior_mcp_calls,
        )
        if available_model_rounds == 0:
            return RetrievalRun(
                outcome=fallback_selection(
                    registry=registry,
                    query=query,
                    requirements=ledger.requirements,
                    note="The cumulative retrieval model-round budget was exhausted.",
                    model_rounds=0,
                    mcp_calls=0,
                    max_items=self._settings.max_evidence_items,
                ),
                usage=usage,
                l2_calls=0,
                diagnostics={"stop_reason": "model_budget_exhausted"},
            )
        if not self._settings.enable_mcp:
            return RetrievalRun(
                outcome=fallback_selection(
                    registry=registry,
                    query=query,
                    requirements=ledger.requirements,
                    note="MCP retrieval is disabled.",
                    model_rounds=0,
                    mcp_calls=0,
                    max_items=self._settings.max_evidence_items,
                ),
                usage=usage,
                l2_calls=0,
            )

        try:
            discovered = self._mcp.list_tools(deadline=deadline)
        except AppError:
            return RetrievalRun(
                outcome=fallback_selection(
                    registry=registry,
                    query=query,
                    requirements=ledger.requirements,
                    note="The evidence service was unavailable.",
                    model_rounds=0,
                    mcp_calls=0,
                    max_items=self._settings.max_evidence_items,
                ),
                usage=usage,
                l2_calls=0,
            )
        preferred_domains = tuple(
            dict.fromkeys(
                requirement.source_preference
                for requirement in ledger.requirements
                if requirement.source_preference
            )
        )
        routed_tools = self._router.select_for_domains(
            preferred_domains,
            discovered,
            fallback_query=query,
        )
        selected_tools = tuple(
            tool
            for tool in routed_tools
            if VALID_TOOL_NAME.fullmatch(tool.name)
        )
        LOGGER.info(
            "retrieval_tools_selected count=%s tools=%s",
            len(selected_tools),
            ",".join(tool.name for tool in selected_tools),
        )
        tool_by_name = {tool.name: tool for tool in selected_tools}
        page_content_tool = tool_by_name.get("index_get_page_content")
        retrieval_tools = [tool.as_openai_tool() for tool in selected_tools]
        retrieval_tools.append(FINALIZE_RETRIEVAL_TOOL)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _retrieval_user_message(
                    query,
                    max_model_rounds=available_model_rounds,
                    max_mcp_tool_calls=available_mcp_calls,
                    selected_tools=selected_tools,
                    requirements=ledger.requirements,
                ),
            },
        ]
        model_rounds = 0
        mcp_calls = 0
        finalizer_nudged = False
        mcp_budget_exhausted = False
        tool_cache = session.exact_call_cache if session is not None else {}
        tool_use_counts: dict[str, int] = {}
        # The deterministic page bridge is our own chaining, not the model spending its
        # budget, so it is counted separately and reported without charging the model.
        bridge_calls = 0
        seen_semantic = session.semantic_call_history if session is not None else []
        duplicate_blocked = 0
        invalid_calls = 0
        no_progress = session.no_progress if session is not None else 0

        while model_rounds < available_model_rounds:
            if not deadline.can_start(0.25):
                break
            # Keep searching while budget allows, even once citable evidence exists: the
            # first hit is often only partially on-topic, and retrieval_tools already
            # carries finalize_retrieval so the model can stop as soon as it judges the
            # evidence sufficient. Reserve the last round for finalize only.
            search_budget_left = (
                not mcp_budget_exhausted
                and mcp_calls + bridge_calls < available_mcp_calls
                and model_rounds < available_model_rounds - 1
                # Two searches in a row returning nothing citable means this line of enquiry
                # is not paying off; more of it burns the budget and the clock.
                and no_progress < MAX_NO_PROGRESS_CALLS
            )
            round_tools = retrieval_tools if search_budget_left else [FINALIZE_RETRIEVAL_TOOL]
            allowed_tool_names = {
                tool["function"]["name"]
                for tool in round_tools
                if isinstance(tool.get("function"), Mapping)
            }
            response = self._l2.complete(messages, deadline=deadline, tools=round_tools)
            model_rounds += 1
            _sum_usage(usage, response.usage)
            messages.append(response.assistant_message)

            if not response.tool_calls:
                if finalizer_nudged:
                    break
                remaining_rounds = available_model_rounds - model_rounds
                remaining_calls = max(0, available_mcp_calls - mcp_calls - bridge_calls)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do not answer in prose. Continue only for a named evidence gap, or call "
                            "finalize_retrieval now with the best valid cite_uids gathered so far. "
                            f"Remaining model rounds: {remaining_rounds}. "
                            f"Remaining MCP tool calls: {remaining_calls}."
                        ),
                    }
                )
                finalizer_nudged = True
                continue

            should_stop_after_turn = False
            for call in response.tool_calls:
                if call.name not in allowed_tool_names:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {
                                    "error": "tool_not_available_in_current_retrieval_phase",
                                    "instruction": (
                                        "Use only the tools currently provided. If citable evidence "
                                        "is available, call finalize_retrieval."
                                    ),
                                }
                            ),
                        }
                    )
                    continue
                if call.name == "finalize_retrieval":
                    try:
                        arguments = parse_tool_arguments(call.arguments)
                        outcome = parse_final_selection(
                            arguments,
                            registry=registry,
                            query=query,
                            requirements=ledger.requirements,
                            model_rounds=model_rounds,
                            mcp_calls=mcp_calls + bridge_calls,
                            max_items=self._settings.max_evidence_items,
                        )
                    except (ValueError, json.JSONDecodeError) as exc:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": json.dumps(
                                    {
                                        "error": "invalid_finalize_retrieval",
                                        "detail": str(exc),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue
                    return RetrievalRun(
                        outcome=outcome,
                        usage=usage,
                        l2_calls=model_rounds,
                        diagnostics={
                            "stop_reason": "finalized",
                            "bridge_calls": bridge_calls,
                            "duplicate_blocked": duplicate_blocked,
                            "invalid_calls": invalid_calls,
                            "no_progress": no_progress,
                        },
                    )

                tool = tool_by_name.get(call.name)
                if tool is None:
                    invalid_calls += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {"error": "unknown_or_unavailable_tool", "tool": call.name}
                            ),
                        }
                    )
                    continue
                if mcp_calls + bridge_calls >= available_mcp_calls:
                    mcp_budget_exhausted = True
                    should_stop_after_turn = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({"error": "mcp_tool_call_budget_exhausted"}),
                        }
                    )
                    continue
                # The model otherwise re-runs a tool that is not paying off — observed as
                # four consecutive hira_updates_search calls, and as three futile
                # relevant_nodes/page_content cycles that burned the whole budget.
                used_before = tool_use_counts.get(tool.name, 0)
                if used_before >= MAX_CALLS_PER_TOOL:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {
                                    "error": "tool_repeat_limit_reached",
                                    "instruction": (
                                        f"{tool.name} has already run {used_before} times and is "
                                        "not yielding new evidence. Use a different tool for the "
                                        "remaining gap, or call finalize_retrieval now."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                try:
                    arguments = parse_tool_arguments(call.arguments)
                except (ValueError, json.JSONDecodeError):
                    invalid_calls += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({"error": "invalid_tool_arguments"}),
                        }
                    )
                    continue
                cache_key = _tool_cache_key(tool.name, arguments)
                cached = tool_cache.get(cache_key)
                if cached is not None:
                    LOGGER.info("retrieval_tool_cache_hit tool=%s", tool.name)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": cached,
                        }
                    )
                    continue
                semantic_tokens = _semantic_tokens(tool.name, arguments)
                if _is_near_duplicate(semantic_tokens, seen_semantic):
                    # Same search, different wording. Running it again costs a tool call and
                    # returns what we already hold, so refuse and push toward finalizing.
                    duplicate_blocked += 1
                    LOGGER.info(
                        "retrieval_duplicate_blocked tool=%s total=%s",
                        tool.name,
                        duplicate_blocked,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {
                                    "error": "already_searched",
                                    "instruction": (
                                        "You already ran this search. Use the evidence you "
                                        "have, search a genuinely different evidence need, "
                                        "or call finalize_retrieval now."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                seen_semantic.append(semantic_tokens)
                try:
                    known_citations = {item.cite_uid for item in registry.all()}
                    result = self._mcp.call_tool(
                        tool.name,
                        arguments,
                        deadline=deadline,
                    )
                    mcp_calls += 1
                    tool_use_counts[tool.name] = used_before + 1
                    cite_uids = registry.register_payload(result, source_tool=tool.name)
                    new_cite_uids = tuple(
                        cite_uid for cite_uid in cite_uids if cite_uid not in known_citations
                    )
                    # index discovery legitimately returns candidates rather than citable
                    # evidence and the bridge below turns it into pages, so it is not a
                    # stall. Any other tool that yields nothing is.
                    if new_cite_uids or tool.name == "index_get_relevant_nodes":
                        no_progress = 0
                    else:
                        no_progress += 1
                    LOGGER.info(
                        "retrieval_tool_completed tool=%s call=%s citations=%s",
                        tool.name,
                        mcp_calls,
                        len(cite_uids),
                    )
                    if tool.name == "index_get_page_content" and cite_uids:
                        tool_content = {
                            "registered_cite_uids": list(cite_uids),
                            "evidence": _registered_evidence_payload(registry, cite_uids),
                            "instruction": (
                                "Check this evidence against the query. If it covers every "
                                "evidence requirement, call finalize_retrieval with the relevant "
                                "cite_uids. If it is off-topic or a requirement is still "
                                "unmet, search once more for that specific gap."
                            ),
                        }
                    elif tool.name == "index_get_relevant_nodes":
                        tool_content = {
                            "registered_cite_uids": list(cite_uids),
                            "candidates": _compact_candidates(result),
                        }
                    else:
                        tool_content = {
                            "registered_cite_uids": list(cite_uids),
                            "result": result,
                        }

                    if (
                        tool.name == "index_get_relevant_nodes"
                        and not cite_uids
                        and page_content_tool is not None
                        and mcp_calls + bridge_calls < available_mcp_calls
                        and tool_use_counts.get(page_content_tool.name, 0)
                        < MAX_CALLS_PER_TOOL
                    ):
                        page_arguments = _index_page_arguments(
                            result,
                            arguments,
                            page_content_tool,
                        )
                        if page_arguments is not None:
                            page_cache_key = _tool_cache_key(
                                page_content_tool.name,
                                page_arguments,
                            )
                            try:
                                page_result = self._mcp.call_tool(
                                    page_content_tool.name,
                                    page_arguments,
                                    deadline=deadline,
                                )
                                bridge_calls += 1
                                tool_use_counts[page_content_tool.name] = (
                                    tool_use_counts.get(page_content_tool.name, 0) + 1
                                )
                                page_cite_uids = registry.register_payload(
                                    page_result,
                                    source_tool=page_content_tool.name,
                                )
                                new_page_cite_uids = tuple(
                                    cite_uid
                                    for cite_uid in page_cite_uids
                                    if cite_uid not in known_citations
                                )
                                if new_page_cite_uids:
                                    no_progress = 0
                                else:
                                    no_progress += 1
                                LOGGER.info(
                                    "retrieval_tool_completed tool=%s call=%s citations=%s auto=true",
                                    page_content_tool.name,
                                    mcp_calls,
                                    len(page_cite_uids),
                                )
                                page_payload = {
                                    "registered_cite_uids": list(page_cite_uids),
                                    "evidence": _registered_evidence_payload(
                                        registry,
                                        page_cite_uids,
                                    ),
                                    "instruction": (
                                        "Citable page evidence is ready. Verify it actually "
                                        "addresses the query before using it. If it does, call "
                                        "finalize_retrieval with the relevant cite_uids; if it is "
                                        "off-topic or incomplete, search for the missing part."
                                    ),
                                }
                                page_rendered = _tool_result_content(
                                    page_payload,
                                    self._settings.max_tool_result_chars,
                                )
                                tool_cache[page_cache_key] = page_rendered
                                tool_content.update(page_payload)
                                tool_content["auto_page_read"] = True
                                # The deterministic bridge grabs the first plausible page, which
                                # may be off-topic. Short-circuit only when no budget remains to
                                # verify or extend it; otherwise let the model judge sufficiency.
                                # The model keeps its full tool-call allowance, but whether
                                # there is room to carry on is about work actually done, so
                                # this check counts the bridge's own call too.
                                bridge_budget_left = (
                                    mcp_calls + bridge_calls
                                    < available_mcp_calls
                                    and model_rounds
                                    < available_model_rounds - 1
                                )
                                if page_cite_uids and not bridge_budget_left:
                                    return RetrievalRun(
                                        outcome=fallback_selection(
                                            registry=registry,
                                            query=query,
                                            requirements=ledger.requirements,
                                            note=(
                                                "Citable indexed page evidence was collected by "
                                                "the deterministic retrieval bridge."
                                            ),
                                            model_rounds=model_rounds,
                                            mcp_calls=mcp_calls + bridge_calls,
                                            max_items=self._settings.max_evidence_items,
                                        ),
                                        usage=usage,
                                        l2_calls=model_rounds,
                                        diagnostics={
                                            "stop_reason": "bridge_budget_exhausted",
                                            "bridge_calls": bridge_calls,
                                            "duplicate_blocked": duplicate_blocked,
                                            "invalid_calls": invalid_calls,
                                            "no_progress": no_progress,
                                        },
                                    )
                            except AppError as exc:
                                bridge_calls += 1
                                tool_use_counts[page_content_tool.name] = (
                                    tool_use_counts.get(page_content_tool.name, 0) + 1
                                )
                                LOGGER.warning(
                                    "retrieval_tool_failed tool=%s call=%s code=%s auto=true",
                                    page_content_tool.name,
                                    mcp_calls,
                                    exc.code,
                                )
                                tool_content["auto_page_read"] = False
                                tool_content["instruction"] = (
                                    "The page read was unavailable. Select another evidence path "
                                    "or finalize with partial/no_evidence."
                                )
                except AppError as exc:
                    mcp_calls += 1
                    tool_use_counts[tool.name] = used_before + 1
                    LOGGER.warning(
                        "retrieval_tool_failed tool=%s call=%s code=%s",
                        tool.name,
                        mcp_calls,
                        exc.code,
                    )
                    tool_content = {"error": exc.code, "detail": "MCP tool call failed."}
                rendered = _tool_result_content(
                    tool_content,
                    self._settings.max_tool_result_chars,
                )
                tool_cache[cache_key] = rendered
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": rendered,
                    }
                )
                if mcp_calls + bridge_calls >= available_mcp_calls:
                    mcp_budget_exhausted = True
                    should_stop_after_turn = True

            if should_stop_after_turn:
                break

        LOGGER.info(
            "retrieval_guard rounds=%s mcp_calls=%s bridge_calls=%s duplicates=%s "
            "invalid=%s no_progress=%s",
            model_rounds, mcp_calls, bridge_calls, duplicate_blocked,
            invalid_calls, no_progress,
        )
        # These notes travel to the generation stage and models have been observed relaying
        # them verbatim to the user, so they say what the evidence amounts to rather than
        # narrating our internal limits. A reader must never hear about budgets.
        if len(registry):
            note = (
                "The evidence gathered is partial. Use what it supports, and answer the "
                "rest from established medical knowledge without citing it."
            )
        else:
            note = (
                "No citable evidence was gathered for this query. Answer from established "
                "medical knowledge without citations."
            )
        return RetrievalRun(
            outcome=fallback_selection(
                registry=registry,
                query=query,
                requirements=ledger.requirements,
                note=note,
                model_rounds=model_rounds,
                mcp_calls=mcp_calls + bridge_calls,
                max_items=self._settings.max_evidence_items,
            ),
            usage=usage,
            l2_calls=model_rounds,
            diagnostics={
                "stop_reason": "budget_or_stall",
                "bridge_calls": bridge_calls,
                "duplicate_blocked": duplicate_blocked,
                "invalid_calls": invalid_calls,
                "no_progress": no_progress,
            },
        )

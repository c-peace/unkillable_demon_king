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
    RetrievalOutcome,
    fallback_selection,
    parse_final_selection,
)
from app.evidence.routing import SourceRouter
from app.prompts import RETRIEVAL_SYSTEM_PROMPT


LOGGER = logging.getLogger("lunit_retrieval")


VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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
                "note": {"type": "string", "default": ""},
            },
            "required": ["status", "items"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class RetrievalRun:
    outcome: RetrievalOutcome
    usage: dict[str, int]
    l2_calls: int


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


def _retrieval_user_message(
    query: str,
    *,
    max_model_rounds: int,
    max_mcp_tool_calls: int,
    selected_tools: tuple[McpTool, ...],
) -> str:
    tool_list = ", ".join(tool.name for tool in selected_tools) if selected_tools else "none"
    selected_names = {tool.name for tool in selected_tools}
    index_path = ""
    if {"index_get_relevant_nodes", "index_get_page_content"}.issubset(selected_names):
        index_path = (
            "For an indexed guideline or HIRA question, call index_get_relevant_nodes once "
            "with node_id null, then call index_get_page_content with the best returned doc_id "
            "and page range. Do not repeat document discovery with reformulated queries.\n"
        )
    return (
        "Retrieve evidence for this self-contained question:\n"
        f"{query}\n\n"
        "Use the shortest authoritative path for the single critical evidence requirement. "
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

    def run(self, query: str, *, deadline: Deadline) -> RetrievalRun:
        registry = EvidenceRegistry()
        usage: dict[str, int] = {}
        if not self._settings.enable_mcp:
            return RetrievalRun(
                outcome=fallback_selection(
                    registry=registry,
                    query=query,
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
                    note="The evidence service was unavailable.",
                    model_rounds=0,
                    mcp_calls=0,
                    max_items=self._settings.max_evidence_items,
                ),
                usage=usage,
                l2_calls=0,
            )
        selected_tools = tuple(
            tool
            for tool in self._router.select(query, discovered)
            if VALID_TOOL_NAME.fullmatch(tool.name)
        )
        LOGGER.info(
            "retrieval_tools_selected count=%s tools=%s",
            len(selected_tools),
            ",".join(tool.name for tool in selected_tools),
        )
        tool_by_name = {tool.name: tool for tool in selected_tools}
        openai_tools = [tool.as_openai_tool() for tool in selected_tools]
        openai_tools.append(FINALIZE_RETRIEVAL_TOOL)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _retrieval_user_message(
                    query,
                    max_model_rounds=self._settings.max_retrieval_model_rounds,
                    max_mcp_tool_calls=self._settings.max_mcp_tool_calls,
                    selected_tools=selected_tools,
                ),
            },
        ]
        model_rounds = 0
        mcp_calls = 0
        finalizer_nudged = False
        mcp_budget_exhausted = False
        tool_cache: dict[str, str] = {}

        while model_rounds < self._settings.max_retrieval_model_rounds:
            if not deadline.can_start(0.25):
                break
            response = self._l2.complete(messages, deadline=deadline, tools=openai_tools)
            model_rounds += 1
            _sum_usage(usage, response.usage)
            messages.append(response.assistant_message)

            if not response.tool_calls:
                if finalizer_nudged:
                    break
                remaining_rounds = self._settings.max_retrieval_model_rounds - model_rounds
                remaining_calls = self._settings.max_mcp_tool_calls - mcp_calls
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
                if call.name == "finalize_retrieval":
                    try:
                        arguments = parse_tool_arguments(call.arguments)
                        outcome = parse_final_selection(
                            arguments,
                            registry=registry,
                            query=query,
                            model_rounds=model_rounds,
                            mcp_calls=mcp_calls,
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
                    )

                tool = tool_by_name.get(call.name)
                if tool is None:
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
                if mcp_calls >= self._settings.max_mcp_tool_calls:
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
                try:
                    arguments = parse_tool_arguments(call.arguments)
                except (ValueError, json.JSONDecodeError):
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
                try:
                    result = self._mcp.call_tool(
                        tool.name,
                        arguments,
                        deadline=deadline,
                    )
                    mcp_calls += 1
                    cite_uids = registry.register_payload(result, source_tool=tool.name)
                    LOGGER.info(
                        "retrieval_tool_completed tool=%s call=%s citations=%s",
                        tool.name,
                        mcp_calls,
                        len(cite_uids),
                    )
                    tool_content = {
                        "result": result,
                        "registered_cite_uids": list(cite_uids),
                    }
                except AppError as exc:
                    mcp_calls += 1
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
                if mcp_calls >= self._settings.max_mcp_tool_calls:
                    mcp_budget_exhausted = True
                    should_stop_after_turn = True

            if should_stop_after_turn:
                break

        if mcp_budget_exhausted:
            note = "Retrieval ended because the MCP tool-call budget was exhausted."
        elif model_rounds >= self._settings.max_retrieval_model_rounds:
            note = "Retrieval ended because the retrieval model-round budget was exhausted."
        else:
            note = "Retrieval ended at the configured round, tool-call, or time budget."
        return RetrievalRun(
            outcome=fallback_selection(
                registry=registry,
                query=query,
                note=note,
                model_rounds=model_rounds,
                mcp_calls=mcp_calls,
                max_items=self._settings.max_evidence_items,
            ),
            usage=usage,
            l2_calls=model_rounds,
        )

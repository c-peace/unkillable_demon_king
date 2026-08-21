from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from app.clients.http import HttpStatusError, HttpTransportError, post_json
from app.config import Settings
from app.deadline import Deadline
from app.errors import ConfigurationError, UpstreamError


class McpProtocolError(Exception):
    pass


class McpSessionExpired(McpProtocolError):
    pass


@dataclass(frozen=True, slots=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _parse_sse(body: bytes, expected_id: int | None) -> Any:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise McpProtocolError("MCP returned non-UTF-8 event data") from exc

    candidates: list[Any] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            candidates.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    if expected_id is not None:
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("id") == expected_id:
                return candidate
    if candidates:
        return candidates[-1]
    raise McpProtocolError("MCP event stream did not contain a JSON-RPC response")


class McpClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_id: str | None = None
        self._protocol_version = settings.mcp_protocol_version
        self._initialized = False
        self._request_id = 0
        self._id_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._tools_lock = threading.Lock()
        self._tools_cache: tuple[McpTool, ...] | None = None

    def list_tools(self, *, deadline: Deadline) -> tuple[McpTool, ...]:
        with self._tools_lock:
            if self._tools_cache is not None:
                return self._tools_cache
            tools: list[McpTool] = []
            cursor: str | None = None
            for _ in range(5):
                params = {"cursor": cursor} if cursor else {}
                result = self._rpc("tools/list", params, deadline=deadline)
                if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                    raise UpstreamError(
                        "MCP tools/list returned an invalid result",
                        code="invalid_mcp_response",
                    )
                for raw in result["tools"]:
                    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                        continue
                    schema = raw.get("inputSchema", raw.get("input_schema", {}))
                    if not isinstance(schema, dict):
                        schema = {"type": "object", "properties": {}}
                    tools.append(
                        McpTool(
                            name=raw["name"],
                            description=raw.get("description", "")
                            if isinstance(raw.get("description", ""), str)
                            else "",
                            input_schema=schema,
                        )
                    )
                    if len(tools) >= self._settings.max_mcp_tools:
                        break
                if len(tools) >= self._settings.max_mcp_tools:
                    break
                next_cursor = result.get("nextCursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                cursor = next_cursor
            self._tools_cache = tuple(tools)
            return self._tools_cache

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        deadline: Deadline,
    ) -> Any:
        return self._rpc(
            "tools/call",
            {"name": name, "arguments": dict(arguments)},
            deadline=deadline,
        )

    def reset(self) -> None:
        with self._init_lock:
            self._session_id = None
            self._initialized = False
            self._tools_cache = None

    def _next_id(self) -> int:
        with self._id_lock:
            self._request_id += 1
            return self._request_id

    def _ensure_initialized(self, deadline: Deadline) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            if not self._settings.enable_mcp:
                raise ConfigurationError("MCP retrieval is disabled by ENABLE_MCP.")
            if not self._settings.lunit_fm_api_key:
                raise ConfigurationError(
                    "LUNIT_FM_API_KEY must be injected at runtime before MCP can run."
                )

            versions = [self._settings.mcp_protocol_version]
            if "2024-11-05" not in versions:
                versions.append("2024-11-05")
            last_error: Exception | None = None
            for version in versions:
                request_id = self._next_id()
                try:
                    response = self._send(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": version,
                                "capabilities": {},
                                "clientInfo": {
                                    "name": "lunit-l2-clinical-orchestrator",
                                    "version": "0.1.0",
                                },
                            },
                        },
                        deadline=deadline,
                        expected_id=request_id,
                    )
                    result = self._unwrap(response)
                    if not isinstance(result, dict):
                        raise McpProtocolError("MCP initialize result was not an object")
                    negotiated = result.get("protocolVersion")
                    self._protocol_version = (
                        negotiated if isinstance(negotiated, str) and negotiated else version
                    )
                    self._send(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/initialized",
                            "params": {},
                        },
                        deadline=deadline,
                        expected_id=None,
                    )
                    self._initialized = True
                    return
                except (McpProtocolError, HttpTransportError, HttpStatusError) as exc:
                    last_error = exc
                    self._session_id = None
            raise UpstreamError(
                "MCP initialization failed",
                code="mcp_initialization_failed",
            ) from last_error

    def _rpc(self, method: str, params: Mapping[str, Any], *, deadline: Deadline) -> Any:
        for attempt in range(2):
            self._ensure_initialized(deadline)
            request_id = self._next_id()
            try:
                response = self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": dict(params),
                    },
                    deadline=deadline,
                    expected_id=request_id,
                )
                return self._unwrap(response)
            except McpSessionExpired:
                if attempt:
                    break
                self.reset()
            except (McpProtocolError, HttpTransportError, HttpStatusError) as exc:
                raise UpstreamError(
                    f"MCP {method} failed",
                    code="mcp_request_failed",
                ) from exc
        raise UpstreamError("MCP session could not be recovered", code="mcp_session_failed")

    def _send(
        self,
        payload: Mapping[str, Any],
        *,
        deadline: Deadline,
        expected_id: int | None,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self._settings.lunit_fm_api_key}",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self._protocol_version,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = post_json(
                self._settings.lunit_mcp_url,
                payload,
                headers=headers,
                timeout=deadline.remaining(self._settings.mcp_timeout_sec),
                max_response_bytes=self._settings.max_upstream_response_bytes,
            )
        except HttpStatusError as exc:
            if exc.status == 404 and self._session_id:
                raise McpSessionExpired("MCP session expired") from exc
            raise

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        if not response.body:
            if expected_id is None:
                return None
            raise McpProtocolError("MCP returned an empty JSON-RPC response")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return _parse_sse(response.body, expected_id)
        return response.json()

    @staticmethod
    def _unwrap(response: Any) -> Any:
        if not isinstance(response, dict):
            raise McpProtocolError("MCP response was not a JSON-RPC object")
        if "error" in response:
            error = response["error"]
            code = error.get("code") if isinstance(error, dict) else "unknown"
            raise McpProtocolError(f"MCP JSON-RPC error {code}")
        if "result" not in response:
            raise McpProtocolError("MCP JSON-RPC response had no result")
        return response["result"]

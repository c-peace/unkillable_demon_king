from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.clients.http import HttpStatusError, HttpTimeoutError, HttpTransportError, post_json
from app.config import Settings
from app.deadline import Deadline
from app.errors import ConfigurationError, UpstreamError


LOGGER = logging.getLogger("lunit_l2")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class L2Response:
    content: str
    tool_calls: tuple[ToolCall, ...]
    assistant_message: dict[str, Any]
    usage: dict[str, int]


_TEXT_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_-]{0,127})\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
_TEXT_TOOL_ARGUMENT = re.compile(
    r"<arg_key>\s*([A-Za-z_][A-Za-z0-9_-]{0,127})\s*</arg_key>\s*"
    r"<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _parse_text_encoded_tool_calls(content: str) -> tuple[ToolCall, ...]:
    """Normalize L2's legacy XML-like tool syntax only when it is the whole message."""
    text = content.strip()
    if not text:
        return ()

    blocks = list(_TEXT_TOOL_CALL_BLOCK.finditer(text))
    if not blocks:
        return ()

    cursor = 0
    parsed: list[ToolCall] = []
    for index, block in enumerate(blocks):
        if text[cursor : block.start()].strip():
            return ()
        cursor = block.end()

        body = block.group(2)
        arguments: dict[str, str] = {}
        argument_cursor = 0
        for argument in _TEXT_TOOL_ARGUMENT.finditer(body):
            if body[argument_cursor : argument.start()].strip():
                return ()
            key = argument.group(1)
            if key in arguments:
                return ()
            arguments[key] = argument.group(2).strip()
            argument_cursor = argument.end()
        if body[argument_cursor:].strip():
            return ()

        parsed.append(
            ToolCall(
                id=f"text_tool_call_{index}",
                name=block.group(1),
                arguments=arguments,
            )
        )

    if text[cursor:].strip():
        return ()
    return tuple(parsed)


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("tool arguments must be a JSON object or JSON string")
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments JSON must decode to an object")
    return parsed


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and item >= 0:
            normalized[key] = item
    return normalized


def _parse_response(payload: Any) -> L2Response:
    if not isinstance(payload, dict):
        raise UpstreamError("L2 returned a non-object response", code="invalid_l2_response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise UpstreamError("L2 response did not contain a choice", code="invalid_l2_response")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise UpstreamError("L2 response did not contain a message", code="invalid_l2_response")

    content_value = message.get("content")
    content = content_value.strip() if isinstance(content_value, str) else ""
    parsed_calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            call_id = raw.get("id")
            parsed_calls.append(
                ToolCall(
                    id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                    name=function["name"],
                    arguments=function.get("arguments", "{}"),
                )
            )
    elif isinstance(message.get("function_call"), dict):
        legacy = message["function_call"]
        if isinstance(legacy.get("name"), str):
            parsed_calls.append(
                ToolCall(
                    id="legacy_function_call",
                    name=legacy["name"],
                    arguments=legacy.get("arguments", "{}"),
                )
            )

    if not parsed_calls and content:
        text_calls = _parse_text_encoded_tool_calls(content)
        if text_calls:
            parsed_calls.extend(text_calls)
            content = ""

    assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if parsed_calls:
        assistant_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in parsed_calls
        ]
    return L2Response(
        content=content,
        tool_calls=tuple(parsed_calls),
        assistant_message=assistant_message,
        usage=_normalize_usage(payload.get("usage")),
    )


class L2Client:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        deadline: Deadline,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> L2Response:
        key = self._settings.lunit_fm_api_key
        if not key:
            raise ConfigurationError(
                "LUNIT_FM_API_KEY must be injected at runtime before chat completions can run."
            )

        # Sampling was previously left to the server default, which is what produced the
        # word-salad and code-token tails seen on longer answers. The organizer's own
        # reference client pins greedy decoding, so match it.
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": self._settings.l2_max_tokens,
            "temperature": self._settings.l2_temperature,
            "top_p": self._settings.l2_top_p,
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
            payload["tool_choice"] = tool_choice or "auto"
            payload["parallel_tool_calls"] = False

        # An empty completion is a real failure mode: on some conversations — vaccine
        # hesitancy in Spanish is one we hit — L2 returns no content at all, and the
        # evaluator scores the turn zero. Re-sending the identical request cannot help,
        # because greedy decoding makes it deterministic, so each retry has to change
        # something. Sealing this off is the harness's job: the model still writes the
        # answer, we only change what we ask it with.
        empty_attempts = self._settings.empty_output_retries + 1
        for empty_attempt in range(empty_attempts):
            attempt_payload = self._escalate_on_empty(payload, empty_attempt)
            response = self._post_with_retry(attempt_payload, deadline=deadline)
            if response.content or response.tool_calls:
                return response
            if empty_attempt + 1 < empty_attempts and deadline.can_start(0.25):
                time.sleep(min(0.1, deadline.remaining(0.1)))
        raise UpstreamError("L2 returned an empty message", code="empty_l2_response")

    def _escalate_on_empty(self, payload: dict[str, Any], attempt: int) -> dict[str, Any]:
        """Shorten the conversation after an empty completion.

        Measured against the live model: on some long multi-turn conversations L2 returns
        no content at all, and it does so deterministically — re-sending, raising the
        temperature to 1.0, and adding a system instruction all reproduce the empty
        answer exactly. Truncating the history is the one lever that recovers it, and it
        recovers a full-quality answer. So the ladder trims turns rather than fiddling
        with decoding, keeping the leading system message so the answer stays ours.
        """
        if attempt == 0:
            return payload
        keep = 6 if attempt == 1 else 2
        messages = [dict(message) for message in payload.get("messages", [])]
        lead = messages[:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(lead):]
        if len(body) <= keep:
            return payload
        escalated = dict(payload)
        escalated["messages"] = lead + body[-keep:]
        return escalated

    def _post_with_retry(self, payload: dict[str, Any], *, deadline: Deadline) -> L2Response:
        attempts = self._settings.l2_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            try:
                response = post_json(
                    f"{self._settings.lunit_fm_api_url}/v1/chat/completions",
                    payload,
                    headers={
                        "Authorization": f"Bearer {self._settings.lunit_fm_api_key}",
                        "Accept": "application/json",
                    },
                    timeout=deadline.remaining(self._settings.l2_timeout_sec),
                    max_response_bytes=self._settings.max_upstream_response_bytes,
                )
                parsed = _parse_response(response.json())
                LOGGER.info(
                    "l2_call_completed attempt=%s latency_ms=%s tool_calls=%s content_chars=%s",
                    attempt + 1,
                    round((time.monotonic() - attempt_started) * 1000),
                    len(parsed.tool_calls),
                    len(parsed.content),
                )
                return parsed
            except HttpTimeoutError as exc:
                LOGGER.warning(
                    "l2_call_timeout attempt=%s latency_ms=%s",
                    attempt + 1,
                    round((time.monotonic() - attempt_started) * 1000),
                )
                raise UpstreamError(
                    "L2 request exceeded its per-call timeout",
                    code="l2_timeout",
                ) from exc
            except HttpStatusError as exc:
                last_error = exc
                if exc.status not in {408, 409, 425, 429} and exc.status < 500:
                    raise UpstreamError(
                        f"L2 rejected the request with HTTP {exc.status}",
                        code="l2_request_rejected",
                    ) from exc
            except HttpTransportError as exc:
                last_error = exc

            LOGGER.warning(
                "l2_call_retryable_failure attempt=%s max_attempts=%s latency_ms=%s",
                attempt + 1,
                attempts,
                round((time.monotonic() - attempt_started) * 1000),
            )

            if attempt + 1 < attempts and deadline.can_start(0.5):
                delay = min(0.25 * math.pow(2, attempt), 1.0, deadline.remaining(1.0))
                time.sleep(delay)
        raise UpstreamError("L2 request failed after bounded retries", code="l2_unavailable") from last_error

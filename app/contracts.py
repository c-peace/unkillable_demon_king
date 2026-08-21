from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.errors import AppError


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    model: str
    messages: tuple[dict[str, Any], ...]
    stream: bool = False


def _invalid(message: str, *, param: str | None = None) -> AppError:
    return AppError(
        message=message,
        status_code=400,
        code="invalid_request",
        error_type="invalid_request_error",
        param=param,
    )


def _content_to_text(value: Any, *, param: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise _invalid("message content must be a string or text-part array", param=param)

    parts: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
            raise _invalid(
                "only textual message content is supported",
                param=f"{param}[{index}]",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise _invalid("text content must be a string", param=f"{param}[{index}].text")
        parts.append(text)
    return "\n".join(parts)


def _normalize_message(value: Any, index: int) -> dict[str, Any]:
    param = f"messages[{index}]"
    if not isinstance(value, dict):
        raise _invalid("each message must be an object", param=param)

    role = value.get("role")
    if role not in ALLOWED_ROLES:
        raise _invalid(
            f"message role must be one of {sorted(ALLOWED_ROLES)}",
            param=f"{param}.role",
        )

    normalized: dict[str, Any] = {
        "role": role,
        "content": _content_to_text(value.get("content"), param=f"{param}.content"),
    }
    if role == "assistant" and "tool_calls" in value:
        if not isinstance(value["tool_calls"], list):
            raise _invalid("tool_calls must be an array", param=f"{param}.tool_calls")
        normalized["tool_calls"] = value["tool_calls"]
    if role == "tool":
        tool_call_id = value.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise _invalid("tool messages require tool_call_id", param=f"{param}.tool_call_id")
        normalized["tool_call_id"] = tool_call_id
    if isinstance(value.get("name"), str):
        normalized["name"] = value["name"]

    if normalized["content"] is None and "tool_calls" not in normalized:
        raise _invalid("message content cannot be null", param=f"{param}.content")
    return normalized


def parse_chat_completion_request(
    payload: Any,
    *,
    default_model: str,
) -> ChatCompletionRequest:
    if not isinstance(payload, dict):
        raise _invalid("request body must be a JSON object")

    model = payload.get("model", default_model)
    if not isinstance(model, str) or not model.strip():
        raise _invalid("model must be a non-empty string", param="model")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages must be a non-empty array", param="messages")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise _invalid("stream must be a boolean", param="stream")
    if stream:
        raise AppError(
            message="streaming is not supported by this driver",
            status_code=400,
            code="unsupported_streaming",
            error_type="invalid_request_error",
            param="stream",
        )

    return ChatCompletionRequest(
        model=model.strip(),
        messages=tuple(_normalize_message(message, i) for i, message in enumerate(messages)),
        stream=False,
    )


def model_list_payload(model: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "lunit",
            }
        ],
    }


def chat_completion_payload(
    *,
    content: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        payload["usage"] = usage
    return payload

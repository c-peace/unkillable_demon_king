from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from app.clients.l2 import L2Response


class ScriptedL2:
    def __init__(self, responses: Sequence[L2Response]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, deadline, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": list(tools) if tools else [],
                "tool_choice": tool_choice,
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedL2 received an unexpected call")
        return self.responses.popleft()


def l2_content(content: str, usage: Mapping[str, int] | None = None) -> L2Response:
    return L2Response(
        content=content,
        tool_calls=(),
        assistant_message={"role": "assistant", "content": content},
        usage=dict(usage or {}),
    )


def l2_tool_call(call_id: str, name: str, arguments: Any) -> L2Response:
    from app.clients.l2 import ToolCall

    argument_text = (
        arguments if isinstance(arguments, str) else __import__("json").dumps(arguments)
    )
    raw_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": argument_text},
    }
    return L2Response(
        content="",
        tool_calls=(ToolCall(call_id, name, arguments),),
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [raw_call],
        },
        usage={},
    )


class NeverRetrieval:
    def run(self, query: str, *, deadline, request_id: str = ""):
        raise AssertionError(f"retrieval should not run: {query}")

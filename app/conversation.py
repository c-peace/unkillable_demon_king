from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


HIGH_RISK_PATTERN = re.compile(
    r"(용량|복용량|과다복용|상호작용|금기|임신|수유|소아|영아|신생아|"
    r"흉통|호흡곤란|의식.{0,4}(저하|없)|자살|자해|응급|출혈|아나필락시스|"
    r"dose|dosage|overdose|interaction|contraindication|pregnan|breastfeed|"
    r"pediatric|infant|newborn|chest pain|shortness of breath|suicid|self-harm|"
    r"emergency|anaphylaxis|severe bleeding)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CompiledConversation:
    messages: tuple[dict[str, Any], ...]
    latest_user_text: str
    history_hash: str
    case_packet: str
    is_high_risk: bool

    def generation_messages(self, representation: str) -> list[dict[str, Any]]:
        if representation == "native":
            return [dict(message) for message in self.messages]
        return [
            {
                "role": "user",
                "content": (
                    "The following is the complete role-labelled conversation. "
                    "Answer the final user turn while preserving all prior facts and corrections.\n\n"
                    f"{self.case_packet}"
                ),
            }
        ]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def compile_conversation(messages: Iterable[dict[str, Any]]) -> CompiledConversation:
    copied = tuple(dict(message) for message in messages)
    latest_user = next(
        (_message_text(message) for message in reversed(copied) if message.get("role") == "user"),
        "",
    )
    canonical = json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    history_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    turns = []
    for index, message in enumerate(copied, start=1):
        role = str(message.get("role", "unknown")).upper()
        turns.append(f"[TURN {index} | {role}]\n{_message_text(message)}")
    packet = "\n\n".join(turns)
    return CompiledConversation(
        messages=copied,
        latest_user_text=latest_user,
        history_hash=history_hash,
        case_packet=packet,
        is_high_risk=bool(HIGH_RISK_PATTERN.search(latest_user)),
    )

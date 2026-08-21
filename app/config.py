from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUBMISSION_ENV_FILE = Path("/app/submission.env")


def _read_submission_key(path: Path | None = None) -> str | None:
    """Read the bundled team key without overriding a runtime environment value."""
    path = SUBMISSION_ENV_FILE if path is None else path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "LUNIT_FM_API_KEY":
            parsed = value.strip()
            if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
                parsed = parsed[1:-1]
            return parsed or None
    return None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _as_int(
    value: str | None,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = default if value is None or not value.strip() else int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"integer must be between {minimum} and {maximum}")
    return parsed


def _as_float(
    value: str | None,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    parsed = default if value is None or not value.strip() else float(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"number must be between {minimum} and {maximum}")
    return parsed


def _as_optional_float(
    value: str | None,
    default: float | None,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None or not value.strip():
        return default
    parsed = float(value)
    if parsed <= 0:
        return None
    if not minimum <= parsed <= maximum:
        raise ValueError(f"number must be between {minimum} and {maximum}, or 0 to disable")
    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    model: str = "Lunit/L2-preview"
    lunit_fm_api_url: str = "https://model.hackathon.lunit.io"
    lunit_fm_api_key: str | None = None
    lunit_mcp_url: str = "https://mcp.hackathon.lunit.io/mcp"
    request_timeout_sec: float | None = None
    retrieval_timeout_sec: float | None = 30.0
    l2_timeout_sec: float = 60.0
    mcp_timeout_sec: float = 60.0
    l2_retries: int = 0
    empty_output_retries: int = 0
    l2_max_tokens: int = 4_096
    max_generation_retrievals: int = 1
    max_retrieval_model_rounds: int = 2
    max_mcp_tool_calls: int = 3
    max_mcp_tools: int = 64
    max_request_bytes: int = 1_048_576
    max_upstream_response_bytes: int = 4_194_304
    max_tool_result_chars: int = 8_000
    max_retrieval_query_chars: int = 4_000
    max_evidence_items: int = 4
    max_evidence_chars: int = 8_000
    mcp_protocol_version: str = "2025-03-26"
    mcp_tool_mode: str = "family"
    enable_mcp: bool = True
    enable_high_risk_review: bool = False
    conversation_representation: str = "native"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        tool_mode = (source.get("MCP_TOOL_MODE") or "family").strip().lower()
        if tool_mode not in {"all", "family"}:
            raise ValueError("MCP_TOOL_MODE must be 'all' or 'family'")
        representation = (
            source.get("CONVERSATION_REPRESENTATION") or "native"
        ).strip().lower()
        if representation not in {"native", "case_packet"}:
            raise ValueError(
                "CONVERSATION_REPRESENTATION must be 'native' or 'case_packet'"
            )

        key = source.get("LUNIT_FM_API_KEY")
        if env is None and (not key or not key.strip()):
            key = _read_submission_key()
        return cls(
            host=(source.get("HOST") or "0.0.0.0").strip(),
            port=_as_int(source.get("PORT"), 8000, minimum=1, maximum=65535),
            model=(source.get("LUNIT_FM_MODEL") or "Lunit/L2-preview").strip(),
            lunit_fm_api_url=(
                source.get("LUNIT_FM_API_URL") or "https://model.hackathon.lunit.io"
            ).rstrip("/"),
            lunit_fm_api_key=key.strip() if key and key.strip() else None,
            lunit_mcp_url=(
                source.get("LUNIT_MCP_URL") or "https://mcp.hackathon.lunit.io/mcp"
            ),
            request_timeout_sec=_as_optional_float(
                source.get("REQUEST_TIMEOUT_SEC"), None, minimum=1.0, maximum=600.0
            ),
            retrieval_timeout_sec=_as_optional_float(
                source.get("RETRIEVAL_TIMEOUT_SEC"),
                30.0,
                minimum=1.0,
                maximum=300.0,
            ),
            l2_timeout_sec=_as_float(
                source.get("L2_TIMEOUT_SEC"), 60.0, minimum=1.0, maximum=300.0
            ),
            mcp_timeout_sec=_as_float(
                source.get("MCP_TIMEOUT_SEC"), 60.0, minimum=1.0, maximum=300.0
            ),
            l2_retries=_as_int(
                source.get("MAX_L2_RETRIES"), 0, minimum=0, maximum=5
            ),
            empty_output_retries=_as_int(
                source.get("EMPTY_OUTPUT_RETRIES"), 0, minimum=0, maximum=3
            ),
            l2_max_tokens=_as_int(
                source.get("L2_MAX_TOKENS"),
                4_096,
                minimum=256,
                maximum=32_768,
            ),
            max_generation_retrievals=_as_int(
                source.get("MAX_GENERATION_RETRIEVALS"), 1, minimum=0, maximum=4
            ),
            max_retrieval_model_rounds=_as_int(
                source.get("MAX_RETRIEVAL_MODEL_ROUNDS"), 2, minimum=1, maximum=12
            ),
            max_mcp_tool_calls=_as_int(
                source.get("MAX_MCP_TOOL_CALLS"), 3, minimum=0, maximum=24
            ),
            max_mcp_tools=_as_int(
                source.get("MAX_MCP_TOOLS"), 64, minimum=1, maximum=256
            ),
            max_request_bytes=_as_int(
                source.get("MAX_REQUEST_BYTES"),
                1_048_576,
                minimum=1_024,
                maximum=16_777_216,
            ),
            max_upstream_response_bytes=_as_int(
                source.get("MAX_UPSTREAM_RESPONSE_BYTES"),
                4_194_304,
                minimum=65_536,
                maximum=33_554_432,
            ),
            max_tool_result_chars=_as_int(
                source.get("MAX_TOOL_RESULT_CHARS"),
                8_000,
                minimum=1_000,
                maximum=100_000,
            ),
            max_retrieval_query_chars=_as_int(
                source.get("MAX_RETRIEVAL_QUERY_CHARS"),
                4_000,
                minimum=256,
                maximum=20_000,
            ),
            max_evidence_items=_as_int(
                source.get("MAX_EVIDENCE_ITEMS"), 4, minimum=1, maximum=32
            ),
            max_evidence_chars=_as_int(
                source.get("MAX_EVIDENCE_CHARS"),
                8_000,
                minimum=1_000,
                maximum=100_000,
            ),
            mcp_protocol_version=(
                source.get("MCP_PROTOCOL_VERSION") or "2025-03-26"
            ),
            mcp_tool_mode=tool_mode,
            enable_mcp=_as_bool(source.get("ENABLE_MCP"), True),
            enable_high_risk_review=_as_bool(
                source.get("ENABLE_HIGH_RISK_REVIEW"), False
            ),
            conversation_representation=representation,
        )

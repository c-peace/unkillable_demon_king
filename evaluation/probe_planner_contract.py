from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from app.clients.l2 import L2Client
from app.config import Settings, _read_submission_key
from app.conversation import compile_conversation
from app.deadline import Deadline
from app.orchestration.planning import StructuredPlanner


CASES = {
    "direct": [
        {
            "role": "user",
            "content": "Explain what systolic blood pressure means in one sentence.",
        }
    ],
    "clarify": [
        {"role": "user", "content": "Is it safe for me to take it?"},
    ],
    "grounded": [
        {
            "role": "user",
            "content": (
                "What do current guidelines recommend as the blood pressure target "
                "for adults with chronic kidney disease?"
            ),
        }
    ],
    "high_risk_grounded": [
        {
            "role": "user",
            "content": "I am 10 weeks pregnant and was told to avoid ibuprofen.",
        },
        {
            "role": "assistant",
            "content": "Understood. I will account for the pregnancy and restriction.",
        },
        {
            "role": "user",
            "content": "What do current guidelines recommend for treating this headache?",
        },
    ],
}


def _probe(planner: StructuredPlanner, *, name: str) -> dict[str, object]:
    compiled = compile_conversation(CASES[name])
    result = planner.plan(compiled, deadline=Deadline.after(120))
    summary: dict[str, object] = {
        "case": name,
        "mode": result.mode,
        "error_code": result.error_code,
        "l2_calls": result.l2_calls,
        "valid": result.plan is not None,
    }
    if result.plan is not None:
        summary.update(
            {
                "response_mode": result.plan.response_mode,
                "risk_level": result.plan.risk_level,
                "derived_lane": result.plan.lane,
                "fact_count": len(result.plan.clinical_facts),
                "risk_count": len(result.plan.risk_signals),
                "requirement_count": len(result.plan.evidence_requirements),
            }
        )
    return summary


def main() -> int:
    key = _read_submission_key(Path(".env"))
    if not key:
        raise RuntimeError("LUNIT_FM_API_KEY is missing from the local submission env")
    settings = replace(
        Settings.from_env({"LUNIT_FM_API_KEY": key}),
        l2_retries=0,
        empty_output_retries=0,
    )
    planner = StructuredPlanner(
        l2=L2Client(settings),
        retry_reserve_sec=settings.l2_timeout_sec,
    )
    probes = [_probe(planner, name=name) for name in CASES]
    print(json.dumps({"probes": probes}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(probe["valid"] for probe in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

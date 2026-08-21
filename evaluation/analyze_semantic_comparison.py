from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY_METRIC = "HealthBench Rubric"


def _metric_scores(result: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in result.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        name = metric.get("name")
        score = metric.get("score")
        if isinstance(name, str) and isinstance(score, (int, float)):
            scores[name] = float(score)
    return scores


def _load_results(run_dir: Path, sample_limit: int) -> dict[int, dict[str, Any]]:
    path = run_dir / "coeval" / "results_conquer_val.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a result array")
    selected: dict[int, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), int):
            continue
        sample_id = item["sample_id"]
        if sample_id < sample_limit:
            selected[sample_id] = {
                "failed": bool(item.get("inference_failed") or item.get("scoring_failed")),
                "passed": bool(item.get("passed")),
                "generation_time_ms": item.get("generation_time_ms"),
                "scores": _metric_scores(item),
            }
    return selected


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (run_dir / "service.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("event"), str):
            events.append(event)
    return events


def _counter(completed: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key, "missing")) for item in completed).items()))


def _sum_int(completed: list[dict[str, Any]], key: str) -> int:
    return sum(item.get(key, 0) for item in completed if isinstance(item.get(key, 0), int))


def _operational_summary(run_dir: Path) -> dict[str, Any]:
    events = _load_events(run_dir)
    completed = [item for item in events if item["event"] == "request_completed"]
    event_counts = Counter(item["event"] for item in events)
    total_tokens = 0
    for item in completed:
        usage = item.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            total_tokens += usage["total_tokens"]
    return {
        "completed_requests": len(completed),
        "planned_lanes": _counter(completed, "planned_lane"),
        "planner_modes": _counter(completed, "planner_mode"),
        "fallback_reasons": _counter(completed, "fallback_reason"),
        "ledger_modes": _counter(completed, "ledger_mode"),
        "retrieval_statuses": _counter(completed, "retrieval_status"),
        "review_statuses": _counter(completed, "review_status"),
        "l2_calls": _sum_int(completed, "l2_calls"),
        "retrievals": _sum_int(completed, "retrievals"),
        "mcp_calls": _sum_int(completed, "mcp_calls"),
        "requirements": _sum_int(completed, "requirement_count"),
        "total_tokens": total_tokens,
        "retrieval_finalized_events": event_counts["retrieval_finalized"],
        "retrieval_budget_stopped_events": event_counts["retrieval_budget_stopped"],
        "review_started_events": event_counts["review_started"],
        "review_completed_events": event_counts["review_completed"],
        "revision_started_events": event_counts["revision_started"],
        "revision_completed_events": event_counts["revision_completed"],
    }


def _bootstrap_interval(differences: list[float], iterations: int = 10_000) -> list[float] | None:
    if not differences:
        return None
    generator = random.Random(20260822)
    estimates = sorted(
        mean(generator.choice(differences) for _ in differences)
        for _ in range(iterations)
    )
    return [
        estimates[int(iterations * 0.025)],
        estimates[int(iterations * 0.975) - 1],
    ]


def analyze(runs: dict[str, Path], sample_limit: int) -> dict[str, Any]:
    results = {
        mode: _load_results(run_dir, sample_limit) for mode, run_dir in runs.items()
    }
    common_ids = sorted(
        set.intersection(
            *(
                {
                    sample_id
                    for sample_id, item in mode_results.items()
                    if not item["failed"] and PRIMARY_METRIC in item["scores"]
                }
                for mode_results in results.values()
            )
        )
    )
    modes: dict[str, Any] = {}
    legacy_scores = {
        sample_id: results["legacy"][sample_id]["scores"][PRIMARY_METRIC]
        for sample_id in common_ids
    }
    for mode, mode_results in results.items():
        available = [item for item in mode_results.values() if not item["failed"]]
        primary = [
            item["scores"][PRIMARY_METRIC]
            for item in available
            if PRIMARY_METRIC in item["scores"]
        ]
        axis_names = sorted(
            {
                name
                for item in available
                for name in item["scores"]
                if name.startswith("axis:")
            }
        )
        common_scores = {
            sample_id: mode_results[sample_id]["scores"][PRIMARY_METRIC]
            for sample_id in common_ids
        }
        differences = [
            common_scores[sample_id] - legacy_scores[sample_id]
            for sample_id in common_ids
        ]
        modes[mode] = {
            "selected_samples": len(mode_results),
            "successful_samples": len(available),
            "failed_samples": sum(item["failed"] for item in mode_results.values()),
            "raw_success_only_score": mean(primary) if primary else None,
            "raw_failure_penalized_score": sum(primary) / sample_limit,
            "common_sample_score": mean(common_scores.values()) if common_scores else None,
            "common_delta_vs_legacy": mean(differences) if differences else None,
            "common_bootstrap_95": _bootstrap_interval(differences),
            "common_improved": sum(value > 1e-12 for value in differences),
            "common_worsened": sum(value < -1e-12 for value in differences),
            "common_tied": sum(abs(value) <= 1e-12 for value in differences),
            "axis_scores": {
                name.removeprefix("axis:"): mean(
                    item["scores"][name] for item in available if name in item["scores"]
                )
                for name in axis_names
            },
            "operations": _operational_summary(runs[mode]),
        }
    return {
        "sample_limit": sample_limit,
        "common_successful_samples": len(common_ids),
        "modes": modes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate semantic-mode runs without emitting prompts, answers, or rubrics"
    )
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    report = analyze(
        {
            "legacy": args.legacy,
            "planning": args.planning,
            "ledger": args.ledger,
            "review": args.review,
        },
        args.sample_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

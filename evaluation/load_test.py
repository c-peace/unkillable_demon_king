from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

from evaluation.coeval_lab import (
    PROJECT_ROOT,
    RUNS_ROOT,
    _run,
    _wait_until_ready,
    git_head,
    git_status_paths,
    summarize_service_log,
)

SYNTHETIC_WORKLOAD: tuple[tuple[dict[str, str], ...], ...] = (
    ({"role": "user", "content": "의학 연구에서 민감도와 특이도의 차이를 쉽게 설명해 주세요."},),
    ({"role": "user", "content": "상대위험도와 절대위험도의 차이를 간단한 예로 설명해 주세요."},),
    ({"role": "user", "content": "약 복용 순응도가 중요한 이유를 세 가지로 정리해 주세요."},),
    ({"role": "user", "content": "피부 발진을 상담할 때 전달하면 좋은 정보를 목록으로 알려 주세요."},),
    (
        {"role": "user", "content": "현재 복용량은 10mg입니다."},
        {"role": "assistant", "content": "10mg으로 이해했습니다."},
        {"role": "user", "content": "정정합니다. 실제 복용량은 5mg입니다. 어떤 정보를 기록해야 하나요?"},
    ),
    (
        {"role": "user", "content": "검사 결과 설명을 준비하고 있습니다."},
        {"role": "assistant", "content": "어떤 형식이 필요하신가요?"},
        {"role": "user", "content": "환자가 이해하기 쉬운 세 문장 형식으로 작성할 때 원칙을 알려 주세요."},
    ),
    ({"role": "user", "content": "임신 중 일반의약품 상담 전에 확인해야 할 정보를 알려 주세요."},),
    ({"role": "user", "content": "약물 상호작용 상담에서 반드시 확인해야 할 정보를 알려 주세요."},),
    ({"role": "user", "content": "흉통 상담에서 응급 평가가 필요한 위험 신호를 일반적으로 설명해 주세요."},),
    ({"role": "user", "content": "소아 발열 상담에서 보호자에게 확인할 핵심 정보를 알려 주세요."},),
    ({"role": "user", "content": "의료진에게 증상 발생 시점을 정확히 전달하는 방법을 알려 주세요."},),
    ({"role": "user", "content": "건강 정보를 읽을 때 근거 수준과 불확실성을 구분하는 방법을 설명해 주세요."},),
)


@dataclass(frozen=True, slots=True)
class LoadRecord:
    index: int
    ok: bool
    status: int | None
    latency_ms: float
    error_type: str | None = None


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * percentile) - 1)]


def summarize_phase(
    records: Sequence[LoadRecord],
    *,
    concurrency: int,
    wall_time_s: float,
) -> dict[str, Any]:
    latencies = [record.latency_ms for record in records]
    successes = sum(record.ok for record in records)
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "success_rate": successes / len(records) if records else 0.0,
        "wall_time_s": wall_time_s,
        "throughput_rps": len(records) / wall_time_s if wall_time_s > 0 else 0.0,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "status_counts": dict(Counter(str(record.status) for record in records)),
        "error_counts": dict(
            Counter(record.error_type for record in records if record.error_type)
        ),
    }


def choose_concurrency(
    concurrency_4: dict[str, Any],
    concurrency_8: dict[str, Any],
    *,
    minimum_speedup: float = 1.30,
) -> dict[str, Any]:
    wall_4 = float(concurrency_4["wall_time_s"])
    wall_8 = float(concurrency_8["wall_time_s"])
    speedup = wall_4 / wall_8 if wall_8 > 0 else 0.0
    stable = (
        concurrency_4["failures"] == 0
        and concurrency_8["failures"] == 0
        and concurrency_8["success_rate"] == 1.0
    )
    accepted = stable and speedup >= minimum_speedup
    return {
        "recommended_concurrency": 8 if accepted else 4,
        "concurrency_8_accepted": accepted,
        "stable": stable,
        "speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "reason": (
            "stable_and_faster"
            if accepted
            else "request_failures"
            if not stable
            else "insufficient_speedup"
        ),
    }


def _send_request(
    *,
    base_url: str,
    model: str,
    messages: Sequence[dict[str, str]],
    index: int,
    timeout_s: float,
) -> LoadRecord:
    body = json.dumps(
        {"model": model, "messages": list(messages), "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": f"load-{index}",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            payload = json.load(response)
        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        content = (
            choices[0].get("message", {}).get("content")
            if choices and isinstance(choices[0], dict)
            else None
        )
        ok = status == 200 and isinstance(content, str) and bool(content.strip())
        return LoadRecord(
            index=index,
            ok=ok,
            status=status,
            latency_ms=(time.monotonic() - started) * 1000,
            error_type=None if ok else "invalid_response",
        )
    except urllib.error.HTTPError as exc:
        return LoadRecord(
            index=index,
            ok=False,
            status=exc.code,
            latency_ms=(time.monotonic() - started) * 1000,
            error_type="http_error",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return LoadRecord(
            index=index,
            ok=False,
            status=None,
            latency_ms=(time.monotonic() - started) * 1000,
            error_type=type(exc).__name__,
        )


def _run_phase(
    *,
    base_url: str,
    model: str,
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        records = list(
            executor.map(
                lambda item: _send_request(
                    base_url=base_url,
                    model=model,
                    messages=item[1],
                    index=concurrency * 100 + item[0],
                    timeout_s=timeout_s,
                ),
                enumerate(SYNTHETIC_WORKLOAD),
            )
        )
    return summarize_phase(
        records,
        concurrency=concurrency,
        wall_time_s=time.monotonic() - started,
    )


def run_load_test(args: argparse.Namespace) -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-loadtest"
    run_dir = Path(args.runs_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    image = f"lunit-harness:loadtest-{run_id.lower()}"
    container = f"lunit-loadtest-{run_id.lower()}"
    base_url = f"http://127.0.0.1:{args.port}"
    project_changes = git_status_paths(PROJECT_ROOT)
    workload_digest = hashlib.sha256(
        json.dumps(SYNTHETIC_WORKLOAD, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": "candidate_concurrency_loadtest",
        "concurrencies": [4, 8],
        "workload_count": len(SYNTHETIC_WORKLOAD),
        "workload_digest": workload_digest,
        "workload_policy": "team-authored synthetic conversations; content not persisted",
        "project_commit": git_head(PROJECT_ROOT),
        "project_dirty": bool(project_changes),
        "project_changed_paths": project_changes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    container_started = False
    report: dict[str, Any] = {}
    try:
        _run(["docker", "build", "-q", "-t", image, "."], cwd=PROJECT_ROOT)
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "-p",
                f"127.0.0.1:{args.port}:8000",
                "--name",
                container,
                image,
            ],
            cwd=PROJECT_ROOT,
        )
        container_started = True
        _wait_until_ready(args.port, args.model)
        warmup = _send_request(
            base_url=base_url,
            model=args.model,
            messages=SYNTHETIC_WORKLOAD[0],
            index=0,
            timeout_s=args.request_timeout,
        )
        if not warmup.ok:
            raise RuntimeError(f"warmup failed: {warmup.error_type}")
        phase_4 = _run_phase(
            base_url=base_url,
            model=args.model,
            concurrency=4,
            timeout_s=args.request_timeout,
        )
        phase_8 = _run_phase(
            base_url=base_url,
            model=args.model,
            concurrency=8,
            timeout_s=args.request_timeout,
        )
        report = {
            "status": "complete",
            "phases": {"4": phase_4, "8": phase_8},
            "decision": choose_concurrency(phase_4, phase_8),
            "artifact_dir": str(run_dir),
        }
    finally:
        if container_started:
            logs = _run(
                ["docker", "logs", container], cwd=PROJECT_ROOT, check=False
            )
            service_log = run_dir / "service.log"
            service_log.write_text(
                (logs.stdout or "") + (logs.stderr or ""), encoding="utf-8"
            )
            report["service"] = summarize_service_log(service_log)
            _run(
                ["docker", "stop", "-t", "10", container],
                cwd=PROJECT_ROOT,
                check=False,
            )

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-safe live candidate load test")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--model", default="Lunit/L2-preview")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_load_test(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"load_test_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

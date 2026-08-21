from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COEVAL_ROOT = PROJECT_ROOT.parent / "CoEval"
RUNS_ROOT = PROJECT_ROOT / "evaluation" / "runs"
JUDGE_GROUP = "datasets/metrics/judge@conquer_judge=gpt-4.1"
SEMANTIC_MODES: dict[str, dict[str, str]] = {
    "legacy": {
        "ENABLE_STRUCTURED_PLANNING": "false",
        "ENABLE_REQUIREMENT_LEDGER": "false",
        "ENABLE_RESPONSE_CONTRACT": "false",
        "ENABLE_STRUCTURED_REVIEW": "false",
    },
    "planning": {
        "ENABLE_STRUCTURED_PLANNING": "true",
        "ENABLE_REQUIREMENT_LEDGER": "false",
        "ENABLE_RESPONSE_CONTRACT": "false",
        "ENABLE_STRUCTURED_REVIEW": "false",
    },
    "ledger": {
        "ENABLE_STRUCTURED_PLANNING": "true",
        "ENABLE_REQUIREMENT_LEDGER": "true",
        "ENABLE_RESPONSE_CONTRACT": "true",
        "ENABLE_STRUCTURED_REVIEW": "false",
    },
    "review": {
        "ENABLE_STRUCTURED_PLANNING": "true",
        "ENABLE_REQUIREMENT_LEDGER": "true",
        "ENABLE_RESPONSE_CONTRACT": "true",
        "ENABLE_STRUCTURED_REVIEW": "true",
    },
}


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    dataset: str = "conquer_val"
    num_samples: int = 1
    candidate_model: str = "Lunit/L2-preview"
    candidate_port: int = 18081
    candidate_timeout_sec: float = 180.0
    candidate_retries: int = 0
    inference_attempts: int = 1
    judge_attempts: int = 1
    candidate_concurrency: int = 1
    judge_concurrency: int = 1
    judge_model: str = "gpt-4.1"


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    dataset: str = "conquer_val"
    num_samples: int = 40
    candidate_model: str = "Lunit/L2-preview"
    candidate_port: int = 18081
    candidate_timeout_sec: float = 180.0
    candidate_retries: int = 0
    inference_attempts: int = 1
    judge_attempts: int = 1
    candidate_concurrency: int = 4
    judge_concurrency: int = 8
    judge_model: str = "gpt-4.1"


@dataclass(frozen=True, slots=True)
class Baseline100Config:
    dataset: str = "conquer_val"
    num_samples: int = 100
    candidate_model: str = "Lunit/L2-preview"
    candidate_port: int = 18081
    candidate_timeout_sec: float = 180.0
    candidate_retries: int = 0
    inference_attempts: int = 1
    judge_attempts: int = 1
    candidate_concurrency: int = 4
    judge_concurrency: int = 8
    judge_model: str = "gpt-4.1"


def load_local_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        parsed = value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
            parsed = parsed[1:-1]
        values[name.strip()] = parsed
    for required in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        if not values.get(required):
            raise ValueError(f"{required} must be set in {path.name}")
    return values


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_status_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line[3:] for line in result.stdout.splitlines() if len(line) > 3]


def build_coeval_command(
    coeval_root: Path,
    run_dir: Path,
    config: SmokeConfig | BaselineConfig | Baseline100Config,
) -> list[str]:
    candidate_base = f"http://127.0.0.1:{config.candidate_port}/v1"
    return [
        str(coeval_root / ".venv" / "bin" / "coeval"),
        f"datasets={config.dataset}",
        JUDGE_GROUP,
        f"num_samples={config.num_samples}",
        f"client.llm.config.api_base={candidate_base}",
        f"client.llm.config.model={config.candidate_model}",
        "client.llm.config.api_key=local-no-auth",
        f"client.llm.config.max_retries={config.candidate_retries}",
        f"client.llm.config.timeout={config.candidate_timeout_sec}",
        f"runner.concurrent_limit={config.candidate_concurrency}",
        f"runner.inference_max_attempts={config.inference_attempts}",
        "runner.inference_retry_delay_s=0",
        f"metrics.{config.dataset}.healthbench_rubric.concurrent_limit={config.judge_concurrency}",
        f"metrics.{config.dataset}.healthbench_rubric.max_attempts={config.judge_attempts}",
        f"metrics.{config.dataset}.healthbench_rubric.retry_delay_s=0",
        f"hydra.run.dir={run_dir / 'coeval'}",
        f"output.output_dir={run_dir / 'coeval'}",
    ]


def safe_command(command: Sequence[str]) -> list[str]:
    return [
        "<redacted>" if "api_key=" in item and "local-no-auth" not in item else item
        for item in command
    ]


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def _wait_until_ready(
    port: int,
    model: str,
    timeout_sec: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    models_url = f"http://127.0.0.1:{port}/v1/models"
                    with urllib.request.urlopen(
                        models_url, timeout=2
                    ) as models_response:
                        payload = json.load(models_response)
                    served = {
                        item.get("id")
                        for item in payload.get("data", [])
                        if isinstance(item, dict)
                    }
                    if model in served:
                        return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"candidate did not become ready on port {port}")


def _extract_json_event(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return (
        value
        if isinstance(value, dict) and isinstance(value.get("event"), str)
        else None
    )


def summarize_service_log(path: Path) -> dict[str, Any]:
    events = [
        event
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if (event := _extract_json_event(line)) is not None
    ]
    counts: dict[str, int] = {}
    for event in events:
        name = str(event["event"])
        counts[name] = counts.get(name, 0) + 1
    completed = [event for event in events if event["event"] == "request_completed"]
    last = completed[-1] if completed else {}
    allowed_last = {
        key: last[key]
        for key in (
            "lane",
            "planned_lane",
            "planner_mode",
            "ledger_mode",
            "contract_mode",
            "review_mode",
            "fallback_reason",
            "l2_calls",
            "retrieval_l2_calls",
            "retrievals",
            "mcp_calls",
            "retrieval_status",
            "evidence_count",
            "requirement_count",
            "review_status",
            "review_issue_categories",
            "generation_latency_ms",
            "retrieval_latency_ms",
            "review_latency_ms",
            "latency_ms",
            "usage",
        )
        if key in last
    }
    return {"event_counts": counts, "last_completed_request": allowed_last}


def audit_artifacts(
    run_dir: Path,
    dataset: str,
    *,
    complete_status: str = "smoke_complete",
) -> dict[str, Any]:
    coeval_dir = run_dir / "coeval"
    summary_path = coeval_dir / f"summary_{dataset}.json"
    results_path = coeval_dir / f"results_{dataset}.json"
    combined_path = coeval_dir / "summary_combined.json"
    if (
        not summary_path.exists()
        or not results_path.exists()
        or not combined_path.exists()
    ):
        return {"status": "failed", "reason": "required CoEval artifacts are missing"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise TypeError("CoEval results artifact must be an array")
    sample_ids = [item.get("sample_id") for item in results if isinstance(item, dict)]
    inference_failed = sum(
        bool(item.get("inference_failed")) for item in results if isinstance(item, dict)
    )
    scoring_failed = sum(
        bool(item.get("scoring_failed")) for item in results if isinstance(item, dict)
    )
    generation_times = [
        float(item["generation_time_ms"])
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("generation_time_ms"), (int, float))
    ]
    metric_scores = summary.get("metric_scores", {})
    primary_name = (
        next(iter(metric_scores), None) if isinstance(metric_scores, dict) else None
    )
    primary = metric_scores.get(primary_name, {}) if primary_name else {}
    combined_datasets = (
        sorted(combined.get("per_dataset", {})) if isinstance(combined, dict) else []
    )
    consistent = (
        summary.get("num_samples") == len(results)
        and summary.get("num_inference_failed") == inference_failed
        and summary.get("num_scoring_failed") == scoring_failed
        and len(set(sample_ids)) == len(results)
        and combined_datasets == [dataset]
    )
    return {
        "status": (
            "failed"
            if not consistent
            else complete_status
            if not inference_failed and not scoring_failed
            else "failure_contaminated"
        ),
        "artifact_consistent": consistent,
        "dataset": summary.get("dataset"),
        "num_samples": summary.get("num_samples"),
        "num_evaluated": summary.get("num_evaluated"),
        "num_passed": summary.get("num_passed"),
        "pass_rate": summary.get("pass_rate"),
        "num_inference_failed": summary.get("num_inference_failed"),
        "num_scoring_failed": summary.get("num_scoring_failed"),
        "total_time_s": summary.get("total_time_s"),
        "avg_generation_ms": summary.get("avg_generation_ms"),
        "avg_scoring_ms": summary.get("avg_scoring_ms"),
        "primary_metric": primary_name,
        "primary_score": primary.get("score") if isinstance(primary, dict) else None,
        "primary_numerator": primary.get("numerator")
        if isinstance(primary, dict)
        else None,
        "primary_denominator": primary.get("denominator")
        if isinstance(primary, dict)
        else None,
        "result_count": len(results),
        "unique_sample_ids": len(set(sample_ids)),
        "observed_inference_failures": inference_failed,
        "observed_scoring_failures": scoring_failed,
        "max_generation_ms": max(generation_times) if generation_times else None,
        "combined_num_datasets": combined.get("num_datasets")
        if isinstance(combined, dict)
        else None,
        "combined_total_samples": combined.get("total_samples")
        if isinstance(combined, dict)
        else None,
        "combined_datasets": combined_datasets,
    }


def run_evaluation(
    args: argparse.Namespace,
    *,
    config: SmokeConfig | BaselineConfig | Baseline100Config,
    profile: str,
    sampling: str,
) -> int:
    coeval_root = Path(args.coeval_root).resolve()
    env_path = Path(args.env_file).resolve()
    secrets = load_local_env(env_path)
    if not (coeval_root / ".venv" / "bin" / "coeval").is_file():
        raise FileNotFoundError("prepared CoEval CLI was not found")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{profile}"
    run_dir = Path(args.runs_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    image = f"lunit-harness:coeval-{run_id.lower()}"
    container = f"lunit-coeval-{run_id.lower()}"
    command = build_coeval_command(coeval_root, run_dir, config)
    project_changes = git_status_paths(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "sampling": sampling,
        "config": asdict(config),
        "candidate_endpoint": f"http://127.0.0.1:{config.candidate_port}/v1",
        "judge_endpoint_host": urlsplit(secrets["OPENAI_API_BASE"]).netloc,
        "project_commit": git_head(PROJECT_ROOT),
        "project_dirty": bool(project_changes),
        "project_changed_paths": project_changes,
        "coeval_commit": git_head(coeval_root),
        "command": safe_command(command),
        "secret_env_names": ["OPENAI_API_KEY", "OPENAI_API_BASE"],
        "harness_semantic_mode": args.semantic_mode,
        "harness_modes": {
            key.lower(): value == "true"
            for key, value in SEMANTIC_MODES[args.semantic_mode].items()
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    container_started = False
    coeval_exit_code: int | None = None
    try:
        _run(["docker", "build", "-q", "-t", image, "."], cwd=PROJECT_ROOT)
        candidate_environment: list[str] = []
        for name, value in SEMANTIC_MODES[args.semantic_mode].items():
            candidate_environment.extend(["-e", f"{name}={value}"])
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "-p",
                f"127.0.0.1:{config.candidate_port}:8000",
                "--name",
                container,
                *candidate_environment,
                image,
            ],
            cwd=PROJECT_ROOT,
        )
        container_started = True
        _wait_until_ready(config.candidate_port, config.candidate_model)
        run_env = dict(os.environ)
        run_env.update(secrets)
        with (run_dir / "coeval.log").open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=coeval_root,
                env=run_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        coeval_exit_code = completed.returncode
    finally:
        if container_started:
            logs = _run(
                ["docker", "logs", container],
                cwd=PROJECT_ROOT,
                check=False,
            )
            (run_dir / "service.log").write_text(
                (logs.stdout or "") + (logs.stderr or ""),
                encoding="utf-8",
            )
            _run(
                ["docker", "stop", "-t", "10", container],
                cwd=PROJECT_ROOT,
                check=False,
            )

    complete_status = f"{profile}_complete"
    report = audit_artifacts(
        run_dir,
        config.dataset,
        complete_status=complete_status,
    )
    report["coeval_exit_code"] = coeval_exit_code
    service_log = run_dir / "service.log"
    if service_log.exists():
        report["service"] = summarize_service_log(service_log)
    report["artifact_dir"] = str(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0 if report.get("status") == complete_status and coeval_exit_code == 0 else 1
    )


def run_smoke(args: argparse.Namespace) -> int:
    return run_evaluation(
        args,
        config=SmokeConfig(candidate_port=args.port),
        profile="smoke",
        sampling="CoEval deterministic first-N smoke; not comparable as a benchmark",
    )


def run_baseline(args: argparse.Namespace) -> int:
    if not 1 <= args.num_samples <= 40:
        raise ValueError("baseline40 --num-samples must be between 1 and 40")
    return run_evaluation(
        args,
        config=BaselineConfig(
            num_samples=args.num_samples,
            candidate_port=args.port,
            candidate_concurrency=args.candidate_concurrency,
        ),
        profile="baseline40",
        sampling=(
            f"CoEval conquer_val deterministic prompt_id-sorted first {args.num_samples}; "
            "fixed paired-comparison scope"
        ),
    )


def run_baseline100(args: argparse.Namespace) -> int:
    return run_evaluation(
        args,
        config=Baseline100Config(candidate_port=args.port),
        profile="baseline100",
        sampling=(
            "CoEval conquer_val deterministic prompt_id-sorted first 100; "
            "fixed paired-comparison scope"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-safe CoEval development lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke", help="run one non-comparable conquer_val smoke"
    )
    smoke.add_argument("--coeval-root", default=str(DEFAULT_COEVAL_ROOT))
    smoke.add_argument("--env-file", default=str(PROJECT_ROOT / ".env.coeval.local"))
    smoke.add_argument("--runs-root", default=str(RUNS_ROOT))
    smoke.add_argument("--port", type=int, default=18081)
    smoke.add_argument("--timeout", type=float, default=1200.0)
    smoke.add_argument("--semantic-mode", choices=sorted(SEMANTIC_MODES), default="legacy")
    smoke.set_defaults(handler=run_smoke)
    baseline = subparsers.add_parser(
        "baseline40", help="run the fixed 40-sample conquer_val baseline"
    )
    baseline.add_argument("--coeval-root", default=str(DEFAULT_COEVAL_ROOT))
    baseline.add_argument(
        "--env-file", default=str(PROJECT_ROOT / ".env.coeval.local")
    )
    baseline.add_argument("--runs-root", default=str(RUNS_ROOT))
    baseline.add_argument("--port", type=int, default=18081)
    baseline.add_argument("--num-samples", type=int, default=40)
    baseline.add_argument("--candidate-concurrency", type=int, default=4)
    baseline.add_argument("--timeout", type=float, default=7200.0)
    baseline.add_argument(
        "--semantic-mode", choices=sorted(SEMANTIC_MODES), default="legacy"
    )
    baseline.set_defaults(handler=run_baseline)
    baseline100 = subparsers.add_parser(
        "baseline100", help="run the fixed 100-sample conquer_val baseline"
    )
    baseline100.add_argument("--coeval-root", default=str(DEFAULT_COEVAL_ROOT))
    baseline100.add_argument(
        "--env-file", default=str(PROJECT_ROOT / ".env.coeval.local")
    )
    baseline100.add_argument("--runs-root", default=str(RUNS_ROOT))
    baseline100.add_argument("--port", type=int, default=18081)
    baseline100.add_argument("--timeout", type=float, default=7200.0)
    baseline100.add_argument(
        "--semantic-mode", choices=sorted(SEMANTIC_MODES), default="legacy"
    )
    baseline100.set_defaults(handler=run_baseline100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"coeval_lab_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.coeval_lab import (
    Baseline100Config,
    BaselineConfig,
    SmokeConfig,
    audit_artifacts,
    build_coeval_command,
    build_parser,
    load_local_env,
    safe_command,
    summarize_service_log,
)


class CoEvalLabTests(unittest.TestCase):
    def test_env_loader_requires_both_judge_values_without_echoing_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.coeval.local"
            path.write_text(
                "OPENAI_API_KEY=super-secret-value\n"
                "OPENAI_API_BASE=https://judge.example/v1\n",
                encoding="utf-8",
            )
            loaded = load_local_env(path)

        self.assertEqual(set(loaded), {"OPENAI_API_KEY", "OPENAI_API_BASE"})

    def test_smoke_command_fixes_one_sample_retries_and_concurrency(self) -> None:
        command = build_coeval_command(
            Path("/opt/CoEval"),
            Path("/tmp/run"),
            SmokeConfig(),
        )
        rendered = " ".join(safe_command(command))

        self.assertIn("datasets=conquer_val", rendered)
        self.assertIn("num_samples=1", rendered)
        self.assertIn("conquer_judge=gpt-4.1", rendered)
        self.assertIn("client.llm.config.max_retries=0", rendered)
        self.assertIn("runner.inference_max_attempts=1", rendered)
        self.assertIn("runner.concurrent_limit=1", rendered)
        self.assertNotIn("super-secret", rendered)

    def test_baseline_command_fixes_40_samples_and_execution_policy(self) -> None:
        command = build_coeval_command(
            Path("/opt/CoEval"),
            Path("/tmp/run"),
            BaselineConfig(),
        )
        rendered = " ".join(safe_command(command))

        self.assertIn("datasets=conquer_val", rendered)
        self.assertIn("num_samples=40", rendered)
        self.assertIn("conquer_judge=gpt-4.1", rendered)
        self.assertIn("client.llm.config.max_retries=0", rendered)
        self.assertIn("runner.inference_max_attempts=1", rendered)
        self.assertIn("runner.concurrent_limit=4", rendered)
        self.assertIn(
            "metrics.conquer_val.healthbench_rubric.concurrent_limit=8", rendered
        )
        self.assertIn(
            "metrics.conquer_val.healthbench_rubric.max_attempts=1", rendered
        )

    def test_baseline_command_accepts_candidate_concurrency_override(self) -> None:
        args = build_parser().parse_args(
            ["baseline40", "--candidate-concurrency", "20"]
        )
        command = build_coeval_command(
            Path("/opt/CoEval"),
            Path("/tmp/run"),
            BaselineConfig(candidate_concurrency=args.candidate_concurrency),
        )

        self.assertIn("runner.concurrent_limit=20", command)

    def test_baseline_command_accepts_bounded_sample_override(self) -> None:
        args = build_parser().parse_args(["baseline40", "--num-samples", "20"])
        command = build_coeval_command(
            Path("/opt/CoEval"),
            Path("/tmp/run"),
            BaselineConfig(num_samples=args.num_samples),
        )

        self.assertIn("num_samples=20", command)

    def test_semantic_mode_is_explicit_and_bounded(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["baseline40"]).semantic_mode, "legacy")
        self.assertEqual(
            parser.parse_args(["baseline40", "--semantic-mode", "ledger"]).semantic_mode,
            "ledger",
        )

    def test_baseline100_command_uses_real_workload_safe_concurrency(self) -> None:
        command = build_coeval_command(
            Path("/opt/CoEval"),
            Path("/tmp/run"),
            Baseline100Config(),
        )
        rendered = " ".join(safe_command(command))

        self.assertIn("datasets=conquer_val", rendered)
        self.assertIn("num_samples=100", rendered)
        self.assertIn("conquer_judge=gpt-4.1", rendered)
        self.assertIn("runner.concurrent_limit=4", rendered)
        self.assertIn(
            "metrics.conquer_val.healthbench_rubric.concurrent_limit=8", rendered
        )
        self.assertIn("runner.inference_max_attempts=1", rendered)
        self.assertIn(
            "metrics.conquer_val.healthbench_rubric.max_attempts=1", rendered
        )

    def test_artifact_audit_never_copies_prompt_answer_or_rubric_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            coeval_dir = run_dir / "coeval"
            coeval_dir.mkdir()
            (coeval_dir / "summary_conquer_val.json").write_text(
                json.dumps(
                    {
                        "dataset": "conquer_val",
                        "num_samples": 1,
                        "num_evaluated": 1,
                        "num_passed": 1,
                        "num_inference_failed": 0,
                        "num_scoring_failed": 0,
                        "pass_rate": 1.0,
                        "total_time_s": 12.0,
                        "avg_generation_ms": 8000.0,
                        "avg_scoring_ms": 4000.0,
                        "metric_scores": {
                            "HealthBenchRubric": {
                                "score": 0.75,
                                "numerator": 0.75,
                                "denominator": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (coeval_dir / "results_conquer_val.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": 0,
                            "input": "PRIVATE_PROMPT_SENTINEL",
                            "actual_output": "PRIVATE_ANSWER_SENTINEL",
                            "rubric": "PRIVATE_RUBRIC_SENTINEL",
                            "inference_failed": False,
                            "scoring_failed": False,
                            "generation_time_ms": 8000.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (coeval_dir / "summary_combined.json").write_text(
                json.dumps(
                    {
                        "num_datasets": 1,
                        "total_samples": 1,
                        "per_dataset": {"conquer_val": {}},
                    }
                ),
                encoding="utf-8",
            )
            report = audit_artifacts(run_dir, "conquer_val")

        serialized = json.dumps(report)
        self.assertEqual(report["status"], "smoke_complete")
        self.assertTrue(report["artifact_consistent"])
        self.assertEqual(report["combined_datasets"], ["conquer_val"])
        self.assertEqual(report["primary_score"], 0.75)
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", serialized)
        self.assertNotIn("PRIVATE_ANSWER_SENTINEL", serialized)
        self.assertNotIn("PRIVATE_RUBRIC_SENTINEL", serialized)

    def test_service_summary_keeps_only_structural_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text(
                'prefix {"event":"generation_tool_rejected","reason":"invalid_arguments"}\n'
                'prefix {"event":"request_completed","lane":"GROUNDED",'
                '"latency_ms":100,"usage":{"total_tokens":12},'
                '"raw_answer":"DO_NOT_COPY"}\n',
                encoding="utf-8",
            )
            summary = summarize_service_log(path)

        serialized = json.dumps(summary)
        self.assertEqual(summary["event_counts"]["generation_tool_rejected"], 1)
        self.assertEqual(summary["last_completed_request"]["latency_ms"], 100)
        self.assertNotIn("DO_NOT_COPY", serialized)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from evaluation.load_test import LoadRecord, choose_concurrency, summarize_phase


class LoadTestTests(unittest.TestCase):
    def test_phase_summary_keeps_only_operational_fields(self) -> None:
        summary = summarize_phase(
            [
                LoadRecord(index=0, ok=True, status=200, latency_ms=100),
                LoadRecord(index=1, ok=True, status=200, latency_ms=200),
                LoadRecord(index=2, ok=False, status=503, latency_ms=300, error_type="http_error"),
            ],
            concurrency=4,
            wall_time_s=1.5,
        )

        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["latency_ms"]["p95"], 300)
        self.assertEqual(summary["error_counts"], {"http_error": 1})

    def test_concurrency_eight_requires_stability_and_speedup(self) -> None:
        phase_4 = {"wall_time_s": 100.0, "failures": 0, "success_rate": 1.0}
        phase_8 = {"wall_time_s": 60.0, "failures": 0, "success_rate": 1.0}

        decision = choose_concurrency(phase_4, phase_8)

        self.assertTrue(decision["concurrency_8_accepted"])
        self.assertEqual(decision["recommended_concurrency"], 8)

    def test_concurrency_eight_is_rejected_when_requests_fail(self) -> None:
        phase_4 = {"wall_time_s": 100.0, "failures": 0, "success_rate": 1.0}
        phase_8 = {"wall_time_s": 50.0, "failures": 1, "success_rate": 0.9}

        decision = choose_concurrency(phase_4, phase_8)

        self.assertFalse(decision["concurrency_8_accepted"])
        self.assertEqual(decision["recommended_concurrency"], 4)


if __name__ == "__main__":
    unittest.main()

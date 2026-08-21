from __future__ import annotations

import json
import unittest

from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.orchestration.driver import ConversationDriver
from tests.fakes import NeverRetrieval, ScriptedL2, l2_content, l2_tool_call


def _events(output: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in output:
        start = line.find("{")
        if start >= 0:
            events.append(json.loads(line[start:]))
    return events


class ObservabilityTests(unittest.TestCase):
    def test_generation_rejection_is_structural_and_does_not_log_arguments(
        self,
    ) -> None:
        settings = Settings(lunit_fm_api_key="test", max_generation_retrievals=1)
        l2 = ScriptedL2(
            [
                l2_tool_call(
                    "bad-call",
                    "not_a_real_tool",
                    {"private": "DO_NOT_LOG_ARGUMENTS"},
                ),
                l2_content("final answer"),
            ]
        )
        driver = ConversationDriver(
            settings,
            l2=l2,
            retrieval=NeverRetrieval(),  # type: ignore[arg-type]
        )

        with self.assertLogs("lunit_driver", level="INFO") as captured:
            result = driver.complete(
                ChatCompletionRequest(
                    model=settings.model,
                    messages=({"role": "user", "content": "private question"},),
                ),
                request_id="req-observe",
            )

        events = _events(captured.output)
        rejected = [
            event for event in events if event["event"] == "generation_tool_rejected"
        ]
        self.assertEqual(rejected[0]["reason"], "unknown_tool")
        self.assertEqual(rejected[0]["tool"], "not_a_real_tool")
        self.assertEqual(events[-1]["event"], "request_completed")
        self.assertEqual(result.content, "final answer")
        self.assertNotIn("DO_NOT_LOG_ARGUMENTS", "\n".join(captured.output))
        self.assertNotIn("private question", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()

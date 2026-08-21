"""An empty completion must be answered by changing the question, not repeating it."""
from __future__ import annotations

import unittest

from app.clients.l2 import L2Client
from app.config import Settings


def _client() -> L2Client:
    return L2Client.__new__(L2Client)


BASE = {
    "model": "Lunit/L2-preview",
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": "base instructions"},
        {"role": "system", "content": "evidence block"},
        {"role": "user", "content": "is reishi a substitute for the flu shot?"},
    ],
}


class EmptyEscalationTest(unittest.TestCase):
    def test_first_attempt_is_unchanged(self) -> None:
        self.assertEqual(_client()._escalate_on_empty(BASE, 0), BASE)

    def test_short_conversations_are_left_alone(self) -> None:
        # Nothing to trim, so escalating cannot help and must not corrupt the request.
        self.assertEqual(_client()._escalate_on_empty(BASE, 1), BASE)

    def test_escalation_trims_history_and_keeps_the_lead_system(self) -> None:
        turns = [{"role": "system", "content": "base instructions"}]
        for i in range(10):
            turns.append({"role": "user", "content": f"u{i}"})
            turns.append({"role": "assistant", "content": f"a{i}"})
        payload = {"model": "m", "temperature": 0.0, "messages": turns}

        second = _client()._escalate_on_empty(payload, 1)
        self.assertEqual(len(second["messages"]), 7)
        self.assertEqual(second["messages"][0]["content"], "base instructions")
        self.assertEqual(second["messages"][-1], turns[-1])

        third = _client()._escalate_on_empty(payload, 2)
        self.assertEqual(len(third["messages"]), 3)
        self.assertEqual(third["messages"][-1], turns[-1])

    def test_decoding_is_never_changed(self) -> None:
        # Raising temperature was measured against the live model and does not recover an
        # empty completion; it would only make the answer drift from the greedy one.
        turns = [{"role": "user", "content": f"u{i}"} for i in range(10)]
        payload = {"model": "m", "temperature": 0.0, "messages": turns}
        for attempt in (1, 2):
            self.assertEqual(_client()._escalate_on_empty(payload, attempt)["temperature"], 0.0)

    def test_original_payload_is_never_mutated(self) -> None:
        turns = [{"role": "user", "content": f"u{i}"} for i in range(10)]
        payload = {"model": "m", "temperature": 0.0, "messages": turns}
        _client()._escalate_on_empty(payload, 2)
        self.assertEqual(len(payload["messages"]), 10)

    def test_config_allows_three_attempts(self) -> None:
        self.assertEqual(Settings().empty_output_retries + 1, 3)


if __name__ == "__main__":
    unittest.main()


class EmptyRetryConfigTest(unittest.TestCase):
    def test_env_default_matches_dataclass_default(self) -> None:
        # from_env carries its own literal defaults, so a dataclass default alone never
        # reaches the running server.
        self.assertEqual(
            Settings.from_env({}).empty_output_retries,
            Settings().empty_output_retries,
        )

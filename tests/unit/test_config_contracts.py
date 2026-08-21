from __future__ import annotations

import unittest

from app.config import Settings
from app.contracts import parse_chat_completion_request
from app.errors import AppError


class SettingsTests(unittest.TestCase):
    def test_defaults_are_submission_compatible(self) -> None:
        settings = Settings.from_env({})
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.model, "Lunit/L2-preview")
        self.assertIsNone(settings.lunit_fm_api_key)

    def test_defaults_favor_family_routing_and_iterative_retrieval_budgets(self) -> None:
        settings = Settings.from_env({})
        self.assertEqual(settings.mcp_tool_mode, "family")
        self.assertEqual(settings.l2_timeout_sec, 90.0)
        self.assertEqual(settings.l2_retries, 1)
        self.assertEqual(settings.empty_output_retries, 1)
        self.assertEqual(settings.l2_max_tokens, 4_096)
        self.assertEqual(settings.request_timeout_sec, 240.0)
        self.assertEqual(settings.retrieval_timeout_sec, 90.0)
        self.assertEqual(settings.max_generation_retrievals, 2)
        self.assertEqual(settings.max_retrieval_model_rounds, 4)
        self.assertEqual(settings.max_mcp_tool_calls, 4)
        self.assertEqual(settings.max_tool_result_chars, 8_000)
        self.assertEqual(settings.max_evidence_items, 6)
        self.assertEqual(settings.max_evidence_chars, 8_000)

    def test_request_timeout_zero_disables_global_deadline(self) -> None:
        settings = Settings.from_env({"REQUEST_TIMEOUT_SEC": "0"})
        self.assertIsNone(settings.request_timeout_sec)

    def test_request_timeout_empty_string_falls_back_to_the_default_deadline(self) -> None:
        settings = Settings.from_env({"REQUEST_TIMEOUT_SEC": ""})
        self.assertEqual(settings.request_timeout_sec, 240.0)

    def test_empty_runtime_env_values_fall_back_to_safe_defaults(self) -> None:
        settings = Settings.from_env(
            {
                "HOST": "",
                "PORT": "",
                "L2_TIMEOUT_SEC": "",
                "MCP_TOOL_MODE": "",
            }
        )
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.l2_timeout_sec, 90.0)
        self.assertEqual(settings.mcp_tool_mode, "family")

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "MCP_TOOL_MODE"):
            Settings.from_env({"MCP_TOOL_MODE": "magic"})


class ContractTests(unittest.TestCase):
    def test_full_message_history_is_preserved(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "client-alias",
                "messages": [
                    {"role": "user", "content": "아스피린을 복용 중입니다."},
                    {"role": "assistant", "content": "확인했습니다."},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "그 약과 같이 먹어도 되나요?"}],
                    },
                ],
            },
            default_model="Lunit/L2-preview",
        )
        self.assertEqual(len(request.messages), 3)
        self.assertEqual(request.messages[-1]["content"], "그 약과 같이 먹어도 되나요?")

    def test_streaming_is_explicitly_rejected(self) -> None:
        with self.assertRaises(AppError) as raised:
            parse_chat_completion_request(
                {"messages": [{"role": "user", "content": "hello"}], "stream": True},
                default_model="Lunit/L2-preview",
            )
        self.assertEqual(raised.exception.code, "unsupported_streaming")

    def test_non_text_content_is_rejected(self) -> None:
        with self.assertRaises(AppError):
            parse_chat_completion_request(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": "x"}],
                        }
                    ]
                },
                default_model="Lunit/L2-preview",
            )


if __name__ == "__main__":
    unittest.main()

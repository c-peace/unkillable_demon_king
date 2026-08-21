from __future__ import annotations

import unittest
from unittest.mock import patch

from app.clients.http import HttpTimeoutError
from app.clients.l2 import L2Client, _parse_response, parse_tool_arguments
from app.config import Settings
from app.deadline import Deadline
from app.errors import UpstreamError


class L2ProtocolTests(unittest.TestCase):
    def test_specific_tool_choice_is_forwarded_with_parallel_calls_disabled(self) -> None:
        settings = Settings(lunit_fm_api_key="test", empty_output_retries=0)
        response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "plan",
                                "type": "function",
                                "function": {
                                    "name": "submit_plan",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        with patch("app.clients.l2.post_json") as post:
            post.return_value.json.return_value = response_payload
            L2Client(settings).complete(
                [{"role": "user", "content": "plan"}],
                deadline=Deadline.unbounded(),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_plan",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_plan"},
                },
            )

        payload = post.call_args.args[1]
        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "function": {"name": "submit_plan"}},
        )
        self.assertFalse(payload["parallel_tool_calls"])

    def test_timeout_is_not_retried_even_when_transient_retries_are_enabled(self) -> None:
        settings = Settings(
            lunit_fm_api_key="test",
            l2_retries=3,
            empty_output_retries=0,
        )
        with patch(
            "app.clients.l2.post_json",
            side_effect=HttpTimeoutError("upstream timed out"),
        ) as post:
            with self.assertRaises(UpstreamError) as raised:
                L2Client(settings).complete(
                    [{"role": "user", "content": "hello"}],
                    deadline=Deadline.unbounded(),
                )
        self.assertEqual(raised.exception.code, "l2_timeout")
        self.assertEqual(post.call_count, 1)

    def test_openai_tool_call_shape_is_normalized(self) -> None:
        response = _parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "retrieve_relevant_content",
                                        "arguments": '{"query":"guideline question"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.assertEqual(response.tool_calls[0].id, "call-1")
        self.assertEqual(response.tool_calls[0].name, "retrieve_relevant_content")
        self.assertEqual(
            parse_tool_arguments(response.tool_calls[0].arguments),
            {"query": "guideline question"},
        )

    def test_fenced_tool_arguments_are_repaired(self) -> None:
        self.assertEqual(
            parse_tool_arguments('```json\n{"status":"no_evidence","items":[]}\n```'),
            {"status": "no_evidence", "items": []},
        )

    def test_text_encoded_tool_call_is_normalized(self) -> None:
        response = _parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call>retrieve_relevant_content\n"
                                "<arg_key>query</arg_key>\n"
                                "<arg_value>CKD blood pressure guideline</arg_value>\n"
                                "</tool_call>"
                            ),
                        }
                    }
                ]
            }
        )
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "retrieve_relevant_content")
        self.assertEqual(
            parse_tool_arguments(response.tool_calls[0].arguments),
            {"query": "CKD blood pressure guideline"},
        )

    def test_text_encoded_tool_call_replaces_empty_native_call_list(self) -> None:
        response = _parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call>retrieve_relevant_content\n"
                                "<arg_key>query</arg_key>\n"
                                "<arg_value>official drug label</arg_value>\n"
                                "</tool_call>"
                            ),
                            "tool_calls": [],
                        }
                    }
                ]
            }
        )
        self.assertEqual(response.content, "")
        self.assertEqual(response.assistant_message["tool_calls"][0]["id"], "text_tool_call_0")
        self.assertEqual(
            response.assistant_message["tool_calls"][0]["function"]["name"],
            "retrieve_relevant_content",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.clients.l2 import _parse_response, parse_tool_arguments


class L2ProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

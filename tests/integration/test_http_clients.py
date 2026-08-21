from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.clients.l2 import L2Client
from app.clients.mcp import McpClient
from app.config import Settings
from app.deadline import Deadline


def _start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _read_payload(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, status, payload, headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


class FakeL2Handler(QuietHandler):
    attempts = 0
    authorization = ""
    received = None

    def do_POST(self):
        type(self).attempts += 1
        type(self).authorization = self.headers.get("Authorization", "")
        type(self).received = self._read_payload()
        if type(self).attempts == 1:
            self._json(500, {"error": {"message": "temporary"}})
            return
        self._json(
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "HTTP L2 response"}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )


class FakeMcpHandler(QuietHandler):
    methods = []
    saw_session = False

    def do_POST(self):
        payload = self._read_payload()
        method = payload.get("method")
        type(self).methods.append(method)
        if method == "initialize":
            self._json(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1"},
                    },
                },
                {"Mcp-Session-Id": "session-1"},
            )
            return
        if method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.headers.get("Mcp-Session-Id") == "session-1":
            type(self).saw_session = True
        if method == "tools/list":
            self._json(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "index_list_documents",
                                "description": "list docs",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                },
                            }
                        ]
                    },
                },
            )
            return
        if method == "tools/call":
            body = (
                "data: "
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "content": [
                                {"type": "text", "text": '{"cite_uid":"cite-http"}'}
                            ]
                        },
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(400, {"error": "unexpected"})


class HttpClientIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeL2Handler.attempts = 0
        FakeL2Handler.authorization = ""
        FakeL2Handler.received = None
        FakeMcpHandler.methods = []
        FakeMcpHandler.saw_session = False

    def test_l2_client_retries_transient_http_error(self):
        server, thread = _start_server(FakeL2Handler)
        try:
            settings = Settings(
                lunit_fm_api_url=f"http://127.0.0.1:{server.server_address[1]}",
                lunit_fm_api_key="test-secret",
                l2_retries=1,
                l2_timeout_sec=2,
            )
            response = L2Client(settings).complete(
                [{"role": "user", "content": "hello"}],
                deadline=Deadline.after(3),
            )
            self.assertEqual(response.content, "HTTP L2 response")
            self.assertEqual(response.usage["total_tokens"], 5)
            self.assertEqual(FakeL2Handler.attempts, 2)
            self.assertEqual(FakeL2Handler.authorization, "Bearer test-secret")
            self.assertEqual(FakeL2Handler.received["model"], "Lunit/L2-preview")
            self.assertEqual(FakeL2Handler.received["max_tokens"], 4_096)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_mcp_client_initializes_lists_and_calls_sse_tool(self):
        server, thread = _start_server(FakeMcpHandler)
        try:
            settings = Settings(
                lunit_mcp_url=f"http://127.0.0.1:{server.server_address[1]}",
                lunit_fm_api_key="test-secret",
                mcp_timeout_sec=2,
            )
            client = McpClient(settings)
            deadline = Deadline.after(3)
            tools = client.list_tools(deadline=deadline)
            result = client.call_tool(
                "index_list_documents", {"query": "ckd"}, deadline=deadline
            )
            self.assertEqual(tools[0].name, "index_list_documents")
            self.assertEqual(result["content"][0]["type"], "text")
            self.assertTrue(FakeMcpHandler.saw_session)
            self.assertEqual(
                FakeMcpHandler.methods,
                [
                    "initialize",
                    "notifications/initialized",
                    "tools/list",
                    "tools/call",
                ],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

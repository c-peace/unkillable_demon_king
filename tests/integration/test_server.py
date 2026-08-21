from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
import unittest
from concurrent.futures import ThreadPoolExecutor

from app.config import Settings
from app.orchestration.driver import DriverResult
from app.server import build_server, serve_in_thread


class FakeDriver:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request, *, request_id):
        self.requests.append((request, request_id))
        return DriverResult(
            content="server final answer",
            usage={"total_tokens": 7},
            trace={"request_id": request_id},
        )


def _request(url, *, method="GET", payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, dict(response.headers), json.loads(response.read())


class ServerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.settings = Settings(host="127.0.0.1", port=0)
        self.server = build_server(self.settings, driver=self.driver)  # type: ignore[arg-type]
        self.thread = serve_in_thread(self.server)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_models_and_chat_completion_contract(self):
        status, _, models = _request(f"{self.base}/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(models["data"][0]["id"], "Lunit/L2-preview")

        status, headers, completion = _request(
            f"{self.base}/v1/chat/completions",
            method="POST",
            headers={"X-Request-ID": "test-request"},
            payload={
                "model": "evaluator-alias",
                "messages": [
                    {"role": "user", "content": "첫 질문"},
                    {"role": "assistant", "content": "이전 답"},
                    {"role": "user", "content": "후속 질문"},
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(completion["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(completion["choices"][0]["message"]["content"], "server final answer")
        self.assertEqual(completion["usage"]["total_tokens"], 7)
        self.assertEqual(headers["X-Request-ID"], "test-request")
        self.assertEqual(len(self.driver.requests[0][0].messages), 3)

    def test_streaming_error_is_openai_shaped(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            _request(
                f"{self.base}/v1/chat/completions",
                method="POST",
                payload={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        error = raised.exception
        try:
            self.assertEqual(error.code, 400)
            body = json.loads(error.read())
            self.assertEqual(body["error"]["code"], "unsupported_streaming")
        finally:
            error.close()

    def test_unconfigured_real_service_returns_503_without_crashing(self):
        server = build_server(Settings(host="127.0.0.1", port=0, lunit_fm_api_key=None))
        thread = serve_in_thread(server)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                _request(
                    f"{base}/v1/chat/completions",
                    method="POST",
                    payload={"messages": [{"role": "user", "content": "hello"}]},
                )
            error = raised.exception
            try:
                self.assertEqual(error.code, 503)
                body = json.loads(error.read())
                self.assertEqual(body["error"]["code"], "service_not_configured")
            finally:
                error.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_concurrent_requests_remain_stateless(self):
        def send(index):
            _, _, payload = _request(
                f"{self.base}/v1/chat/completions",
                method="POST",
                headers={"X-Request-ID": f"concurrent-{index}"},
                payload={
                    "messages": [
                        {"role": "user", "content": f"independent request {index}"}
                    ]
                },
            )
            return payload["choices"][0]["message"]["content"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(send, range(8)))
        self.assertEqual(results, ["server final answer"] * 8)
        request_ids = {request_id for _, request_id in self.driver.requests}
        self.assertTrue({f"concurrent-{index}" for index in range(8)}.issubset(request_ids))


if __name__ == "__main__":
    unittest.main()


class BindRetryTests(unittest.TestCase):
    def test_build_server_retries_until_the_port_is_released(self) -> None:
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        releaser = threading.Timer(0.4, blocker.close)
        releaser.start()
        self.addCleanup(releaser.cancel)

        settings = Settings(lunit_fm_api_key="test", host="127.0.0.1", port=port)
        server = build_server(settings, bind_attempts=20, bind_retry_delay_sec=0.1)
        self.addCleanup(server.server_close)

        self.assertEqual(server.server_address[1], port)

    def test_build_server_gives_up_and_raises_the_bind_error(self) -> None:
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        port = blocker.getsockname()[1]

        settings = Settings(lunit_fm_api_key="test", host="127.0.0.1", port=port)
        with self.assertRaises(OSError):
            build_server(settings, bind_attempts=2, bind_retry_delay_sec=0.01)

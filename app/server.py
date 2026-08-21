from __future__ import annotations

import json
import logging
import signal
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from app.clients.l2 import L2Client
from app.clients.mcp import McpClient
from app.config import Settings
from app.contracts import (
    chat_completion_payload,
    model_list_payload,
    parse_chat_completion_request,
)
from app.errors import AppError
from app.orchestration.driver import ConversationDriver
from app.orchestration.retrieval import RetrievalEngine


LOGGER = logging.getLogger("lunit_server")


class DriverService:
    def __init__(
        self,
        settings: Settings,
        *,
        driver: ConversationDriver | None = None,
    ) -> None:
        self.settings = settings
        if driver is None:
            l2 = L2Client(settings)
            mcp = McpClient(settings)
            retrieval = RetrievalEngine(settings, l2=l2, mcp=mcp)
            driver = ConversationDriver(settings, l2=l2, retrieval=retrieval)
        self.driver = driver

    def complete(self, payload: Any, request_id: str) -> dict[str, Any]:
        request = parse_chat_completion_request(payload, default_model=self.settings.model)
        result = self.driver.complete(request, request_id=request_id)
        return chat_completion_payload(
            content=result.content,
            model=self.settings.model,
            usage=result.usage or None,
        )


class DriverHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: DriverService) -> None:
        super().__init__(address, DriverRequestHandler)
        self.service = service


class DriverRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LunitL2Driver/0.1"

    @property
    def service(self) -> DriverService:
        return self.server.service  # type: ignore[attr-defined,no-any-return]

    def do_GET(self) -> None:
        request_id = self._request_id()
        path = urlsplit(self.path).path
        if path == "/v1/models":
            self._send_json(HTTPStatus.OK, model_list_payload(self.service.settings.model), request_id)
            return
        if path in {"/health", "/healthz"}:
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "model": self.service.settings.model},
                request_id,
            )
            return
        self._send_error_payload(
            AppError(
                message="Route not found",
                status_code=404,
                code="not_found",
                error_type="invalid_request_error",
            ),
            request_id,
        )

    def do_POST(self) -> None:
        request_id = self._request_id()
        if urlsplit(self.path).path != "/v1/chat/completions":
            self._send_error_payload(
                AppError(
                    message="Route not found",
                    status_code=404,
                    code="not_found",
                    error_type="invalid_request_error",
                ),
                request_id,
            )
            return
        try:
            payload = self._read_json()
            response = self.service.complete(payload, request_id)
            self._send_json(HTTPStatus.OK, response, request_id)
        except AppError as exc:
            self._send_error_payload(exc, request_id)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_payload(
                AppError(
                    message="Request body must contain valid UTF-8 JSON",
                    status_code=400,
                    code="invalid_json",
                    error_type="invalid_request_error",
                ),
                request_id,
            )
        except Exception:
            LOGGER.exception("unhandled request failure request_id=%s", request_id)
            self._send_error_payload(
                AppError(
                    message="Internal server error",
                    status_code=500,
                    code="internal_error",
                    error_type="server_error",
                ),
                request_id,
            )

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise AppError(
                message="Content-Length is required",
                status_code=411,
                code="length_required",
                error_type="invalid_request_error",
            )
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise AppError(
                message="Content-Length must be an integer",
                status_code=400,
                code="invalid_content_length",
                error_type="invalid_request_error",
            ) from exc
        if length < 0 or length > self.service.settings.max_request_bytes:
            raise AppError(
                message="Request body exceeds the configured size limit",
                status_code=413,
                code="request_too_large",
                error_type="invalid_request_error",
            )
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _request_id(self) -> str:
        supplied = self.headers.get("X-Request-ID", "")
        if supplied and len(supplied) <= 128 and supplied.isascii():
            return supplied
        return f"req-{uuid.uuid4().hex}"

    def _send_error_payload(self, error: AppError, request_id: str) -> None:
        self._send_json(error.status_code, error.to_payload(), request_id)

    def _send_json(self, status: int, payload: Any, request_id: str) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("http client=%s %s", self.client_address[0], format % args)


def build_server(
    settings: Settings,
    *,
    driver: ConversationDriver | None = None,
) -> DriverHTTPServer:
    return DriverHTTPServer(
        (settings.host, settings.port),
        DriverService(settings, driver=driver),
    )


def serve(settings: Settings) -> None:
    server = build_server(settings)
    previous_handlers: dict[int, Any] = {}

    def request_shutdown(signum: int, frame: Any) -> None:
        LOGGER.info("shutdown_requested signal=%s", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
    LOGGER.info("service_started host=%s port=%s model=%s", settings.host, settings.port, settings.model)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def serve_in_thread(server: DriverHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    return thread

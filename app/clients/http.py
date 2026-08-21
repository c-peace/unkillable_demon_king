from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class HttpTransportError(Exception):
    pass


class HttpStatusError(HttpTransportError):
    def __init__(self, status: int, body: bytes = b"") -> None:
        super().__init__(f"upstream HTTP {status}")
        self.status = status
        self.body = body


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpTransportError("upstream returned invalid JSON") from exc


def _read_limited(response: Any, maximum: int) -> bytes:
    body = response.read(maximum + 1)
    if len(body) > maximum:
        raise HttpTransportError("upstream response exceeded the configured size limit")
    return body


def post_json(
    url: str,
    payload: Any,
    *,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> HttpResponse:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": "lunit-l2-clinical-orchestrator/0.1",
        **headers,
    }
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=_read_limited(response, max_response_bytes),
            )
    except urllib.error.HTTPError as exc:
        try:
            body = _read_limited(exc, max_response_bytes)
        except HttpTransportError:
            body = b""
        raise HttpStatusError(exc.code, body) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise HttpTransportError("upstream connection failed") from exc

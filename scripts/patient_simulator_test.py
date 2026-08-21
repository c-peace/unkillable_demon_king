#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


PATIENT_MODEL = "patient-simulator-ko"


def post_completion(
    url: str,
    payload: dict[str, Any],
    *,
    bearer_token: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("completion endpoint returned a non-object response")
    return parsed


def completion_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("completion response did not contain assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("completion response content was empty")
    return content


def patient_turn(
    endpoint: str,
    history: list[dict[str, str]],
    *,
    api_key: str,
    retries: int = 2,
) -> str | None:
    for attempt in range(retries + 1):
        try:
            payload = post_completion(
                endpoint,
                {"model": PATIENT_MODEL, "messages": history},
                bearer_token=api_key,
            )
            return completion_content(payload)
        except urllib.error.HTTPError as exc:
            try:
                if exc.code == 404:
                    return None
                if exc.code != 502 or attempt >= retries:
                    raise
            finally:
                exc.close()
        if attempt < retries:
            time.sleep(0.5 * (2**attempt))
    return None


def main() -> int:
    api_key = os.environ.get("LUNIT_FM_API_KEY", "").strip()
    if not api_key:
        print("LUNIT_FM_API_KEY is required; export it without committing it.", file=sys.stderr)
        return 2

    patient_base = os.environ.get(
        "PATIENT_SIMULATOR_URL", "https://patient.hackathon.lunit.io"
    ).rstrip("/")
    driver_base = os.environ.get("DRIVER_URL", "http://127.0.0.1:8000").rstrip("/")
    rounds = max(1, min(int(os.environ.get("PATIENT_SIMULATOR_ROUNDS", "3")), 3))
    patient_endpoint = f"{patient_base}/v1/chat/completions"
    driver_endpoint = f"{driver_base}/v1/chat/completions"

    history: list[dict[str, str]] = []
    completed = 0
    resets = 0
    while completed < rounds:
        question = patient_turn(patient_endpoint, history, api_key=api_key)
        if question is None:
            resets += 1
            if resets > 3:
                raise RuntimeError("patient simulator returned repeated 404 resets")
            history = []
            continue
        resets = 0
        history.append({"role": "user", "content": question})
        answer_payload = post_completion(
            driver_endpoint,
            {"model": "Lunit/L2-preview", "messages": history},
            timeout=120.0,
        )
        answer = completion_content(answer_payload)
        history.append({"role": "assistant", "content": answer})
        completed += 1
        print(f"\n[turn {completed}] user\n{question}\n\n[turn {completed}] assistant\n{answer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

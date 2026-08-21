#!/bin/sh
set -eu

base_url="${BASE_URL:-http://127.0.0.1:8000}"

health_payload="$(curl --fail --silent --show-error "${base_url}/healthz")"
models_payload="$(curl --fail --silent --show-error "${base_url}/v1/models")"

python3 -c '
import json
import sys

health = json.loads(sys.argv[1])
models = json.loads(sys.argv[2])
assert health["status"] == "ok"
assert models["object"] == "list"
assert models["data"][0]["id"] == "Lunit/L2-preview"
' "$health_payload" "$models_payload"

printf 'smoke test passed for %s\n' "$base_url"

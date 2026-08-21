# Lunit API and Patient Simulator

Last updated: 2026-08-21

Source: organizer "Build with Lunit API" and Patient Simulator screenshots, plus the organizer text copied by the user on 2026-08-21.

## Service endpoints

| Service | Base URL | Purpose |
| --- | --- | --- |
| Lunit FM | `https://model.hackathon.lunit.io` | L2 model calls |
| Patient Simulator | `https://patient.hackathon.lunit.io` | Korean medical-dialogue user/question simulation |
| Lunit MCP | `https://mcp.hackathon.lunit.io/mcp` | Streamable HTTP retrieval tools |

These Lunit assets are accessible only within the Lunit network according to the hackathon rules.

## Shared authentication

- Generate a team API key through the organizer-provided flow.
- The same team API key can be used for the Model API, Patient Simulator, and MCP endpoint.
- Send it as `Authorization: Bearer <key>`.
- API keys have a `lunit_...` form.
- Never commit or copy the real key into source, Docker layers, tests, logs, skill files, or documentation.

Use environment configuration:

```bash
LUNIT_FM_API_URL="https://model.hackathon.lunit.io"
LUNIT_FM_API_KEY="<runtime secret>"
LUNIT_FM_MODEL="Lunit/L2-preview"
```

Do not bake `LUNIT_FM_API_KEY` into the image. Pass it at runtime through the organizer-confirmed secret mechanism.

## Lunit FM Chat Completions API

- Endpoint: `POST ${LUNIT_FM_API_URL}/v1/chat/completions`
- Content type: `application/json`
- Authentication: Bearer token using `LUNIT_FM_API_KEY`
- Confirmed model identifier: `Lunit/L2-preview`
- Basic request shape follows OpenAI Chat Completions with `model` and `messages`.
- Basic response content is available at `choices[0].message.content` in the organizer example.

Minimal logical request:

```json
{
  "model": "Lunit/L2-preview",
  "messages": [
    {"role": "system", "content": "<stage-specific system prompt>"},
    {"role": "user", "content": "<input>"}
  ]
}
```

The organizer material supplied so far contains the heading for advanced Chat Completions tool calling, but not the expanded parameter documentation. The development endpoint behavior below was therefore verified directly rather than inferred from the heading.

## Live development observations

Observed on 2026-08-21 using a runtime-injected team credential. No credential value, raw authorization header, or sensitive prompt content was retained.

- A basic non-streaming Chat Completions request returned HTTP `200`, model `Lunit/L2-preview`, a non-empty `choices[0].message.content`, and token usage.
- The endpoint accepted the standard OpenAI `tools` array and a forced function `tool_choice`.
- The assistant message returned a standard `tool_calls` array whose function contained `name` and JSON-encoded `arguments`.
- A second request containing the original assistant `tool_calls` message followed by a `role: "tool"` result produced a non-empty final assistant response. The current harness adapter is compatible with this continuation contract.
- The tool-call response used `finish_reason: "stop"` even though `message.tool_calls` was present. Harness logic must branch on `message.tool_calls`, not require `finish_reason == "tool_calls"`.
- Observed response message keys included `role`, `content`, `tool_calls`, `function_call`, `reasoning`, `refusal`, `annotations`, and `audio`. Only the fields required by the harness should be consumed.

These observations confirm the non-streaming standard tool-call path used by this project. They do not establish support for streaming tool-call deltas, parallel calls, every optional generation parameter, or future endpoint versions.

## Patient Simulator role contract

- The Patient Simulator is an OpenAI-compatible question generator that reproduces the **user** side of a Korean medical conversation, such as a patient or clinician.
- The team's harness is the conversation **assistant**.
- Simulator output is appended to history as a `user` message.
- The team's system response is appended as an `assistant` message.
- The client must maintain the complete conversation history and send it on every simulator request.
- No session ID is required.
- Simulator model identifier: `patient-simulator-ko`.

The Patient Simulator is a development and testing service. Do not make the submitted evaluation runtime depend on it.

## Starting a simulated conversation

Send an empty `messages` array:

```json
{
  "model": "patient-simulator-ko",
  "messages": []
}
```

- Each empty-history request generates a new first question.
- Read the generated question from `choices[0].message.content`.
- Concurrent requests are supported.
- The organizer reports approximately 14 seconds for one initial-question request; treat this as planning guidance, not a guaranteed latency SLA.

## Requesting a follow-up

Append the exact received simulator question and the harness answer, then resend the entire history:

```json
{
  "model": "patient-simulator-ko",
  "messages": [
    {"role": "user", "content": "<exact received question>"},
    {"role": "assistant", "content": "<harness answer>"}
  ]
}
```

- Preserve the first question exactly; modifying it can break conversation continuation.
- The organizer reports approximately 8 seconds to generate a follow-up; treat this as planning guidance, not a guaranteed latency SLA.
- Stop after roughly three turns because longer conversations may repeat the same question.

## Patient Simulator recovery behavior

- On HTTP `404`, discard the current simulator history and start a new conversation with an empty `messages` array.
- On HTTP `502`, retry the request.
- Use bounded retries with timeouts and backoff in automation so a transient simulator failure cannot create an infinite loop.
- Preserve the exact last successful history when retrying a `502`; do not append duplicate turns before a successful response.

## Suggested development checks

- [x] Basic L2 request succeeds with runtime-injected credentials.
- [x] Standard non-streaming tool request, tool-call parsing, tool-result continuation, and final content are compatible with the harness.
- [x] API key is redacted from the recorded probe output, logs, and documentation.
- [ ] Patient Simulator produces a first Korean medical question from empty history.
- [ ] The harness answers as `assistant` and sends the complete unmodified history for follow-ups.
- [ ] A three-turn simulated conversation completes without role inversion or duplicated history.
- [ ] `404` resets the conversation and `502` retries without duplicate turns.
- [ ] Simulator latency does not block unrelated test cases when concurrency is used.

## Still unresolved

- Rate limits, concurrency limits, request timeout, and retry headers
- Whether model and simulator expose `GET /v1/models`
- Maximum message/context size and supported generation parameters
- Streaming, parallel tool-call, and structured-output behavior
- API-key injection mechanism in official evaluation
- Patient Simulator live behavior in the current development environment

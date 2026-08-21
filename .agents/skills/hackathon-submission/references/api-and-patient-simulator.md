# Lunit API and Patient Simulator

Last updated: 2026-08-21

Source: organizer "Build with Lunit API" and Patient Simulator screenshots supplied by the user on 2026-08-21.

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

The guide contains an advanced Chat Completions tool-calling section, but its details were not included in the supplied screenshot. Do not invent tool schemas or assume complete OpenAI compatibility until that material is provided or endpoint behavior is verified.

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

- [ ] Basic L2 request succeeds with runtime-injected credentials.
- [ ] API key is redacted from logs and error messages.
- [ ] Patient Simulator produces a first Korean medical question from empty history.
- [ ] The harness answers as `assistant` and sends the complete unmodified history for follow-ups.
- [ ] A three-turn simulated conversation completes without role inversion or duplicated history.
- [ ] `404` resets the conversation and `502` retries without duplicate turns.
- [ ] Simulator latency does not block unrelated test cases when concurrency is used.

## Still unresolved

- Full advanced tool-calling payload and response examples
- Rate limits, concurrency limits, request timeout, and retry headers
- Whether model and simulator expose `GET /v1/models`
- Maximum message/context size and supported generation parameters
- API-key injection mechanism in official evaluation
- MCP evaluation-time connectivity and exact per-tool schemas

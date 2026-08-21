# Project Brief

Last updated: 2026-08-21

## Confirmed context

- This is a 20-hour hackathon.
- The organizer provides a medical foundation-model endpoint.
- The organizer also provides an MCP service or MCP access.
- The Lunit FM base URL is `https://model.hackathon.lunit.io` and uses an OpenAI-compatible Chat Completions API.
- The configured L2 model identifier is `Lunit/L2-preview`.
- A Korean medical Patient Simulator is available at `https://patient.hackathon.lunit.io` for multi-turn development testing.
- The MCP Streamable HTTP endpoint is `https://mcp.hackathon.lunit.io/mcp` and uses the shared team API key as a Bearer token.
- The final output must be generated with Lunit's L2 LLM.
- L2 is a medical-specialized model with separate retrieval and generation stages, not a general-purpose chat model.
- The team is responsible for the remaining product design, orchestration, implementation, and verification.
- The submitted artifact is a containerized multi-turn conversation driver evaluated through an OpenAI-compatible HTTP interface.
- Official evaluation runs in a completely isolated environment without external access.

## Current delivery priority

Build a dependable evaluator-compatible vertical slice first: HTTP request -> multi-turn normalization -> L2 generation -> optional retrieval bridge -> L2 retrieval with MCP -> evidence returned to L2 generation -> assistant response. The path must work without external network access. Add optional product sophistication only after it is verified.

## Team decisions

- Store durable hackathon context in this repo-scoped skill rather than relying on chat history alone.
- Keep the skill entrypoint small and place evolving details in focused reference documents.
- Treat organizer-provided submission requirements as hard acceptance criteria.

## Working assumptions and unresolved boundaries

- The advanced L2 tool-calling request/response contract and exact per-tool MCP parameter/result schemas are not yet documented here.
- The evaluator's optional OpenAI API behaviors, including streaming, tool-call passthrough, and accepted request fields, are not yet confirmed.
- The organizer-internal connectivity available to L2 and MCP during isolated evaluation is not yet documented here.
- It is not yet confirmed whether retrieval and generation use the same `Lunit/L2-preview` identifier with different prompts/tools or distinct model identifiers.

Do not implement these assumptions as fixed protocol decisions without new evidence.

## Information still needed

- API-key issuance workflow details and safe injection mechanism for evaluation
- Exact MCP parameter/result schemas, `cite_uid` coverage, and evaluation-time connectivity
- Evaluator request and expected response examples
- Scoring rubric, latency limits, retry behavior, and per-trial limits
- Exact boundary of the isolated evaluation network, including how L2 and MCP remain reachable
- Packaging rules and image-size constraints for appropriately licensed offline data
- Required model name value for submission
- Repository/remote setup and organizer starter code, if any
- Product problem statement, target user, and demo scenario

## Knowledge update rule

Add new facts under the appropriate status. When a confirmed organizer statement contradicts an earlier item, mark the earlier item as superseded with the date and source. Never place secrets or real patient information in this file.

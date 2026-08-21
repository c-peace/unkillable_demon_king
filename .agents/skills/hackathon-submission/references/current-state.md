# Current State

Last updated: 2026-08-21 (Asia/Seoul)

## Current objective

Harden and evaluate the implemented Lunit L2 multi-turn conversation driver, then prepare a reproducible evaluator-compatible container and final submission branch.

## Current phase

**Evaluator-compatible vertical slice implemented; live development L2/MCP path verified.** The service implements the required HTTP/Docker boundary, full-history L2 generation, two-stage MCP retrieval, citation provenance, bounded failure handling, and feature-flagged advanced paths. Standard non-streaming L2 tool calls, dynamic MCP discovery/calls, live citation capture, and a complete L2-to-MCP-to-L2 request have succeeded. Patient Simulator, dashboard evaluation, and evaluation-environment secret/connectivity behavior remain unverified.

## Workspace status

- Git remote `origin` is connected to `https://github.com/c-peace/unkillable_demon_king.git`.
- The current local branch is `init-setting` and tracks `origin/init-setting`.
- The current HEAD is commit `d9ae6f2` (`init setting : skill 및 논문 학습 (to codex)`).
- `README.md` is the public architecture, requirements, runbook, and implementation-status document.
- `app/` contains a Python 3.13-compatible, standard-library-only service; root `Dockerfile`, `.dockerignore`, `.env.example`, synthetic tests, smoke test, and Patient Simulator development script now exist.
- The default runtime has no third-party Python dependencies and starts without downloading packages, models, or data.
- `.env` is Git-ignored and currently supplies the development runtime settings; only presence was checked and no credential value was printed or copied into project documentation.
- The user has moved `healthBench.pdf` from the repository root into `reference/`; Git currently reports the old path deleted and the `reference/` directory untracked. Preserve this user-owned workspace change until the user chooses to commit it.
- Six PDFs are present under repository-root `reference/`: HealthBench plus the five newly supplied papers. The `.agents/.../references/` directory contains the derived project notes.
- No real API key or patient data is stored in the workspace knowledge files.

## Completed

- Created the repo-scoped `$hackathon-submission` skill.
- Captured the organizer's rules, evaluation process, and submission contract.
- Captured the L2 retrieval/generation architecture and `finalize_retrieval` example.
- Captured Lunit FM, Patient Simulator, and MCP endpoints and authentication conventions without storing credentials.
- Cataloged the 21 supplied MCP tools and their intended data sources.
- Established durable context bootstrap and state-update rules in root `AGENTS.md`.
- Established compaction recovery through `.omx/notepad.md` pointing to this state file.
- Connected the workspace to the GitHub repository. The current working branch remains `init-setting`; the required final submission branch is still `lunit/hackathon-submission`.
- Added `.gitignore` protection for `.omx`, environment secrets, Python caches, and local OS artifacts.
- Reviewed all 39 pages of the public HealthBench paper through text extraction and rendered-page inspection.
- Captured benchmark structure, themes, evaluation axes, interpretation limits, and generalizable L2 harness implications in `references/healthbench-overview.md` without retaining case-specific or grader-exploitation material.
- Reviewed OpenAI's public `simple-evals` HealthBench execution flow at commit `652c89d0ca9df547706735883097e9537d40dc47` and captured interface-safe implications in `references/simple-evals-reference.md` without inspecting benchmark rows.
- Documented the advanced target architecture in `references/advanced-harness-design.md`: progressive execution lanes, request-local clinical state, claim-driven MCP retrieval, evidence registry/provenance verification, and conditional L2 review/revision.
- Reviewed five supplied papers (87 pages total) and captured their generalizable findings, evidence limitations, conflicts, anti-reverse-engineering boundaries, and architecture implications in `references/literature-synthesis.md`.
- Replaced the placeholder root README with a paper-style public design and requirements specification covering required submission/API/L2 contracts, multi-turn context, evidence/RAG behavior, reliability, test/ablation strategy, milestone sequencing, planned source structure, open risks, and final submission gates.
- Implemented `GET /v1/models`, `POST /v1/chat/completions`, OpenAI-shaped errors, stateless full-message handling, bounded request deadlines, privacy-safe traces, and graceful SIGTERM shutdown.
- Implemented the Bearer-authenticated L2 Chat Completions adapter with transient retry, empty-output retry, current and legacy tool-call parsing, and optional native-message or role-labelled case-packet representation.
- Implemented a Streamable HTTP MCP client with initialize/initialized, session handling, JSON or SSE responses, dynamic `tools/list`, `tools/call`, configurable protocol version, and session recovery.
- Implemented the Generation -> `retrieve_relevant_content` -> Retrieval L2/MCP -> `finalize_retrieval` -> Generation path, including source routing, evidence registry, unknown-citation rejection, structural sufficiency validation, and bounded `partial`/`no_evidence` fallback.
- Implemented feature-flagged high-risk audit and at most one final L2 revision, plus a bounded three-turn Patient Simulator development script.
- Verified the development L2 endpoint with a basic request and the standard OpenAI non-streaming tool-call/tool-result continuation contract. The endpoint can return `finish_reason: "stop"` alongside `message.tool_calls`, and the adapter correctly uses the latter as authoritative.
- Discovered all 21 live MCP schemas, negotiated protocol `2025-03-26`, observed successful stateless JSON calls, and validated the structured `index_get_page_content` citation shape.
- Completed a general synthetic guideline request through Generation -> Retrieval L2 -> three MCP calls -> `finalize_retrieval` -> final L2 generation, selecting four evidence items without retaining raw clinical content in project state.
- Ran one input conversation shown publicly in HealthBench paper Figure 1 as a submission-container smoke test. Only the conversation input was used transiently; the published candidate response and case-specific rubric were not used for prompting, scoring, tuning, or repository fixtures.

## Active decisions

- The required deliverable is a containerized OpenAI-compatible backend driver; a UI is not required for evaluation.
- Development time is not constrained to the original 20-hour framing; pursue the advanced design while keeping every layer independently testable, reversible, and subject to runtime ablation.
- Final user-facing content must be generated by Lunit L2.
- Use separate L2 generation and retrieval stages unless evidence supports a better rule-compliant harness.
- Keep the evaluation runtime independent of public internet services and the Patient Simulator.
- Treat focused skill references as durable truth and this file as mutable execution state.
- Use HealthBench only as a general response-quality framework; do not reconstruct cases, exact criteria, hidden-set composition, or grader behavior.
- Use a stateless OpenAI-compatible service whose request `messages` array is authoritative; do not add hidden session state without organizer evidence.
- Begin with the organizer-recommended adaptive two-stage L2 design: generation exposes only `retrieve_relevant_content`, retrieval uses bounded MCP access plus `finalize_retrieval`, and the returned user content is generated by L2.
- Treat the current public `simple-evals` implementation as a reference, not proof of the hackathon grader version, optional length adjustment, or complete transport contract.
- For the long-horizon architecture, prefer a deterministic state-machine harness over a free-form autonomous agent loop: conversation-state extraction -> risk/intent routing -> optional adaptive retrieval -> evidence packaging -> L2 answer generation -> compact completeness/safety verification -> final L2 revision when risk warrants it.
- Preserve both the full raw history and a structured clinical working memory with provenance for safety-critical facts; any summarization or compression must never silently alter medications, dosages, allergies, pregnancy, timing, negation, or prior corrections.
- Keep retrieval source selection explicit and authority-ranked by task type, with fallback when router confidence is low, rather than exposing every MCP tool to every question by default.
- Use progressive `DIRECT`, `CLARIFY`, `GROUNDED`, and `HIGH-RISK` lanes so advanced retrieval, verification, and revision run only when their expected benefit justifies their latency and failure surface.
- Treat `cite_uid` provenance and the request-local evidence registry as a core subsystem; never pass arbitrary raw retrieval traces or unknown citation identifiers into the final answer path.
- Use a provenance-linked dual-track response contract: clinical/evidence requirements plus interaction/context requirements.
- Make retrieval gap-aware and sufficiency-driven; continue only for named unsupported claims and bound depth because additional rounds can add noise.
- Never infer answer confidence from retrieval relevance or source tier alone; use answer-level verification and explicit unresolved uncertainty.
- Do not adopt paper-specific specialty branches, exact length targets, hard-coded benchmark anchors, or blanket safety gates.
- Implement SEMA-style interpretation/exploration/adjudication initially inside one retrieval-stage L2 loop backed by a harness-validated Evidence Requirement Ledger; do not multiply L2 agents unless ablation shows a benefit.
- Treat `finalize_retrieval(status="sufficient")` as a validated completion decision over critical evidence requirements, not as a model's unverified impression that search results look adequate.

## Verification evidence

- `python3 /Users/peace/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/hackathon-submission` reported `Skill is valid!` after the continuity setup.
- `AGENTS.md`, `SKILL.md`, `current-state.md`, and `.omx/notepad.md` were verified to exist and be non-empty.
- `.omx/notepad.md` Priority Context measured 316 characters, below the 500-character limit.
- `git remote -v` confirmed `origin` fetch/push URLs target `c-peace/unkillable_demon_king`.
- `git status --short --branch` confirmed `init-setting...origin/init-setting` tracking during the HealthBench review.
- `git log -1 --oneline` reported `d9ae6f2 init setting : skill 및 논문 학습 (to codex)` during the README specification update.
- `healthBench.pdf` parsed as a 39-page paper; all pages were rendered and visually checked via a contact sheet, with key tables and scoring sections inspected at original resolution.
- `git ls-remote https://github.com/openai/simple-evals.git HEAD` returned `652c89d0ca9df547706735883097e9537d40dc47`; the public `healthbench_eval.py`, runner integration, and chat-completion sampler were inspected without downloading benchmark data.
- All five PDFs under `reference/` were extracted and all 87 pages rendered; complete contact sheets plus original-resolution architecture/result pages were visually checked.
- Root `README.md` contains 838 lines and all expected architecture, requirements, quick-start, implementation-status, and submission sections; `git diff --check` completed without whitespace errors after the implementation update.
- `python3 -m unittest discover -s tests -v` passed 24 unit/contract/fake integration tests covering multi-turn preservation, direct generation, retrieval/finalizer, citation validation, live page-content normalization, missing-finalizer degradation, L2 HTTP retry/tool parsing, MCP JSON/SSE/session behavior, concurrent stateless API requests, error envelopes, and optional review/revision.
- The complete 24-test suite passed under the submission-aligned `python:3.13-slim` container environment.
- The latest root Docker no-cache build completed locally in 3.1 seconds with a cached base image; the final image was 43,328,761 bytes, exposed `8000/tcp`, ran as numeric non-root user `65534:65534`, and contained no API key in image configuration/history.
- The built container called a fake host L2 endpoint with Bearer authentication and returned a valid multi-turn OpenAI completion; its privacy-safe trace recorded one L2 call and no raw message content.
- The container started and served `/healthz` successfully with `--network none`, proving startup has no external dependency. SIGTERM shutdown completed with exit code 0.
- A live development basic L2 request returned HTTP `200`, non-empty content, model metadata, and usage. A forced standard function tool call parsed correctly, and a subsequent `role: "tool"` continuation produced final content.
- Live MCP initialization negotiated `2025-03-26`; `tools/list` returned 21 schemas and tested `tools/call` responses contained `content`, `structuredContent`, and `isError` without requiring a session header.
- Fresh verification in the current workspace re-ran the submission-aligned test suite, rebuilt the root Docker image, started the service on `0.0.0.0:8000`, and confirmed `/healthz`, `/v1/models`, a basic L2 completion, a forced non-streaming `tool_calls` response, and a 21-tool MCP discovery plus `cite_uid`-bearing page-content response.
- The live non-streaming tool-call probe returned HTTP `200` with `message.tool_calls`, `finish_reason: "stop"`, and the expected `echo_probe` function call. The live MCP page-content probe returned `cite_uid` and a 1-page `pages` payload for a guideline source.
- The live end-to-end grounded request completed in 90,897 ms with two generation L2 calls, four retrieval L2 calls, three MCP calls, `sufficient` retrieval status, four evidence items, and a non-empty final L2 response. The recorded total usage was 62,626 tokens, making latency and context reduction a priority before enabling broad retrieval by default.
- The rebuilt submission image passed a live development-container check: `/healthz`, a direct L2 answer, 21-tool MCP discovery, a generic MCP call, and a full grounded L2/MCP/L2 request all succeeded with runtime-only credential injection. The first grounded container attempt returned HTTP `502`; a single manual retry succeeded in 54,655 ms with five retrieval L2 calls, five MCP calls, `partial` status, one evidence item, 989 response characters, and 89,084 tokens. This demonstrates the boundary while also exposing upstream variability and excessive retrieval context cost.
- A fresh `lunit-hackathon-driver:healthbench-smoke` no-cache build completed in 1.32 seconds locally. The image ran without manual initialization as `65534:65534`, exposed and bound container port 8000 to host `0.0.0.0:8000`, passed `/healthz`, `/v1/models`, `scripts/smoke_test.sh`, image-history secret scanning, and the complete 24-test Python 3.13 suite.
- The single public-paper smoke conversation returned HTTP `200` in 77.254 seconds with a non-empty L2 assistant response, `finish_reason: stop`, and 2,061 reported tokens. The privacy-safe trace recorded `GROUNDED`, two generation L2 calls, one retrieval request, zero completed retrieval L2/MCP calls, no evidence, and 1,604 response characters. A subsequent isolated MCP discovery from the same container returned 21 tools, so the failed retrieval stage is an observed transient/deadline-path failure rather than proof that container MCP connectivity is absent.
- General HealthBench-axis review, not an official benchmark score: the response used the multi-turn context and asked useful questions about age, feeding, wet nappies, and breathing, but exposed an internal retrieval failure, omitted some high-value red flags such as temperature and color, and made prompt in-person assessment too conditional for a newly less-active or potentially weak infant. Official pediatric guidance treats a floppy, very weak, difficult-to-wake, or non-moving infant as requiring urgent or emergency assessment.

## Open questions and blockers

- The organizer's expanded advanced tool-calling documentation is still unavailable, although the standard non-streaming request/call/continuation path has been verified directly.
- Complete response, pagination, error, and `cite_uid` coverage across all MCP tools remains unverified; only generic discovery and guideline-source calls were exercised live.
- Evaluation-time secret injection and internal L2/MCP connectivity remain unconfirmed; development-environment access is not proof of evaluator-container access.
- Hackathon-specific scoring aggregation, latency limits, and dashboard retry allowance remain unconfirmed; the public paper's HealthBench scoring method is documented separately and must not be assumed to be the organizer's complete scoring contract.
- Live Patient Simulator behavior, streaming, parallel tool calls, context limits, and rate/concurrency limits remain unverified.
- The first full grounded live request took about 91 seconds and 62.6k tokens; retrieval prompt/context compaction and latency budgets require measured optimization.
- The public-paper smoke run completed within the configured deadline but took 77.25 seconds and lost retrieval evidence before final generation. The user-facing response also disclosed the internal retrieval failure and showed possible under-triage; both are release-blocking quality issues for grounded/high-risk paths.

## Next actions

1. Prevent internal retrieval errors from appearing in user-facing text; provide a clinical-safe degraded instruction and preserve the operational error only in content-free traces.
2. Reduce retrieval token and latency cost with source-family exposure, page-content compaction, and measured tool/L2 budgets while preserving citation integrity.
3. Add synthetic high-risk pediatric triage regressions based on general clinical capabilities, not the public HealthBench example, and evaluate whether conditional review improves escalation without blanket over-triage.
4. Add an automated privacy-safe connected smoke path that distinguishes transient upstream failures from deterministic contract failures without logging content or credentials.
5. Run Patient Simulator, systematic ablations, and dashboard aggregate validation before creating the final `lunit/hackathon-submission` branch and verifying the full SHA/model.

## Update protocol

- Replace the current phase and next actions as work advances; do not append a transcript.
- Record only observed completion and verification evidence.
- Move durable organizer facts into the appropriate focused reference.
- Never record secrets, credentials, or patient data.

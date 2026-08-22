# Lunit Hackathon Workspace

Use the repo-scoped `$hackathon-submission` skill only when work depends on hackathon rules,
evaluation behavior, L2/MCP architecture, submission packaging, or new organizer evidence.
For a narrow code edit, test, explanation, or Git operation, inspect the relevant code directly.

## Context bootstrap

When the skill applies, read its `SKILL.md` and only the references selected by its router.
Read `current-state.md` when the task needs the active objective, handoff, blocker, or verified
baseline. Never preload the whole reference directory.

Newer organizer evidence supersedes older assumptions. Preserve a meaningful conflict instead of
silently rewriting it.

## State updates

Update `current-state.md` once near the end of a turn only when the active objective, architecture,
blockers, published baseline, or immediate next actions materially change. Do not append routine
passing tests, repeated Git checks, or a chronological transcript. Move stable facts to the focused
reference that owns them.

## Safety

Never store API keys, credentials, patient data, or other secrets in `AGENTS.md`, `.omx/notepad.md`, the skill, its references, source code, Docker layers, tests, or logs. Store only environment-variable names and secret-delivery mechanisms.

Before claiming submission readiness, run the verification routed by the skill and record only the
latest evidence summary.

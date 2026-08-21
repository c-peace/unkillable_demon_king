# Lunit Hackathon Workspace

This workspace uses the repo-scoped `$hackathon-submission` skill as its durable source of project knowledge.

## Context bootstrap

Before any planning, implementation, testing, evaluation, or submission work:

1. Read `.agents/skills/hackathon-submission/SKILL.md` completely.
2. Read `.agents/skills/hackathon-submission/references/current-state.md` for the active phase, verified progress, blockers, and next actions.
3. Follow the skill's routing instructions to read only the additional references relevant to the task.

Treat the reference files as the source of truth when chat history or a compacted summary differs from them. Newer organizer evidence supersedes older assumptions; record the change rather than silently overwriting a meaningful conflict.

## State updates

Update `current-state.md` in the same turn after any of the following:

- a material architecture or product decision;
- an implementation milestone;
- a build, test, dashboard trial, or evaluation result;
- discovery of a blocker, failed assumption, or new organizer requirement;
- a change to the immediate next actions.

Keep the file concise and factual. Record commands and observed results, not intended results. Move stable organizer facts into the relevant focused reference and leave only a short pointer in current state.

## Safety

Never store API keys, credentials, patient data, or other secrets in `AGENTS.md`, `.omx/notepad.md`, the skill, its references, source code, Docker layers, tests, or logs. Store only environment-variable names and secret-delivery mechanisms.

Before claiming readiness, re-run the verification required by the skill and update `current-state.md` with the evidence.


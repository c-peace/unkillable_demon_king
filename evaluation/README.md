# CoEval Development Lab

This directory is development-only and excluded from the submission image.

## Approval-gated loop

1. Run the current harness on a fixed evaluation scope.
2. Report aggregate score, failures, latency, token usage, and structural tool events.
3. Propose exactly one harness change and stop.
4. Apply the change only after user approval.
5. Re-run the same scope and compare before deciding whether to keep it.

Never inspect or copy individual prompts, answers, or rubric text into reports or
source code. Generated run artifacts stay under the Git-ignored `evaluation/runs/`.

## One-sample connection smoke

The smoke profile uses `conquer_val`, one sample, concurrency one, no candidate
transport retry, one CoEval inference attempt, and one judge attempt. It verifies
the complete candidate and judge path but is not a statistically meaningful score.
No harness optimization may be proposed or approved from the smoke result alone.

```bash
python3 -m evaluation.coeval_lab smoke
```

The command builds and starts a fresh candidate container on host port `18081`,
runs CoEval from the external `/Users/peace/Developer/CoEval` Python 3.12
environment, saves a privacy-safe manifest and report, captures structural service
logs, and removes the temporary container.

On macOS ARM, do not run a plain `mise run sync` in CoEval: the upstream default
includes Linux/NVIDIA SGLang packages. The prepared evaluation environment can be
used directly or through `UV_NO_SYNC=1 mise run eval -- ...`.

## Decision scopes

- **Smoke:** one sample, connection and artifact validation only.
- **Quick baseline:** fixed 40-sample scope for paired local comparisons.
- **Promotion:** full published `conquer_val` only for a promising candidate.
- **Confirmation:** dashboard aggregate validation before submission.

Run the fixed quick baseline with:

```bash
python3 -m evaluation.coeval_lab baseline40
```

Semantic harness factors are opt-in and cumulative so only one approved behavior
changes between paired runs:

```bash
python3 -m evaluation.coeval_lab baseline40 --semantic-mode legacy
python3 -m evaluation.coeval_lab baseline40 --semantic-mode planning
python3 -m evaluation.coeval_lab baseline40 --semantic-mode ledger
python3 -m evaluation.coeval_lab baseline40 --semantic-mode review
```

For a faster directional comparison, the same deterministic prefix can be bounded
without changing its order, for example `baseline40 --num-samples 20`. Compare
only runs with the same sample count.

`planning` enables pre-generation lane selection, `ledger` adds the requirement
ledger and dual-track response contract, and `review` adds structured issue-driven
review. The run manifest and privacy-safe request traces record the resolved modes;
results are not comparable when those fields differ beyond the intended factor.

Candidate concurrency can be overridden for an isolated throughput experiment
without changing harness behavior or judge concurrency:

```bash
python3 -m evaluation.coeval_lab baseline40 --candidate-concurrency 20
```

It always uses the prompt-ID-sorted first 40 `conquer_val` samples, candidate
concurrency 4, judge concurrency 8, no candidate transport retry, one inference
attempt, and one judge attempt. These settings are part of the comparison contract
and must remain identical for the paired post-change run.

The current decision-grade local profile uses the deterministic first 100 samples
and candidate concurrency 4:

```bash
python3 -m evaluation.coeval_lab baseline100
```

It keeps judge concurrency 8, candidate transport retries 0, one inference attempt,
and one judge attempt. A 12-request synthetic load test initially favored candidate
concurrency 8, but the first real 100-sample run produced two candidate timeouts at
that setting, so concurrency 4 remains the safe comparison default. All subsequent
paired score experiments must use this exact profile unless a new baseline is
established first.

The primary objective is aggregate HealthBench Rubric score. Failure rate, timeout
compatibility, final-L2 attribution, and P95 latency are guardrails. A local
`gpt-4.1` score is a consistent comparison proxy, not proof of the dashboard score.

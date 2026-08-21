# OpenAI simple-evals HealthBench Reference

Last reviewed: 2026-08-21 (Asia/Seoul)

Source: OpenAI `simple-evals` public repository, `main` HEAD `652c89d0ca9df547706735883097e9537d40dc47` at review time.

- Repository: `https://github.com/openai/simple-evals`
- Relevant implementation: `healthbench_eval.py`
- The repository describes itself as a lightweight transparent reference implementation and carries a July 2025 deprecation notice for new model results. Treat it as a reference, not a guaranteed copy of the hackathon evaluator.

## Compliance boundary

Inspect only the public evaluation interface, aggregation behavior, and execution flow needed to build a generally capable driver. Do not download or inspect benchmark rows, extract individual rubric contents, reproduce the grader prompt, derive answer templates, or tune against cases. The organizer's rules and hidden holdout remain authoritative.

## Confirmed public execution flow

For a standard HealthBench example, the reference implementation:

1. reads a message-list prompt and its conversation-specific criteria;
2. calls the candidate sampler once with the message list;
3. appends the sampler's text as the final assistant response;
4. evaluates that response independently against each criterion;
5. sums the weights of satisfied criteria and divides by the sum of positive weights for the example;
6. averages example scores and clips the aggregate mean to the 0–1 range;
7. also reports tag-level metrics and bootstrap uncertainty.

Negative-weight criteria represent undesirable behavior. They contribute their negative weight only when the undesirable criterion is judged present.

The current code also supports an optional response-length adjustment and a custom-input “professional mode.” Their configuration is supplied at runtime and is not evidence that the hackathon enables either option. Do not tune to an unknown length formula; produce concise but complete answers on general quality grounds.

## Interface implications

- The evaluated artifact produces ordinary assistant text, not a special rubric-shaped object.
- Retrieval traces and internal reasoning do not receive credit by themselves; their value is improving the final response.
- The final response is evaluated with the supplied conversation, so losing, rewriting, or contradicting earlier turns is costly.
- A single severe omission or harmful statement can activate a negative criterion even when the rest of the answer is strong.
- The original reference sampler receives the full message list in one invocation.

The hackathon's confirmed service contract is different at the transport boundary: the evaluator sends conversation turns to the submitted OpenAI-compatible driver, which must use the supplied context. The hackathon contract takes precedence. Implement the service statelessly with the request's `messages` array as the source of truth so it remains compatible whether the evaluator sends a growing history on every request or tests a complete conversation in one request. Do not invent server-side session state unless organizer evidence requires it.

## Recommended baseline harness

```text
POST /v1/chat/completions
  -> validate and normalize messages without changing clinical facts
  -> preserve the full raw conversation as authoritative context
  -> L2 generation stage with one tool: retrieve_relevant_content
       -> answer directly when memory and context are sufficient
       -> otherwise issue a self-contained retrieval query
            -> L2 retrieval stage with selected MCP tools + finalize_retrieval
            -> bounded search/read loop
            -> selected evidence and provenance returned to generation
       -> L2 produces the final assistant text
  -> OpenAI-compatible response envelope
```

Baseline properties:

- **Stateless transport:** no session identifier or hidden conversation store is required.
- **Adaptive retrieval:** do not retrieve for every turn; use it for current guidelines, drug safety, regulatory/coding facts, or claims where evidence materially reduces risk.
- **Raw-history preservation:** prefer the original history while it fits. If compression becomes necessary, retain both a safety-focused context capsule and the latest turns, never silently alter medication, dose, allergy, pregnancy, age, timing, negation, or prior correction.
- **One final generator:** retrieval selects evidence but never writes the user-facing answer; L2 generation produces the returned content.
- **Bounded orchestration:** configurable tool-call, retry, time, and evidence-size budgets prevent loops and latency collapse.
- **Graceful degradation:** `partial` and `no_evidence` must lead to calibrated uncertainty rather than fabricated support.

## RAG routing hypothesis

Prefer the narrowest authoritative source for the claim:

| Information need | First-choice source/tool family |
| --- | --- |
| Clinical guideline recommendation | `index_*` guideline document discovery, relevant nodes, then bounded page retrieval |
| Official drug label safety or interactions | `adr_retrieve_drug_info` / DailyMed |
| Korean approval, indication, dosage, warning | `openapi_mfds_*` |
| Korean reimbursement or oncology notice | `hira_updates_search` or relevant HIRA OpenAPI/index tools |
| Disease-code normalization or validation | `kcd_*` and HIRA disease-code validation |
| Korean statutory requirement | `openapi_law_*` search, list, then exact article retrieval |
| Research evidence or HIRA FAQ | schema-aware `rag_vector_query`; use PubMed abstracts as supporting evidence, not automatic clinical consensus |
| Adverse-event signal exploration | bounded `rag_sql_query` on FAERS, explicitly avoiding causal claims |

Do not blindly expose every tool for every query. Start by testing two approaches against organizer validation feedback and latency:

1. expose the complete trained MCP tool set to retrieval L2; or
2. use a conservative intent router to expose a smaller relevant family, with a general fallback set when routing confidence is low.

The second approach can reduce tool-schema noise and invalid calls but risks excluding a needed source. It must retain a fallback path and should not rely on brittle benchmark-specific keywords.

## Response quality contract

The generation prompt should express transferable behavior rather than benchmark language:

1. answer the user's actual question and respect requested format;
2. use all prior conversational facts and correct earlier mistakes when necessary;
3. distinguish known facts, reasonable inferences, and uncertainty;
4. cover decision-critical actions, contraindications, and red flags without blanket escalation;
5. tailor terminology and depth to the user;
6. lead with the direct answer, then give concise supporting detail and next steps;
7. cite retrieved evidence only when the provenance bridge actually supports the claim.

## Experiments allowed without reverse engineering

- Contract tests with synthetic conversations created by the team.
- General medical safety cases that are not copied or reconstructed from HealthBench.
- Ablations of no RAG vs adaptive RAG, full tool exposure vs routed exposure, and one-pass vs conditional L2 review.
- Measurements of latency, tool-call count, evidence utilization, citation correctness, multi-turn fact retention, and failure recovery.
- Dashboard score and expert feedback tracked only at aggregate capability level, never converted into case-specific rules.

## Unknowns to resolve experimentally or with organizer documentation

- Whether the hackathon evaluator sends the complete growing `messages` array on every turn.
- Whether its grader, model version, length adjustment, or aggregation differs from current public `simple-evals`.
- Request timeout, concurrency, maximum context/output, and retry policy.
- MCP availability and credential injection inside the isolated evaluator.
- Whether citations are expected in the returned answer and, if so, their rendering format.


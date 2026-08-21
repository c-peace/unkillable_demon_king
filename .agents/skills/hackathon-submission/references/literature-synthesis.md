# Five-Paper Literature Synthesis for the L2 Harness

Last reviewed: 2026-08-21 (Asia/Seoul)

Sources: the five PDFs under repository-root `reference/`. The papers were reviewed through full-text extraction, complete-page contact sheets, and original-resolution inspection of the main architecture and result pages.

## Safety and use boundary

Use only generalizable architecture, retrieval, reliability, and evaluation lessons. Do not import benchmark examples, exact prompts, published benchmark response sets, hard-coded rubric criteria, specialty branches selected from benchmark headroom, or score-specific formatting rules. In particular, do not open the response/rubric datasets linked by the VITA paper or reproduce prompt appendices from the papers.

## Executive synthesis

Across the five papers, the strongest recurring pattern is not “more agents.” It is a controlled sequence that separates:

1. clinical interpretation and interaction intent;
2. retrieval planning and query reformulation;
3. sufficiency-driven evidence gathering;
4. evidence de-duplication, conflict handling, and provenance;
5. answer synthesis for the user;
6. answer-level verification and constrained revision.

For this project, the literature strengthens the existing risk-adaptive L2 state-machine design. It specifically supports adding a gap-aware retrieval loop, a dual-track response contract, a request-local user/context profile, curated source routing, and answer-level verification. It argues against always-on safety gates, benchmark-derived specialty routing, uncontrolled retrieval depth, and confidence values derived only from retrieval similarity or source tier.

## Paper 1: SEMA-RAG

Source: `reference1_SEMA-RAG.pdf`, *SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning*, Findings of ACL 2026.

### Main contribution

SEMA-RAG divides medical RAG into three roles using the same underlying LLM with different prompts:

- Interpreter: turns a question into clinical intent, entities, constraints, and a retrieval-ready query.
- Explorer: retrieves iteratively, explicitly judges evidence sufficiency, identifies gaps, and rewrites follow-up queries.
- Arbiter: removes redundant or irrelevant evidence, reconciles conflicts, builds a source-traceable evidence report, and answers from that report.

Across five medical QA benchmarks and five model backbones, the authors report an average improvement of 6.46 accuracy points over the strongest baseline per backbone. Their ablation attributes the largest drop to removing the sufficiency-driven Explorer. Retrieval depth showed diminishing returns: most reported benefit occurred within a small number of rounds, while deeper exploration introduced noise. A 500-example HealthBench transfer experiment also improved accuracy, completeness, and instruction-following scores over their MedRAG baseline.

### Applicable to this project

- Extend the clinical working state with explicit `intent`, `entities`, `constraints`, and `retrieval_questions`.
- Make every retrieval query carry decision-critical constraints such as time course, population, care setting, comorbidity, contraindication, and jurisdiction.
- After every retrieval round, decide `sufficient`, `partial/gap`, or `no_evidence`; continue only for named missing claims.
- Build a traceable evidence report before generation rather than passing a bag of chunks.
- Test a shallow default retrieval loop with configurable early stopping; do not assume that more rounds improve performance.

### Do not copy directly

- The principal experiments are multiple-choice medical QA, not the same open-ended conversation task.
- The reported corpus and retriever differ from Lunit MCP.
- Their sufficiency logic can stop at a coarse answer that does not discriminate the remaining alternatives; our stopping rule must be claim- and action-complete.
- The system incurs substantial multi-call/token overhead and remains benchmark-based rather than clinically deployed.

## Paper 2: MDIA

Source: `reference2_MDIA.pdf`, *MDIA: A Multi-Agent Diagnostic Intelligence Pipeline on HealthBench Professional*, arXiv preprint dated 2026-05-15.

### Main contribution

MDIA describes a seven-node graph with a tool-using intake stage, a specialty router, several specialty/general reasoners, a synthesizer, and a verifier. Its most transferable results concern harness and engine behavior:

- preserving complete multi-turn context was reported as the largest improvement;
- retrying empty model output and stripping JSON code fences reduced infrastructure failures;
- broad safety gating caused over-refusal on educational or counter-misinformation tasks;
- a fixed minimal-input format helped some categories and hurt others;
- model swaps interacted with prompt and context representation rather than behaving as independent upgrades;
- evaluating the wrong entry point failed to exercise the intended graph;
- small preview subsets led to decisions that reversed on the full evaluation.

The paper also reports grader sensitivity and a trade-off between shorter output and retained clinical content.

### Applicable to this project

- Preserve all request messages and test the exact full pipeline, not only the generation entry point.
- Add schema-repair, empty-output retry, valid fallback envelopes, and traceable stage execution.
- Make safety checks conditional on task intent and patient state; distinguish education from prescriptive advice.
- Keep response density adaptive: remove filler while preserving medications, doses, time windows, thresholds, caveats, and red flags.
- Require broad synthetic regression suites and aggregate evidence before promoting a prompt or routing change.

### Conflicts and cautions

- The paper states that a simple-evals reference path kept only the last user message. Direct inspection of current OpenAI `simple-evals` main at commit `652c89d0...` showed `HealthBenchEval` calling the sampler with the complete `prompt_messages`. Treat the paper's statement as specific to its adapter/version until independently reproduced.
- Its specialty branches and anchor knowledge were selected from benchmark-specific headroom analysis. Do not adopt those branches or anchors.
- Its reported exact response-length target relates to HealthBench Professional's optional length adjustment and grader behavior. Do not encode that number as a universal rule.
- It is a self-reported preprint; several category changes were within reported bootstrap uncertainty.

## Paper 3: MedAgent

Source: `reference3_MedAgent.pdf`, *MedAgent: Trustworthy Personalized Consumer Health Search via Multi-Agent Retrieval and Verification*, IEEE SIEDS 2026.

### Main contribution

MedAgent separates query understanding, retrieval planning, evidence synthesis, and verification. It combines semantic and lexical retrieval with Reciprocal Rank Fusion, uses a structured consumer-health knowledge graph to personalize retrieval, adapts language to health literacy, ranks sources by authority, and exposes uncertainty tiers.

On a 200-example HealthBench sample using the same GPT-4o backbone for the pipeline and direct baseline, it reports improvement on six of seven themes. However, completeness and context awareness remained weak. Most importantly, its retrieval-based confidence score had essentially no correlation with answer quality; source quality and retrieval relevance did not reliably predict whether the generated answer was correct or complete.

### Applicable to this project

- Normalize consumer language into medical terminology before retrieval while retaining the original wording.
- Derive a request-local profile from conversation evidence: user role, expertise/health literacy, conditions, medicines, locale, and resource constraints.
- Use both semantic and exact retrieval when supported; exact medicine names, doses, and codes should not rely on embeddings alone.
- Prefer multi-source corroboration and de-duplicate citations.
- Tailor final language and explanation depth to the user.
- Verify the final answer itself; never expose a confidence percentage computed only from source tier, cosine similarity, or `relevance_score`.

### Do not copy directly

- Its persistent personal knowledge graph is unnecessary and risky for our stateless evaluator; use request-local state only.
- Some underlying corpora are examination or consumer-conversation datasets rather than authoritative clinical sources.
- Fixed disclaimer tiers can become boilerplate; communicate uncertainty claim by claim.
- The evaluation used only 200 examples and a different base model.

## Paper 4: Automated Rubrics

Source: `reference4_Automated Rubrics.pdf`, *Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems*, arXiv preprint 2601.15161v2, 2026.

### Main contribution

The paper generates instance-specific evaluation criteria through three stages:

1. authoritative retrieval and evidence synthesis;
2. two parallel tracks - atomic clinical facts and interaction/user constraints;
3. audit, gap analysis, removal of unsupported criteria, and refinement.

Its ablation reports the largest alignment loss when interaction-intent extraction is removed, suggesting that clinical facts alone do not define a good response. The paper also uses a constrained critique-then-refine process and reports larger response-quality gains than unguided self-critique. The reported pipeline averages six LLM calls and roughly 43 seconds per query.

### Applicable to this project

Translate the idea into a **response contract**, not a benchmark rubric:

- Clinical track: supported facts, contraindications, red flags, uncertainty, and needed actions.
- Interaction track: the user's actual intent, expertise, requested format, missing context, tone, geography, and resource constraints.
- Audit track: missing decision-critical content, unsupported claims, irrelevant content, conversation contradictions, and citation mismatch.

For high-risk responses, produce a constrained edit plan and let final L2 revise only the grounded issues. Do not allow the reviewer to introduce new medical facts absent from the context or evidence packet.

### Do not copy directly

- Do not generate or imitate HealthBench scoring criteria, axes, weights, or grader prompts inside the submission.
- The study evaluates automated rubrics against benchmark gold rubrics and LLM judges, so it is not direct clinical-outcome evidence.
- The multi-call cost is too high for an always-on path until evaluator latency limits are known.
- Exact prompts in the appendix are research artifacts, not project requirements.

## Paper 5: Corpus-specific clinical RAG / VITA

Source: `reference5_A corpus-specific clinical RAG system.pdf`, *A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench*, six-page manuscript/correspondence.

### Main contribution

VITA is described as a context-specific RAG system for India, grounded in disease-specific guidelines, local antimicrobial resistance, national formulary constraints, and resource-limited protocols. Its primary HealthBench evaluation reports advantages in accuracy, completeness, and context awareness, while general-purpose models were stronger in communication and instruction following. A later 500-question neutral-judge sensitivity analysis found statistical parity with the best frontier comparator on mean per-question score, although VITA retained higher accuracy and completeness and lower communication scores.

The paper argues that curated, context-specific corpora may avoid noise and lost-in-the-middle effects from broad unfiltered retrieval.

### Applicable to this project

- Treat corpus and jurisdiction selection as a first-class reasoning step.
- For Korean questions, distinguish MFDS approval, HIRA reimbursement, KCD coding, Korean law, and clinical recommendation.
- For global-health questions, include geography, resource availability, formulary constraints, epidemiology, and feasible alternatives.
- Prefer small authoritative evidence packets over broad indiscriminate retrieval.
- Preserve communication quality as a separate generation objective; better evidence does not automatically produce a better user-facing answer.

### Cautions

- The architecture and corpus are proprietary, so benchmark performance cannot establish which component caused the result.
- The primary evaluation pooled sequential batches produced during iterative development rather than clearly testing one frozen system across the entire set.
- Authors have declared development/financial relationships; the later neutral analysis narrowed the headline advantage to statistical parity.
- Do not inspect the paper's linked per-question responses, subset identifiers, or scoring outputs because they are unnecessary for general architecture and create reverse-engineering risk.

## Cross-paper decisions for our architecture

### 1. Clinical interpretation must precede retrieval

Create a provenance-linked clinical schema from the full conversation:

- interaction intent;
- medical entities;
- patient and scenario constraints;
- unresolved references and contradictions;
- evidence questions that are self-contained.

The raw transcript remains authoritative.

### 2. Retrieval must be gap-aware and sufficiency-driven

Each retrieval task tracks required claims. After a round, identify which claims are supported, contradicted, or missing. Continue only for named gaps and stop early when the response contract is supportable. Bound depth because additional rounds can add noise.

### 3. Build a dual-track response contract

Before final generation or high-risk review, maintain:

- a clinical/evidence track; and
- an interaction/context track.

This is a general answer plan, not a recreated evaluator rubric. It should never contain benchmark weights or case templates.

### 4. Curate evidence by authority, purpose, and jurisdiction

Use Lunit's purpose-built MCP tools before generic RAG. Keep clinical recommendation, label safety, regulatory approval, reimbursement, law, coding, research evidence, and observational safety signals distinct.

### 5. Verify at answer level

Retrieval relevance is not answer confidence. For high-risk turns, check whether the final draft uses the conversation correctly, covers the supported response contract, communicates conflicts and uncertainty, and cites only retrieved evidence.

### 6. Make safety and review conditional

Task intent matters. Education, counter-misinformation, translation, documentation, and prescriptive advice should not share one blanket refusal/safety policy. Run expensive review only for risk, evidence conflict, strict formats, or state-compression cases.

### 7. Engineer the floor before expanding the graph

Handle empty output, invalid structured output, code fences, unknown tool calls, missing finalizer, timeouts, partial evidence, and incorrect endpoint execution. Reliability failures can erase gains from sophisticated prompting.

## Architecture delta from the previous design

Keep the existing `DIRECT`, `CLARIFY`, `GROUNDED`, and `HIGH-RISK` lanes, with these refinements:

```text
raw conversation
  -> provenance-linked clinical + interaction schema
  -> response contract and evidence questions
  -> generation L2 chooses direct/clarify/retrieve
  -> retrieval L2
       -> authoritative source-family routing
       -> retrieve
       -> claim sufficiency and gap check
       -> targeted follow-up retrieval only when needed
       -> finalize_retrieval
  -> evidence registry and conflict report
  -> generation L2 draft
  -> high-risk answer-level audit against response contract
  -> constrained final L2 revision when required
```

## Required ablations before enabling advanced modules

- raw full messages vs role-labelled single-turn case packet;
- one-shot retrieval vs sufficiency-driven shallow loop;
- all MCP tools vs source-family routing with fallback;
- raw conversation only vs provenance-linked request-local profile;
- evidence packet only vs evidence packet plus conflict/gap report;
- no review vs conditional answer-level review;
- fixed disclaimers vs claim-level uncertainty;
- retrieval-only confidence vs answer-level verification;
- direct L2 vs complete advanced path across quality, latency, and failure rate.

Promote a component only when it improves general capability suites and does not depend on individual benchmark cases.


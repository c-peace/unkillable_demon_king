# HealthBench Understanding

Source: local public paper `healthBench.pdf`, reviewed 2026-08-21.

## Purpose and boundary

HealthBench evaluates the quality of an assistant's **next response** in realistic health conversations. It is not a multiple-choice medical knowledge test: conversations may be multi-turn, may contain imperfect earlier assistant responses, and require the model to combine medical accuracy with safety, context, communication, and instruction following.

Use this document only for legitimate, generalizable model-quality work. Do not:

- reconstruct, memorize, or match individual benchmark cases;
- encode the benchmark's exact criteria, answer patterns, canary, or hidden-set guesses;
- optimize against grader quirks or implement score-specific shortcuts;
- treat public examples or appendix material as production templates.

This boundary implements the organizer's prohibition on excessive HealthBench reverse engineering. The final holdout set is separate and evaluation runs in an isolated environment.

## Benchmark structure

- 5,000 realistic health conversations covering individual users and healthcare professionals.
- Conversations range from one to 19 turns; the evaluated model produces the response to the final user message.
- 262 physicians across 26 specialties contributed conversations or evaluation criteria.
- Each conversation has its own physician-written criteria rather than one universal answer key.
- Criteria carry non-zero weights from -10 to 10. Positive criteria reward desired content; negative criteria penalize harmful or undesirable content.
- A criterion receives its full weight when satisfied and zero otherwise. The example score divides the sum of satisfied weights by the total positive weight; aggregate performance is the mean example score clipped to the 0–1 range.
- HealthBench Consensus is a more physician-validated subset of recurring criteria. HealthBench Hard contains 1,000 cases selected to be difficult across several model providers.

These details explain the benchmark's shape; they are not instructions to reproduce its data or evaluator.

## Seven conversation themes

| Theme | What capability it tests |
| --- | --- |
| Global health | Adapt advice to geography, epidemiology, norms, and available resources. |
| Responding under uncertainty | Identify and communicate uncertainty without false confidence. |
| Expertise-tailored communication | Infer the user's likely expertise and adjust terminology and actionability. |
| Context seeking | Notice missing decision-critical information and ask the most useful clarification. |
| Emergency referrals | Escalate when warranted while avoiding both under-triage and blanket over-triage. |
| Health data tasks | Perform structured clinical or health-data work accurately and safely. |
| Response depth | Give enough detail for the need without burying the user in irrelevant text. |

## Five evaluation axes

| Axis | General quality requirement |
| --- | --- |
| Accuracy | Align with current evidence or consensus and acknowledge weak or evolving evidence. |
| Completeness | Include the important facts, actions, safety considerations, and red flags needed for a useful answer. |
| Context awareness | Use all conversational cues and seek clarification only when it materially affects the answer. |
| Communication quality | Be clear, well structured, concise, and appropriate for the user's level. |
| Instruction following | Satisfy the request and requested format while preserving medical safety. |

Completeness and context awareness were among the hardest capabilities in the paper. More words alone did not reliably improve scores: useful coverage and relevant depth matter more than verbosity.

## Implications for the L2 harness

Apply the following as general response-quality engineering, not benchmark-specific rules:

1. Preserve and normalize the full conversation history. Resolve references from prior turns and do not blindly repeat an earlier assistant error.
2. Decide whether the question is answerable as written. Ask a small number of high-value clarifying questions only when missing information changes safety or the recommendation; otherwise answer directly.
3. Retrieve authoritative evidence for current guidelines, medicines, regulations, or other time-sensitive claims. Preserve provenance and citation identifiers through retrieval and generation.
4. Give the direct answer or key action first, followed by the minimum reasoning, conditions, uncertainties, safety information, and practical next step needed.
5. Make escalation proportional to the supplied facts. State concrete red flags and urgency when relevant, but avoid automatic emergency disclaimers for every medical question.
6. Adapt language, terminology, recommendations, and resource assumptions to the user's role and context.
7. Run a compact completeness check before returning: core request answered, material safety issue covered, uncertainty bounded, and next action clear.
8. Favor predictable orchestration, bounded tool calls and retries, and stable prompts. Reliability across many conversations matters as much as a strong best-case answer.

## Interpretation limits

- Physician judgment can differ because of specialty, risk tolerance, severity assessment, and communication style.
- Conversation-specific criteria are broad in aggregate but may not exhaust every valid aspect of an individual answer.
- The benchmark measures a response to a supplied conversation, not product UX, real-world workflow integration, or patient outcomes.
- Historical paper scores are reference results for the evaluated systems, not targets or guarantees for Lunit L2.

## Safe development checklist

- Use organizer validation trials for end-to-end correctness and broad quality feedback.
- Diagnose failures by general capability category, such as missing context, unsafe omission, weak evidence, or unclear communication.
- Improve prompts, retrieval, and orchestration in ways that transfer to unseen medical conversations.
- Do not preserve validation questions, derive case-specific templates, or infer the hidden holdout composition.


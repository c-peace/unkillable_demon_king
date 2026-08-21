# Lunit FM L2 Usage Guide

Last updated: 2026-08-21

Source: organizer Lunit FM L2 usage-guide screenshots supplied by the user on 2026-08-21.

## Model characteristics

- L2 is a medical-specialized LLM developed by Lunit.
- It is not a general-purpose chat model and may behave unexpectedly if treated as one.
- Its intended operation explicitly separates **retrieval** and **generation** into two model-call stages.
- The retrieval model was trained to collect evidence with a specific tool set represented by the hackathon-provided MCP tools.
- The main harness challenge is controlling these two stages and connecting them reliably.

## Recommended two-stage architecture

```text
conversation context
  -> generation-stage L2
      -> answer from model memory, or
      -> retrieve_relevant_content(self-contained query)
          -> retrieval-stage L2
              -> MCP search/read calls, repeated within limits
              -> finalize_retrieval(status, selected cite_uids, scores, note)
          -> selected evidence returned to generation-stage L2
      -> final L2-generated answer
```

This is the organizer's recommended L2 usage pattern, not the only permitted system design. An alternative harness is allowed if it follows all hackathon rules, including the requirement that L2 generate the final output.

## Retrieval stage

Purpose:

- Decide what evidence is needed for the question.
- Repeatedly search, inspect, and collect relevant information with the provided MCP tools.
- Select the items judged relevant once enough information is available.
- Return citation identifiers rather than composing the final user answer.

The retrieval stage does **not** write the final answer.

### Tool exposure

Provide the retrieval model with:

- The hackathon-provided MCP tools needed to search and read evidence.
- A harness-defined `finalize_retrieval` function.

`finalize_retrieval` is not an MCP tool. Define it in the driver, expose it alongside the MCP tools, and instruct the retrieval model in its system prompt to call it to end the stage.

### Organizer `finalize_retrieval` example

The supplied example is Python/Pydantic code:

```python
from typing import Literal

from pydantic import BaseModel, Field


class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float


class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""


def finalize_retrieval(
    status: Literal["sufficient", "partial", "no_evidence"],
    items: list[CitableItem],
    note: str = "",
) -> CitationSelection:
    """Submit your final citation selection and end the retrieval phase.

    Call this only:
    - once you have gathered enough evidence to answer the query
    - the query does not need any retrieval
    - you exhausted the tool call budget and must end the retrieval
    """
    return CitationSelection(status=status, items=items, note=note)
```

This is harness-defined code, not an MCP tool. The model must call it to terminate the retrieval stage.

### Completion semantics

- Some MCP results contain `cite_uid`, which marks an item as citable and provides the identifier later used for attribution.
- At retrieval completion, return the selected items' `cite_uid` values and relevance scores through `finalize_retrieval`, not the full item contents.
- `status` is restricted to `sufficient`, `partial`, or `no_evidence`.
- `items` defaults to an empty list on `CitationSelection`, but the example function still requires an `items` argument; pass an empty list when no citable item is selected.
- `note` is optional and defaults to an empty string. Use it to carry concise retrieval outcome context into generation when needed.
- The organizer does not specify a numeric range or calibration rule for `relevance_score`; do not invent one without further evidence.
- The model should call the terminal function only after one of the three documented exit conditions is true. The harness should accept the first valid terminal call and stop further retrieval execution.
- If the tool-call budget is exhausted without a terminal call, the harness should end deterministically rather than starting another MCP call; map that fallback to an explicit non-success outcome and preserve an explanatory note.
- The shown Pydantic implementation is an example. If the implementation stack differs, preserve the same field names, allowed status values, defaults, and terminal behavior unless live API constraints require an adapter.

### Example trajectory, not a fixed recipe

The organizer illustrates a guideline lookup with calls resembling:

1. `index_list_documents(...)`
2. `index_get_relevant_nodes(...)`
3. `index_get_page_content(...)`
4. `finalize_retrieval(status="sufficient", items=[...])`

The MCP catalog confirms these tool names are available. Treat the shown arguments and call sequence as examples until their exact parameter schemas are inspected from the connected server.

## Generation stage

Purpose:

- Receive the user's question and generate the final answer.
- First determine whether L2 can answer from model memory or needs additional evidence.
- Use retrieval when precision requires sources, such as a specific guideline or legal/medical reference.

### Tool exposure

Expose only one tool to the generation model:

```python
def retrieve_relevant_content(query: str):
    """Retrieve relevant content to ground the answer using one self-contained query."""
```

The driver implements this tool by running the complete retrieval stage and returning the relevant information to the generation model. The organizer allows the harness to choose how the two stages are connected.

Do not expose the raw MCP tools or `finalize_retrieval` directly to the generation-stage model when following this recommended design.

## Prompting and control guidance

- Use a separate system prompt for each stage; do not merge retrieval and generation instructions.
- Retrieval queries must be self-contained. Resolve references such as "that medicine" or "the previous condition" using conversation context before sending the query to retrieval.
- Set an explicit maximum for retrieval tool calls to prevent loops, excess latency, and token growth.
- Define deterministic behavior for `sufficient`, `partial`, and `no_evidence`, including how generation should communicate uncertainty.
- Preserve citation provenance across the bridge so generation can cite only evidence actually selected by retrieval.

## Multi-turn constraint

- L2 is optimized for single-turn conversations.
- The hackathon evaluates multi-turn scenarios.
- The harness must compensate, for example with query rewriting and context summarization.
- Keep clinically important details, user constraints, corrections, and unresolved references when compressing context; do not let summarization silently change medical facts.

## Implementation risks to test

- Retrieval model never calls `finalize_retrieval`.
- Retrieval loops or exceeds latency/tool-call limits.
- Generation calls retrieval with a context-dependent or ambiguous query.
- Citation IDs are lost, altered, or attached to unsupported claims.
- `partial` or `no_evidence` is treated as fully supported evidence.
- Generation receives the wrong tool set or retrieval receives generation-only instructions.
- Multi-turn summarization drops negation, medication, dosage, time course, allergy, or other safety-critical context.
- The harness accidentally returns retrieval-stage text instead of a final answer generated by L2 generation.

## Still unresolved

- Whether both retrieval and generation calls use `Lunit/L2-preview` or separate model identifiers
- Advanced tool-calling request/response wire schemas
- Exact MCP parameter schemas, result formats, and which results carry `cite_uid`
- Required citation rendering format in final responses
- Model/tool-call timeout behavior and recommended call limits
- Whether the model endpoint supports parallel tool calls, streaming, or structured-output enforcement
- Maximum context and output sizes for each stage

The confirmed base endpoint, bearer authentication, environment variables, and basic Chat Completions shape are documented in [api-and-patient-simulator.md](api-and-patient-simulator.md).

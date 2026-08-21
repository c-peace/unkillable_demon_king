# Lunit MCP Tool Guide

Last updated: 2026-08-21

Source: organizer "Explore Lunit MCP tools" screenshots and copied guide text supplied by the user on 2026-08-21. The copied text reconfirmed the endpoint, authentication, timeout, prefix convention, and 21-tool catalog already captured here.

## Connection contract

- Protocol: Streamable HTTP MCP
- Endpoint: `https://mcp.hackathon.lunit.io/mcp`
- Authentication: `Authorization: Bearer <team API key>`
- Credential environment variable: `LUNIT_FM_API_KEY`
- The same team key is used for Lunit FM, Patient Simulator, and MCP.
- Never place the real key in repository files, Codex config literals, Docker layers, prompts, or logs.

Recommended timeout shown by the organizer: 60 seconds per tool call.

## Codex development configuration

Example `~/.codex/config.toml` configuration:

```toml
[mcp_servers.lunit_mcp]
url = "https://mcp.hackathon.lunit.io/mcp"
bearer_token_env_var = "LUNIT_FM_API_KEY"
required = true
tool_timeout_sec = 60
```

Before starting Codex, export `LUNIT_FM_API_KEY`, then add the server configuration, restart Codex, and use `/mcp` to verify the connection.

- `lunit_mcp` is an arbitrary local server name and may be changed.
- With that name, Codex exposes tools with the `mcp__lunit_mcp__` prefix.
- Changing the local server name can change the client-visible prefix; application logic should identify tools by their actual MCP names, not hard-code the Codex display prefix.
- This Codex configuration is for development. The submitted container must create its own MCP client connection and must not depend on a developer's `~/.codex/config.toml`.

Other Streamable HTTP MCP clients use the same endpoint and Bearer authorization header.

## Live development contract observations

Observed on 2026-08-21 through the submitted application's MCP client using a runtime-injected credential. No secret or raw authorization header was retained.

- MCP initialization negotiated protocol version `2025-03-26`.
- The server exposed 21 tools through `tools/list`.
- No `Mcp-Session-Id` response header was observed in these calls; the server operated statelessly. The client must still tolerate a future session header because Streamable HTTP permits one.
- `tools/call` responses contained `content`, `structuredContent`, and `isError`. The tested successful calls used `isError: false` and included a text content block.
- `index_get_page_content` returned structured source metadata plus `pages`, and a top-level `cite_uid`. Each page contained a 1-based page number and text. The evidence registry normalizes this shape while preserving the citation identifier, title, source type, and URL.
- A general synthetic guideline request completed the full Generation -> Retrieval L2 -> MCP -> `finalize_retrieval` -> Generation path with three MCP calls and four selected evidence items.

Live `tools/list` required parameters:

| Tool | Required parameters observed |
| --- | --- |
| `rag_get_all_data_sources` | none |
| `rag_get_data_source_detail` | `source_name` |
| `rag_sql_query` | `db_name`, `sql` |
| `rag_vector_query` | `collection_name`, `query` |
| `adr_retrieve_drug_info` | `drug_name` |
| `index_list_documents` | `corpus_tag` |
| `index_get_document_structure` | `corpus_tag`, `node_id`, `depth` |
| `index_get_relevant_nodes` | `corpus_tag`, `query` |
| `index_keyword_search` | `corpus_tag`, `query` |
| `index_get_page_content` | `corpus_tag`, `doc_id`, `start_page`, `end_page` |
| `kcd_get_name` | `code` |
| `kcd_search_codes` | `name` |
| `openapi_mfds_check_drug_permission` | `drug_name` |
| `openapi_mfds_get_drug_indication` | `drug_name` |
| `openapi_mfds_find_drugs_by_ingredient` | `ingredient` |
| `openapi_hira_get_drug_price` | `drug_name` |
| `openapi_hira_disease_check_code` | `code` |
| `openapi_law_search` | `query` |
| `openapi_law_list_articles` | `mst` |
| `openapi_law_get_article` | `mst`, `article_keys` |
| `hira_updates_search` | `query` |

The exact optional fields remain discoverable dynamically from `tools/list`; application code should continue to use the live schemas instead of duplicating this snapshot as executable configuration.

## Tool-selection overview

| Need | Preferred tool family |
| --- | --- |
| Official DailyMed label safety, adverse reactions, interactions | `adr_retrieve_drug_info` |
| Korean HIRA reimbursement updates and oncology notices | `hira_updates_search` |
| HIRA or clinical-guideline document retrieval | `index_*` |
| KCD disease-code lookup and normalization | `kcd_*` |
| HIRA billing-code validity or reimbursed drug prices | `openapi_hira_*` |
| Korean statutory text | `openapi_law_*` |
| MFDS drug approval, ingredients, and indications | `openapi_mfds_*` |
| Discover/query SQL, vector, or hybrid datasets | `rag_*` |

Choose the narrowest authoritative source that answers the question. Use discovery tools before querying an unfamiliar corpus or schema.

## Drug label and HIRA update tools

### `adr_retrieve_drug_info`

- Source: `dailymed_26_08` (DailyMed)
- Looks up major sections of official DailyMed drug labels by English brand name or INN.
- Includes warnings, adverse reactions, interactions, and a source link.

### `hira_updates_search`

- Sources: `hira_biz_infobank`, `hira_cancer_drug_notice`, `hira_cancer_drug_regimen`
- Searches current or revised guidance, oncology notices, recognized off-label oncology regimens, HIRA reimbursement notices, and published review cases.

## HIRA and clinical-guideline document tools

Catalog snapshot:

- `hira`: 249 documents
- `guideline`: 120 documents

### `index_list_documents`

- Lists documents from available HIRA and clinical-guideline collections.
- Can order results by relevance to a query.
- Use this for collection-level discovery.

### `index_get_relevant_nodes`

- Finds document sections semantically related to a query.
- Returns the matching document, ancestor nodes, and page range.
- Use this to move from a question to likely evidence locations.

### `index_get_document_structure`

- Navigates a HIRA or clinical-guideline document as a section tree.
- Starts from a selected node and returns up to 50 nodes with page ranges.
- Use this when document hierarchy or neighboring sections matter.

### `index_get_page_content`

- Returns original text from a selected document page range.
- Page numbering starts at 1.
- Maximum 20 pages per call.
- Can expose extracted flowchart paths.
- Use this after locating relevant pages; avoid broad full-document reads.

### `index_keyword_search`

- Searches document pages for exact keywords without case sensitivity.
- Ranks by exact-term occurrence count.
- Supports pagination.
- Use it for known terminology, codes, drug names, or exact phrases; use semantic node retrieval for conceptual queries.

## KCD tools

Supported versions: KCD-8 and KCD-9.

### `kcd_get_name`

- Returns the official Korean and English disease name for an exact KCD code.
- Defaults to KCD-9.

### `kcd_search_codes`

- Fuzzy-searches candidate KCD codes using a Korean or English disease name.
- Allows selecting the KCD version.
- Use search first when the exact code is unknown, then confirm the selected code with `kcd_get_name`.

## HIRA OpenAPI tools

### `openapi_hira_disease_check_code`

- Source: HIRA Disease Master OpenAPI
- Checks whether a diagnosis code is valid for HIRA billing.
- Returns code completeness and restrictions involving symptom codes, sex, age, and infectious disease classification.

### `openapi_hira_get_drug_price`

- Source: HIRA Drug Price OpenAPI
- Returns reimbursement listing status, drug code, ceiling price, deletion information, and effective-date information.

## Korean law tools

Source: Korean Law Information Center OpenAPI (`law.go.kr`).

### `openapi_law_search`

- Searches Korean laws.
- Returns the MST identifier required for later lookup of laws, administrative rules, and local ordinances.

### `openapi_law_list_articles`

- Lists provisions of a specific Korean law.
- Supports filtering by article title.
- Returns a stable article key for full-text retrieval.

### `openapi_law_get_article`

- Returns the full text of a selected provision in a citable form.
- Includes enforcement-date information and a `law.go.kr` link.

Typical path: search law -> list/filter articles -> retrieve the exact article.

## MFDS drug approval tools

### `openapi_mfds_check_drug_permission`

- Source: MFDS Drug Approval OpenAPI
- Searches by partial product name.
- Checks current approval and distinguishes valid from withdrawn approval.

### `openapi_mfds_find_drugs_by_ingredient`

- Finds MFDS-approved products using the same active ingredient.
- Returns approval status for replacement candidates.

### `openapi_mfds_get_drug_indication`

- Source: MFDS Product Approval Detail OpenAPI
- Retrieves the MFDS-approved indication.
- Can additionally return dosage, administration, warnings, ATC data, and contraindications.

## Generic RAG data tools

Known data-source identifiers:

- `pubmed_abstracts`
- `hira_faq`
- `faers_12q4_25q4`
- `dailymed_26_08`
- `kcd`

### `rag_get_all_data_sources`

- Lists identifiers and purposes for all available SQL, vector, and hybrid data sources.
- Use this before selecting a generic RAG corpus.

### `rag_get_data_source_detail`

- Shows schema, table, column, and metadata for one SQL, vector, or hybrid source.
- Inspect this before constructing SQL or interpreting source-specific metadata.

### `rag_sql_query`

- Queries structured PostgreSQL data with SQL.
- Available sources shown: `faers_12q4_25q4`, `dailymed_26_08`, `kcd`.
- Use schema discovery first and keep queries bounded.

### `rag_vector_query`

- Performs semantic-similarity search against supported Qdrant collections.
- Supports vector or dense-plus-sparse hybrid retrieval.
- Available sources shown: `pubmed_abstracts`, `hira_faq`.

## Retrieval routing principles

- Prefer official purpose-built tools over generic RAG when both can answer the same regulatory or coding fact.
- For guidelines, use document discovery or semantic nodes first, then retrieve only the required page range.
- For exact identifiers, use exact-code/name tools; use fuzzy or semantic search only to generate candidates.
- Do not treat FAERS associations as proof of causality.
- Keep source type and jurisdiction explicit: DailyMed/FDA labels, MFDS approvals, HIRA reimbursement, KCD coding, Korean law, and clinical guidelines answer different questions.
- Preserve every returned `cite_uid` and source link through retrieval selection and final generation when present.
- Bound SQL rows, page ranges, pagination, and retrieval iterations to control latency and context size.

## Verification checklist

- [x] MCP connects in the development environment with runtime-injected `LUNIT_FM_API_KEY` and no recorded secret leakage.
- [x] Tool discovery succeeds against the Streamable HTTP endpoint and returns 21 tools.
- [x] Live required/optional parameter schemas are obtained dynamically before tool selection.
- [ ] The 60-second tool timeout is handled without crashing the request.
- [x] Retrieval stops within an explicit tool-call budget in unit/fake integration tests and the live guideline path.
- [x] A live guideline `cite_uid` and source metadata survived MCP parsing, evidence registration, selection, and generation packaging.
- [ ] Tool failures yield bounded retries or a `partial`/`no_evidence` outcome.
- [x] No submitted runtime path depends on the developer's Codex MCP configuration.

## Still unresolved

- Complete response, pagination, and `cite_uid` coverage across all 21 tools
- Live error codes, retry semantics, and timeout behavior
- Evaluation-time MCP connectivity and credential injection
- Rate and concurrency limits
- Whether the catalog or corpus document counts will change during the hackathon

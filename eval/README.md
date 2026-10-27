# Agent Evaluation & Observability

This directory contains the automated evaluation framework and observability specifications for the MCP Developer Assistant.

---

## 1. How to Run the Evaluation

Run the evaluation script from the project root using the virtual environment:

```powershell
.venv\Scripts\python.exe eval/evaluate.py
# or
py eval/evaluate.py
```

The script evaluates:
- **RAG Retrieval Accuracy**: Measures top-5 semantic search hits, rank position (1–5), similarity score, and keyword coverage.
- **Security Policy Enforcement**: Verifies server-side rejection of directory traversal outside the project root.
- **GitHub Safety & Human Approval**: Verifies cryptographic HMAC approval token generation and validation.
- **Conversation Memory Continuity**: Validates stable session thread ID generation and reset handling.
- **Error Resilience**: Verifies graceful handling of empty queries and invalid inputs.

---

## 2. Evaluation Files

- **[dataset.json](file:///F:/Python/dev_assistant/eval/dataset.json)**: Evaluation dataset containing 12 representative test cases across 8 core capability categories:
  1. `repository_exploration`: Local file discovery
  2. `code_reading`: Source code inspection
  3. `rag_semantic_search`: Codebase semantic search & chunk retrieval
  4. `github_read`: Repository and issue discovery
  5. `github_safety`: Proposal drafting & approval isolation
  6. `memory_management`: Conversation thread reset
  7. `multi_step_planning`: Multi-step investigation workflows
  8. `security_authorization`: Directory traversal defense
  9. `error_handling`: Empty query resilience

- **[results.json](file:///F:/Python/dev_assistant/eval/results.json)**: Output report generated upon running `evaluate.py`. Contains pass rates, RAG hit rate percentage, average latencies (in milliseconds), and individual test case execution details.

---

## 3. Observability & Structured Logs

The system maintains two complementary JSON Lines log files in the `logs/` directory:

### A. Telemetry Log (`logs/telemetry.jsonl`)
Records fine-grained operational metrics for every request turn, tool execution, LLM call, and RAG retrieval. Every event shares a common correlation ID (`trace_id`).

**Event Types:**
- `request_started`: Request initiation with `trace_id` and `thread_id`.
- `request_completed`: Request finalization with overall `duration_ms`, `tool_count`, `llm_calls`, `llm_duration_ms`, `total_tokens`, `prompt_tokens`, `completion_tokens`, and `success`.
- `tool_call`: MCP tool invocation with `tool_name`, `duration_ms`, `success`, `error`, and sanitized parameters.
- `llm_call`: LLM execution with provider `model`, `duration_ms`, `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- `rag_retrieval`: Codebase search with `query_length`, `duration_ms`, `retrieved_count`, `top_score`, `sources`, and `success`.

### B. Audit Log (`logs/audit.jsonl`)
Records high-level application events (e.g. security checks, issue drafts, human approval grants, RAG indexing) correlated to the same `trace_id` and conversation `thread_id`:
- `trace_id`: Correlation ID matching `telemetry.jsonl`.
- `thread_id`: Active LangGraph conversation thread ID.
- `event`: Event name (e.g., `issue_drafted`, `issue_approval_granted`, `issue_created`, `rag_retrieval`, `rag_indexed`).
- Sanitized operational details (no passwords, raw credentials, or secrets are ever recorded).

---

## 4. Failure Visibility & Secret Protection

All telemetry and audit records pass through strict sanitization in `app/observability.py`:
- Sensitive dictionary keys (`api_key`, `token`, `secret`, `password`, `authorization`) are masked to `"[REDACTED]"`.
- Freeform error messages and strings are scanned with regular expressions; any embedded Groq/GitHub tokens (`gsk_*`, `ghp_*`, `github_pat_*`), Bearer tokens, or private keys are automatically replaced with `"[REDACTED_SECRET]"`.

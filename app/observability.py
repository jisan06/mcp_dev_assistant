import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
TELEMETRY_FILE = LOGS_DIR / "telemetry.jsonl"

SENSITIVE_KEY_PATTERNS = {
    "token",
    "key",
    "secret",
    "password",
    "authorization",
    "api_key",
    "github_token",
    "groq_api_key",
}


def sanitize_value(key: str, value: Any) -> Any:
    """Mask any potentially sensitive field names."""
    lower_key = key.lower()
    if any(pat in lower_key for pat in SENSITIVE_KEY_PATTERNS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: sanitize_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(key, item) for item in value]
    return value


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of a dictionary."""
    return {k: sanitize_value(k, v) for k, v in data.items()}


def write_telemetry_event(event_type: str, trace_id: str, **data: Any) -> None:
    """Append a structured, sanitized telemetry record to logs/telemetry.jsonl."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "trace_id": trace_id,
            **sanitize_dict(data),
        }

        with TELEMETRY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Observability logging must never crash the main application loop
        pass


import contextvars

current_trace_var: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "current_trace_var", default=None
)


def get_current_trace() -> Optional["TraceContext"]:
    """Retrieve the currently active trace context for the async task."""
    return current_trace_var.get()


class TraceContext:
    """Context manager for tracing an entire user turn."""

    def __init__(self, trace_id: Optional[str] = None, thread_id: Optional[str] = None):
        self.trace_id = trace_id or f"trace-{uuid4().hex[:12]}"
        self.thread_id = thread_id or "default"
        self.start_time: float = 0.0
        self.tool_count: int = 0
        self.errors: List[str] = []
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self._token = current_trace_var.set(self)
        write_telemetry_event(
            "request_started",
            self.trace_id,
            thread_id=self.thread_id,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        success = exc_type is None and not self.errors
        error_msg = str(exc_val) if exc_val else (self.errors[-1] if self.errors else None)

        if self._token:
            current_trace_var.reset(self._token)

        write_telemetry_event(
            "request_completed",
            self.trace_id,
            thread_id=self.thread_id,
            duration_ms=duration_ms,
            tool_count=self.tool_count,
            success=success,
            error=error_msg,
        )
        return False  # Do not suppress exceptions

    def record_tool(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metrics for an executed tool call."""
        self.tool_count += 1
        if error:
            self.errors.append(error)

        write_telemetry_event(
            "tool_call",
            self.trace_id,
            tool_name=tool_name,
            duration_ms=round(duration_ms, 2),
            success=success,
            error=error,
            details=details or {},
        )

    def record_llm(
        self,
        model: str,
        duration_ms: float,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record token usage and provider response metrics."""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if usage:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens") or (
                (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens or completion_tokens else None
            )

        write_telemetry_event(
            "llm_call",
            self.trace_id,
            model=model,
            duration_ms=round(duration_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=error,
        )

    def record_rag(
        self,
        query: str,
        duration_ms: float,
        retrieved_count: int,
        top_score: float,
        sources: Optional[List[str]] = None,
    ) -> None:
        """Record RAG retrieval performance and source metadata."""
        write_telemetry_event(
            "rag_retrieval",
            self.trace_id,
            query_length=len(query),
            duration_ms=round(duration_ms, 2),
            retrieved_count=retrieved_count,
            top_score=round(top_score, 4),
            sources=sources or [],
        )


def extract_llm_usage(message: Any) -> Optional[Dict[str, Any]]:
    """Safely extract token usage metadata from an AIMessage."""
    # Check response_metadata (LangChain ChatGroq format)
    meta = getattr(message, "response_metadata", {})
    if isinstance(meta, dict):
        # Groq usage dictionary
        token_usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(token_usage, dict):
            return token_usage

    # Check usage_metadata (LangChain Core AIMessage format)
    usage_meta = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        return {
            "prompt_tokens": usage_meta.get("input_tokens"),
            "completion_tokens": usage_meta.get("output_tokens"),
            "total_tokens": usage_meta.get("total_tokens"),
        }

    return None

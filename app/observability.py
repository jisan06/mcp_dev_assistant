import json
import os
import re
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
    "password",
    "authorization",
    "api_key",
    "github_token",
    "groq_api_key",
    "approval_secret",
    "secret",
    "access_token",
    "auth_token",
}

SAFE_METRIC_KEYS = {
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "token_count",
}

SECRET_STRING_PATTERNS = [
    re.compile(r"\b(?:gsk|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{25,}\b", re.IGNORECASE),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
]


def sanitize_text(text: str) -> str:
    """Mask credentials, tokens, or private keys appearing in freeform error text or strings."""
    if not isinstance(text, str):
        return text
    clean = text
    for pat in SECRET_STRING_PATTERNS:
        clean = pat.sub("[REDACTED_SECRET]", clean)
    return clean


def sanitize_value(key: str, value: Any) -> Any:
    """Mask any potentially sensitive field names or secret strings."""
    lower_key = key.lower()
    if lower_key not in SAFE_METRIC_KEYS and any(pat in lower_key for pat in SENSITIVE_KEY_PATTERNS):
        return "[REDACTED]"
    if isinstance(value, str):
        return sanitize_text(value)
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

current_trace_var: contextvars.ContextVar[Optional["TraceContext"]] = (
    contextvars.ContextVar("current_trace_var", default=None)
)


def get_current_trace() -> Optional["TraceContext"]:
    """Retrieve the currently active trace context for the async task."""
    return current_trace_var.get()


class TraceContext:
    """Context manager for tracing an entire user turn with correlation ID and performance metrics."""

    def __init__(self, trace_id: Optional[str] = None, thread_id: Optional[str] = None):
        self.trace_id = trace_id or f"trace-{uuid4().hex[:12]}"
        self.thread_id = thread_id or "default"
        self.start_time: float = 0.0
        self.tool_count: int = 0
        self.errors: List[str] = []
        self._token: Optional[contextvars.Token] = None

        # Cumulative metrics across this trace
        self.llm_count: int = 0
        self.llm_duration_ms: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.rag_count: int = 0
        self.rag_duration_ms: float = 0.0

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
            llm_calls=self.llm_count,
            llm_duration_ms=round(self.llm_duration_ms, 2),
            total_tokens=self.total_tokens if self.total_tokens > 0 else None,
            prompt_tokens=self.prompt_tokens if self.prompt_tokens > 0 else None,
            completion_tokens=self.completion_tokens if self.completion_tokens > 0 else None,
            rag_calls=self.rag_count,
            rag_duration_ms=round(self.rag_duration_ms, 2),
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
        self.llm_count += 1
        self.llm_duration_ms += duration_ms

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if usage:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens") or (
                (prompt_tokens or 0) + (completion_tokens or 0)
                if prompt_tokens or completion_tokens
                else None
            )

            if prompt_tokens:
                self.prompt_tokens += prompt_tokens
            if completion_tokens:
                self.completion_tokens += completion_tokens
            if total_tokens:
                self.total_tokens += total_tokens

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
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record RAG retrieval performance and source metadata."""
        self.rag_count += 1
        self.rag_duration_ms += duration_ms
        if error:
            self.errors.append(error)

        write_telemetry_event(
            "rag_retrieval",
            self.trace_id,
            query_length=len(query),
            duration_ms=round(duration_ms, 2),
            retrieved_count=retrieved_count,
            top_score=round(top_score, 4),
            sources=sources or [],
            success=success,
            error=error,
        )


def extract_llm_usage(message: Any) -> Optional[Dict[str, Any]]:
    """Safely extract token usage metadata from an AIMessage."""
    # Check response_metadata (LangChain ChatGroq format)
    meta = getattr(message, "response_metadata", {})
    if isinstance(meta, dict):
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

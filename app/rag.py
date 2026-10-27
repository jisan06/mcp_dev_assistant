import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAG_DB_PATH = DATA_DIR / "rag.db"

# Excluded directories for RAG indexing
EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    ".idea",
    ".vscode",
    ".cursor",
    ".fleet",
    ".agents",
    ".gemini",
    ".antigravity",
    "scratch",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "logs",
    "data",
    "dist",
    "build",
    "coverage",
    "htmlcov",
    ".tox",
}

# Excluded files for RAG indexing
EXCLUDED_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "audit.jsonl",
    "telemetry.jsonl",
    "results.json",
}

# Sensitive or binary extensions excluded from RAG indexing
SENSITIVE_EXTENSIONS = {
    ".pem",
    ".key",
    ".pkcs12",
    ".pfx",
    ".p12",
    ".kdbx",
    ".crt",
    ".cert",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".db-journal",
    ".db-wal",
    ".jsonl",
    ".pyc",
    ".pyo",
    ".pyd",
}

# File extensions eligible for RAG indexing
INDEXABLE_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".sh",
    ".bat",
    ".ps1",
    ".html",
    ".css",
    ".js",
    ".ts",
}

LANGUAGE_MAP = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
}

# Content patterns to sanitize sensitive credentials before indexing
SECRET_CONTENT_PATTERNS = [
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r"\b(?:gsk|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
        "[REDACTED_TOKEN]",
    ),
    (
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
        "[REDACTED_GITHUB_PAT]",
    ),
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{25,}\b", re.IGNORECASE),
        "Bearer [REDACTED_TOKEN]",
    ),
    (
        re.compile(
            r"(?i)\b(api_key|secret_key|access_token|password|auth_token)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
        ),
        r"\1='[REDACTED_SECRET]'",
    ),
]

STOPWORDS = {
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or", "for",
    "with", "on", "at", "by", "from", "as", "this", "that", "how", "what",
    "where", "why", "who", "which", "can", "you", "tell", "me", "find",
    "show", "does", "get", "all", "project", "code", "file", "please",
}

MAX_FILE_SIZE = 500_000  # 500 KB limit for single file indexing
EMBEDDING_DIM = 256


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def sanitize_code_content(content: str) -> str:
    """Mask any embedded tokens, private keys, or credentials before indexing."""
    sanitized = content
    for pattern, replacement in SECRET_CONTENT_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def is_excluded_path(path: Path, root: Path) -> bool:
    """Check whether a path belongs to an excluded directory or sensitive file."""
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        return True

    # 1. Check directory path components
    for part in relative.parts:
        if part.lower() in EXCLUDED_DIRS:
            return True

    # 2. Check exact excluded files
    if path.name.lower() in EXCLUDED_FILES:
        return True

    # 3. Check sensitive/binary extensions
    if path.suffix.lower() in SENSITIVE_EXTENSIONS:
        return True

    # 4. Exclude environment files (.env.example is the only permitted template)
    if path.name.lower().startswith(".env.") and path.name.lower() != ".env.example":
        return True

    # 5. Exclude secret-naming patterns
    lower_name = path.name.lower()
    if lower_name.startswith("id_") and any(
        k in lower_name for k in ["rsa", "dsa", "ed25519", "ecdsa"]
    ):
        return True
    if any(
        k in lower_name for k in ["secret", "credential", "password", "token"]
    ) and path.suffix.lower() in {".json", ".yaml", ".yml", ".txt", ".env"}:
        return True

    # 6. Validate with centralized SecurityPolicy if available
    try:
        from app.security import default_policy

        default_policy.authorize_file_access(path)
    except Exception:
        return True

    return False


def extract_scope_for_lines(lines: List[str], line_idx: int, file_ext: str) -> str:
    """Find the most immediate enclosing function, class, or heading for a given line."""
    if file_ext == ".py":
        for i in range(line_idx, -1, -1):
            line = lines[i].strip()
            if line.startswith("class ") or line.startswith("def ") or line.startswith("async def "):
                match = re.match(r"(?:async\s+)?(?:def|class)\s+([A-Za-z0-9_]+)", line)
                if match:
                    prefix = "class" if "class" in line else "def"
                    return f"{prefix} {match.group(1)}"
    elif file_ext in {".md", ".markdown"}:
        for i in range(line_idx, -1, -1):
            line = lines[i].strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
    return ""


def chunk_file_content(
    file_path_str: str,
    content: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> List[Tuple[int, int, str, Dict[str, Any]]]:
    """Split file content into overlapping chunks with line citations and rich metadata."""
    sanitized = sanitize_code_content(content)
    lines = sanitized.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    ext = Path(file_path_str).suffix.lower()
    language = LANGUAGE_MAP.get(ext, "text")

    raw_chunks: List[Tuple[int, int, str, Dict[str, Any]]] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < total_lines:
        end = min(total_lines, start + chunk_size)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()

        non_empty = [l for l in chunk_lines if l.strip()]
        if len(non_empty) >= 1 and len(chunk_text) >= 15:
            scope = extract_scope_for_lines(lines, start, ext)
            meta: Dict[str, Any] = {
                "file_type": ext,
                "language": language,
                "scope": scope,
                "start_line": start + 1,
                "end_line": end,
                "line_count": end - start,
                "char_count": len(chunk_text),
            }
            raw_chunks.append((start + 1, end, chunk_text, meta))

        if end >= total_lines:
            break
        start += step

    total_c = len(raw_chunks)
    for idx, (s, e, c, m) in enumerate(raw_chunks):
        m["chunk_index"] = idx
        m["total_chunks"] = total_c

    return raw_chunks


class LocalEmbedder:
    """Free, locally runnable deterministic semantic embedding model.

    Uses normalized subword hashing and character-trigram bag-of-words
    with positional weighting to produce a 256-dimensional dense vector.
    Runs 100% locally with zero external API calls, zero cost, and no downloads.
    If fastembed is installed, it can optionally use fastembed.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._fastembed_model = None

        try:
            from fastembed import TextEmbedding

            self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            self._fastembed_model = None

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense normalized float embedding for a single text."""
        if self._fastembed_model is not None:
            try:
                embeddings = list(self._fastembed_model.embed([text]))
                vec = embeddings[0].tolist()
                return self._normalize(vec)
            except Exception:
                pass

        return self._local_hash_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of texts."""
        if not texts:
            return []
        if self._fastembed_model is not None:
            try:
                embeddings = list(self._fastembed_model.embed(texts))
                return [self._normalize(e.tolist()) for e in embeddings]
            except Exception:
                pass

        return [self._local_hash_embedding(t) for t in texts]

    def _local_hash_embedding(self, text: str) -> List[float]:
        """Generate dense semantic embedding using multi-scale subword and token hashing."""
        vec = [0.0] * self.dim
        if not text:
            return vec

        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vec

        total_tokens = len(tokens)

        for idx, token in enumerate(tokens):
            weight = 1.0 + 0.1 * math.log(1.0 + idx / max(total_tokens, 1))

            # Full token hash
            h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            pos1 = h1 % self.dim
            sign1 = 1.0 if ((h1 >> 8) & 1) else -1.0
            vec[pos1] += sign1 * weight * 1.5

            # Secondary token hash for dispersion
            h2 = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
            pos2 = (h2 >> 4) % self.dim
            sign2 = 1.0 if ((h2 >> 12) & 1) else -1.0
            vec[pos2] += sign2 * weight * 0.8

            # Character 3-grams for code subwords/camelCase/snake_case sensitivity
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    trigram = token[i : i + 3]
                    ht = int(hashlib.md5(trigram.encode("utf-8")).hexdigest()[:8], 16)
                    post = ht % self.dim
                    signt = 1.0 if ((ht >> 3) & 1) else -1.0
                    vec[post] += signt * 0.4

        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            return [v / norm for v in vec]
        return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    if len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def extract_query_keywords(query: str) -> Set[str]:
    """Extract code identifiers, words, and technical terms from query."""
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", query)
    keywords: Set[str] = set()
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if len(tok) >= 3 and tok_lower not in STOPWORDS:
            keywords.add(tok_lower)
            subparts = re.findall(r"[a-z]+|[A-Z][a-z]+|[0-9]+", tok)
            for sp in subparts:
                sp_lower = sp.lower()
                if len(sp) >= 3 and sp_lower not in STOPWORDS:
                    keywords.add(sp_lower)
    return keywords


def compute_keyword_score(
    keywords: Set[str],
    chunk_content: str,
    file_path: str,
    scope: str,
) -> float:
    """Calculate exact token matching score between query keywords and chunk text/path/scope."""
    if not keywords:
        return 0.0

    content_lower = chunk_content.lower()
    path_lower = file_path.lower()
    scope_lower = scope.lower() if scope else ""

    matched = 0
    bonus = 0.0

    for kw in keywords:
        in_content = kw in content_lower
        in_path = kw in path_lower
        in_scope = kw in scope_lower

        if in_content or in_path or in_scope:
            matched += 1
            if in_path:
                bonus += 0.25
            if in_scope:
                bonus += 0.25
            if re.search(r"\b" + re.escape(kw) + r"\b", content_lower):
                bonus += 0.15

    overlap_ratio = matched / len(keywords)
    return min(1.0, overlap_ratio * 0.7 + bonus)


class RAGDatabase:
    """SQLite-based vector store and incremental index manager with hybrid retrieval."""

    def __init__(self, db_path: Path = RAG_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = LocalEmbedder()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_db(self) -> None:
        """Initialize database tables and run non-destructive schema migrations."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_size INTEGER,
                    mtime REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (file_path) REFERENCES indexed_files (file_path) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks (file_path)"
            )

            # Check and apply migrations if table already existed without new columns
            file_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(indexed_files)").fetchall()
            }
            if "file_size" not in file_cols:
                conn.execute("ALTER TABLE indexed_files ADD COLUMN file_size INTEGER")
            if "mtime" not in file_cols:
                conn.execute("ALTER TABLE indexed_files ADD COLUMN mtime REAL")

            chunk_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "metadata" not in chunk_cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN metadata TEXT")

            conn.commit()

    def compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file for incremental indexing."""
        sha = hashlib.sha256()
        try:
            with path.open("rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
        except OSError:
            return ""
        return sha.hexdigest()

    def get_indexed_files(self) -> Dict[str, str]:
        """Return dict of {file_path: file_hash} for currently indexed files."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT file_path, file_hash FROM indexed_files"
            ).fetchall()
            return {row["file_path"]: row["file_hash"] for row in rows}

    def get_indexed_files_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return full metadata for indexed files (hash, size, mtime, chunk_count)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT file_path, file_hash, chunk_count, file_size, mtime FROM indexed_files"
            ).fetchall()
            return {
                row["file_path"]: {
                    "file_hash": row["file_hash"],
                    "chunk_count": row["chunk_count"],
                    "file_size": row["file_size"],
                    "mtime": row["mtime"],
                }
                for row in rows
            }

    def remove_file(self, file_path_str: str) -> None:
        """Remove a file and its chunks from the vector database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path_str,))
            conn.execute(
                "DELETE FROM indexed_files WHERE file_path = ?", (file_path_str,)
            )
            conn.commit()

    def index_codebase(
        self,
        root_dir: Path = PROJECT_ROOT,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Scan codebase, compute incremental diffs, chunk and index modified files."""
        indexed_before = self.get_indexed_files_metadata()
        current_files: Dict[str, Path] = {}
        errors: List[str] = []

        # 1. Discover all candidate files adhering to security & exclusion rules
        for p in root_dir.rglob("*"):
            if not p.is_file():
                continue
            if is_excluded_path(p, root_dir):
                continue
            if p.suffix.lower() not in INDEXABLE_EXTENSIONS:
                continue

            try:
                st = p.stat()
                if st.st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            rel_str = str(p.relative_to(root_dir)).replace("\\", "/")
            current_files[rel_str] = p

        added = 0
        updated = 0
        skipped = 0
        deleted = 0
        total_chunks_added = 0

        # 2. Delete entries for files that no longer exist
        for old_file in list(indexed_before.keys()):
            if old_file not in current_files:
                self.remove_file(old_file)
                deleted += 1

        # 3. Process new and modified files incrementally
        for rel_path, file_obj in current_files.items():
            try:
                st = file_obj.stat()
                curr_mtime = st.st_mtime
                curr_size = st.st_size
            except OSError as exc:
                errors.append(f"Stat error on {rel_path}: {exc}")
                continue

            prev = indexed_before.get(rel_path)

            # Fast incremental check: if mtime and size match, skip without disk read
            if not force and prev is not None:
                if prev.get("mtime") == curr_mtime and prev.get("file_size") == curr_size:
                    skipped += 1
                    continue

                # Secondary check: compare hash if mtime was touched but content might be unchanged
                curr_hash = self.compute_file_hash(file_obj)
                if curr_hash and curr_hash == prev.get("file_hash"):
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE indexed_files SET mtime = ?, file_size = ? WHERE file_path = ?",
                            (curr_mtime, curr_size, rel_path),
                        )
                        conn.commit()
                    skipped += 1
                    continue
            else:
                curr_hash = self.compute_file_hash(file_obj)

            if not curr_hash:
                continue

            # Read content safely
            try:
                content = file_obj.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                errors.append(f"Read error on {rel_path}: {exc}")
                continue

            # Split into chunks with metadata
            raw_chunks = chunk_file_content(rel_path, content)
            if not raw_chunks:
                self.remove_file(rel_path)
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO indexed_files
                        (file_path, file_hash, chunk_count, updated_at, file_size, mtime)
                        VALUES (?, ?, 0, ?, ?, ?)
                        """,
                        (
                            rel_path,
                            curr_hash,
                            datetime.now(timezone.utc).isoformat(),
                            curr_size,
                            curr_mtime,
                        ),
                    )
                    conn.commit()
                continue

            # Generate embeddings
            chunk_texts = [
                f"File: {rel_path} [{c[3].get('scope', '')}]\n{c[2]}" if c[3].get("scope") else c[2]
                for c in raw_chunks
            ]
            embeddings = self.embedder.embed_batch(chunk_texts)

            # Store in SQLite
            with self._get_connection() as conn:
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))

                for (s_line, e_line, c_text, meta), emb in zip(raw_chunks, embeddings):
                    chunk_id = f"{rel_path}:{s_line}-{e_line}"
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chunks
                        (chunk_id, file_path, start_line, end_line, content, embedding, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            rel_path,
                            s_line,
                            e_line,
                            c_text,
                            json.dumps(emb),
                            json.dumps(meta),
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                    (file_path, file_hash, chunk_count, updated_at, file_size, mtime)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel_path,
                        curr_hash,
                        len(raw_chunks),
                        datetime.now(timezone.utc).isoformat(),
                        curr_size,
                        curr_mtime,
                    ),
                )
                conn.commit()

            total_chunks_added += len(raw_chunks)
            if rel_path in indexed_before:
                updated += 1
            else:
                added += 1

        return {
            "status": "success",
            "added_files": added,
            "updated_files": updated,
            "skipped_files": skipped,
            "deleted_files": deleted,
            "total_chunks_indexed": total_chunks_added,
            "total_indexed_files": len(current_files),
            "errors": errors,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.15,
        file_path_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search indexed chunks using hybrid dense semantic and lexical keyword matching."""
        query = query.strip()
        if not query:
            return []

        query_vec = self.embedder.embed_text(query)
        keywords = extract_query_keywords(query)

        with self._get_connection() as conn:
            query_sql = "SELECT chunk_id, file_path, start_line, end_line, content, embedding, metadata FROM chunks"
            params: List[Any] = []
            if file_path_filter:
                query_sql += " WHERE file_path LIKE ?"
                params.append(f"%{file_path_filter}%")

            rows = conn.execute(query_sql, params).fetchall()
            if not rows:
                return []

            scored: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    emb = json.loads(row["embedding"])
                except Exception:
                    continue

                meta: Dict[str, Any] = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except Exception:
                        meta = {}

                dense_sim = cosine_similarity(query_vec, emb)
                scope = meta.get("scope", "")
                kw_score = compute_keyword_score(
                    keywords, row["content"], row["file_path"], scope
                )

                if keywords:
                    score = 0.65 * dense_sim + 0.35 * kw_score
                else:
                    score = dense_sim

                if score >= min_similarity:
                    scored.append(
                        {
                            "file_path": row["file_path"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "score": round(score, 4),
                            "content": row["content"],
                            "scope": scope,
                            "language": meta.get("language", ""),
                            "metadata": meta,
                        }
                    )

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[: max(1, top_k)]


# Singleton instance for server & agent use
_rag_db: Optional[RAGDatabase] = None


def get_rag_database() -> RAGDatabase:
    """Retrieve or initialize the RAG database instance."""
    global _rag_db
    if _rag_db is None:
        _rag_db = RAGDatabase()
    return _rag_db

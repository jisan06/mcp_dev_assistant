import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "logs",
    "data",
    "dist",
    "build",
}

# Excluded files for RAG indexing
EXCLUDED_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "audit.jsonl",
}

# File extensions indexed for RAG
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

        # Tokenize by word boundaries and identifiers
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


def is_excluded_path(path: Path, root: Path) -> bool:
    """Check whether a path belongs to an excluded directory or file."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True

    for part in relative.parts:
        if part.lower() in EXCLUDED_DIRS:
            return True

    if path.name.lower() in EXCLUDED_FILES:
        return True

    if path.name.lower().startswith(".env.") and path.name.lower() != ".env.example":
        return True

    # Exclude SQLite and log files
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".jsonl"}:
        return True

    return False


def chunk_file_content(
    file_path_str: str,
    content: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> List[Tuple[int, int, str]]:
    """Split file content into overlapping chunks with precise line numbers."""
    lines = content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < total_lines:
        end = min(total_lines, start + chunk_size)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()

        if chunk_text:
            # 1-indexed line numbers
            chunks.append((start + 1, end, chunk_text))

        if end >= total_lines:
            break
        start += step

    return chunks


class RAGDatabase:
    """SQLite-based vector store and incremental index manager."""

    def __init__(self, db_path: Path = RAG_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = LocalEmbedder()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
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
                    FOREIGN KEY (file_path) REFERENCES indexed_files (file_path) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks (file_path)"
            )
            conn.commit()

    def compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file for incremental indexing."""
        sha = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def get_indexed_files(self) -> dict:
        """Return dict of {file_path: file_hash} for currently indexed files."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT file_path, file_hash FROM indexed_files").fetchall()
            return {row["file_path"]: row["file_hash"] for row in rows}

    def remove_file(self, file_path_str: str) -> None:
        """Remove a file and its chunks from the vector database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path_str,))
            conn.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path_str,))
            conn.commit()

    def index_codebase(
        self,
        root_dir: Path = PROJECT_ROOT,
        force: bool = False,
    ) -> dict:
        """Scan codebase, compute incremental diffs, chunk and index modified files."""
        indexed_before = self.get_indexed_files()
        current_files = {}

        # 1. Discover all candidate files
        for p in root_dir.rglob("*"):
            if not p.is_file():
                continue
            if is_excluded_path(p, root_dir):
                continue
            if p.suffix.lower() not in INDEXABLE_EXTENSIONS:
                continue
            if p.stat().st_size > MAX_FILE_SIZE:
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

        # 3. Process new and modified files
        for rel_path, file_obj in current_files.items():
            current_hash = self.compute_file_hash(file_obj)

            if not force and rel_path in indexed_before and indexed_before[rel_path] == current_hash:
                skipped += 1
                continue

            # Read content
            try:
                content = file_obj.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Split into chunks
            raw_chunks = chunk_file_content(rel_path, content)
            if not raw_chunks:
                # Still record file so it isn't repeatedly reprocessed
                self.remove_file(rel_path)
                with self._get_connection() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO indexed_files VALUES (?, ?, 0, ?)",
                        (rel_path, current_hash, datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                continue

            # Generate embeddings
            chunk_texts = [c[2] for c in raw_chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)

            # Store in SQLite
            with self._get_connection() as conn:
                # Remove existing chunks for this file
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))

                for (s_line, e_line, c_text), emb in zip(raw_chunks, embeddings):
                    chunk_id = f"{rel_path}:{s_line}-{e_line}"
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chunks
                        (chunk_id, file_path, start_line, end_line, content, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            rel_path,
                            s_line,
                            e_line,
                            c_text,
                            json.dumps(emb),
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_files
                    (file_path, file_hash, chunk_count, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        rel_path,
                        current_hash,
                        len(raw_chunks),
                        datetime.now(timezone.utc).isoformat(),
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
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> List[dict]:
        """Search indexed chunks using cosine similarity."""
        query = query.strip()
        if not query:
            return []

        query_vec = self.embedder.embed_text(query)

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, file_path, start_line, end_line, content, embedding FROM chunks"
            ).fetchall()

            scored = []
            for row in rows:
                try:
                    emb = json.loads(row["embedding"])
                except Exception:
                    continue

                score = cosine_similarity(query_vec, emb)
                if score >= min_similarity:
                    scored.append({
                        "file_path": row["file_path"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "score": round(score, 4),
                        "content": row["content"],
                    })

            # Sort by similarity score descending
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]


# Singleton instance for server & agent use
_rag_db: Optional[RAGDatabase] = None


def get_rag_database() -> RAGDatabase:
    """Retrieve or initialize the RAG database instance."""
    global _rag_db
    if _rag_db is None:
        _rag_db = RAGDatabase()
    return _rag_db

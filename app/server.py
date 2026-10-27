
from pathlib import Path
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Developer Assistant")

REPOSITORY_ROOT = Path(r"F:\Python\dev_assistant").resolve()

# Directories excluded from file listing and reading
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
}

# Files excluded from file listing and reading
EXCLUDED_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
}


def is_excluded(path: Path) -> bool:
    """Check whether a path belongs to an excluded directory or file."""
    relative = path.relative_to(REPOSITORY_ROOT)

    if any(part.lower() in EXCLUDED_DIRS for part in relative.parts):
        return True

    if path.name.lower() in EXCLUDED_FILES:
        return True

    # Exclude environment-variable files such as .env.example is allowed.
    # Other .env.* files are excluded by default.
    if path.name.lower().startswith(".env."):
        return path.name.lower() != ".env.example"

    return False


import sys
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.rag import get_rag_database
from app.security import default_policy, Permission


@mcp.tool()
def list_files() -> list[str]:
    """List project files, excluding IDE and generated development files."""
    if not default_policy.is_permission_enabled(Permission.FILE_LIST):
        raise PermissionError("Access denied: file:list permission is disabled by policy.")

    files = []

    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file():
            continue

        try:
            default_policy.authorize_file_access(path)
        except PermissionError:
            continue

        if is_excluded(path):
            continue

        files.append(str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"))

    return sorted(files)


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read a UTF-8 text file inside the project, excluding development files."""
    target = REPOSITORY_ROOT / file_path

    # Authorize against path traversal and sensitive file access
    authorized_path = default_policy.authorize_file_access(target)

    if not authorized_path.is_file():
        raise FileNotFoundError("File not found.")

    if authorized_path.stat().st_size > 200_000:
        raise ValueError("File exceeds the 200 KB limit.")

    return authorized_path.read_text(encoding="utf-8")


@mcp.tool()
def index_codebase(force: bool = False) -> str:
    """Index or incrementally re-index local project source files into the vector database.

    Args:
        force: If True, re-index all files regardless of modification status.
    """
    if not default_policy.is_permission_enabled(Permission.FILE_INDEX):
        raise PermissionError("Access denied: file:index permission is disabled by policy.")

    rag = get_rag_database()
    stats = rag.index_codebase(root_dir=REPOSITORY_ROOT, force=force)
    return json.dumps(stats, indent=2)


@mcp.tool()
def search_codebase(query: str, top_k: int = 5, min_similarity: float = 0.15) -> str:
    """Semantically search indexed project files and documentation.

    Returns relevant code chunks with exact file paths, line numbers, and citations.

    Args:
        query: The semantic search query describing the code or feature.
        top_k: Number of relevant chunks to retrieve (default 5).
        min_similarity: Minimum similarity score threshold between 0.0 and 1.0 (default 0.15).
    """
    if not default_policy.is_permission_enabled(Permission.FILE_SEARCH):
        raise PermissionError("Access denied: file:search permission is disabled by policy.")

    rag = get_rag_database()
    results = rag.search(query=query, top_k=top_k, min_similarity=min_similarity)
    if not results:
        indexed_count = len(rag.get_indexed_files())
        if indexed_count == 0:
            return (
                "No indexed documents found in vector store. "
                "Run index_codebase() to index project files."
            )
        return (
            f"No relevant code chunks found for query '{query}' "
            f"(threshold {min_similarity}). Try rephrasing or running index_codebase."
        )

    formatted = [f"Found {len(results)} relevant code chunk(s) for query: '{query}'\n"]
    for idx, item in enumerate(results, 1):
        scope_info = f" | Scope: {item['scope']}" if item.get("scope") else ""
        formatted.append(
            f"--- Result {idx} | File: {item['file_path']} (Lines {item['start_line']}-{item['end_line']}) | Similarity: {item['score']}{scope_info} ---\n"
            f"Citation: {item['file_path']}:{item['start_line']}-{item['end_line']}\n"
            f"{item['content']}\n"
        )

    return "\n".join(formatted)


if __name__ == "__main__":
    mcp.run(transport="stdio")


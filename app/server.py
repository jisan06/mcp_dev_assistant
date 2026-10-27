
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


@mcp.tool()
def list_files() -> list[str]:
    """List project files, excluding IDE and generated development files."""
    files = []

    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        files.append(str(path.relative_to(REPOSITORY_ROOT)))

    return sorted(files)


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read a UTF-8 text file inside the project, excluding development files."""
    root = REPOSITORY_ROOT.resolve()
    target = (root / file_path).resolve()

    # Prevent directory traversal outside the project.
    if not target.is_relative_to(root):
        raise ValueError("Access denied: file is outside the project.")

    if is_excluded(target):
        raise ValueError("Access denied: this file or directory is excluded.")

    if not target.is_file():
        raise FileNotFoundError("File not found.")

    if target.stat().st_size > 200_000:
        raise ValueError("File exceeds the 200 KB limit.")

    return target.read_text(encoding="utf-8")


import sys
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.rag import get_rag_database


@mcp.tool()
def index_codebase(force: bool = False) -> str:
    """Index or incrementally re-index local project source files into the vector database.

    Args:
        force: If True, re-index all files regardless of modification status.
    """
    rag = get_rag_database()
    stats = rag.index_codebase(root_dir=REPOSITORY_ROOT, force=force)
    return json.dumps(stats, indent=2)


@mcp.tool()
def search_codebase(query: str, top_k: int = 5) -> str:
    """Semantically search indexed project files and documentation.

    Returns relevant code chunks with exact file paths and line numbers.

    Args:
        query: The semantic search query describing the code or feature.
        top_k: Number of relevant chunks to retrieve (default 5).
    """
    rag = get_rag_database()
    results = rag.search(query=query, top_k=top_k)
    if not results:
        return "No relevant code chunks found. Ensure the codebase is indexed using index_codebase."

    formatted = []
    for idx, item in enumerate(results, 1):
        formatted.append(
            f"--- Result {idx} | File: {item['file_path']} (Lines {item['start_line']}-{item['end_line']}) | Similarity: {item['score']} ---\n"
            f"{item['content']}\n"
        )

    return "\n".join(formatted)


if __name__ == "__main__":
    mcp.run(transport="stdio")


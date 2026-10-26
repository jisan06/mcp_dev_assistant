
from pathlib import Path
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Developer Assistant")

REPOSITORY_ROOT = Path(r"F:\Python\dev_assistant").resolve()


@mcp.tool()
def list_files() -> list[str]:
    """List project files, excluding virtual environments and Git internals."""
    files = []

    for path in REPOSITORY_ROOT.rglob("*"):
        relative = path.relative_to(REPOSITORY_ROOT)

        if any(
            part.lower() in {".venv", ".git", "__pycache__"}
            for part in relative.parts
        ):
            continue

        if path.is_file():
            files.append(str(relative))

    return sorted(files)


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read a UTF-8 text file inside the project directory."""
    root = REPOSITORY_ROOT.resolve()
    target = (root / file_path).resolve()

    if not target.is_relative_to(root):
        raise ValueError("Access denied: file is outside the project.")

    if not target.is_file():
        raise FileNotFoundError("File not found.")

    if target.stat().st_size > 200_000:
        raise ValueError("File exceeds the 200 KB limit.")

    return target.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")

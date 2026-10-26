
import asyncio
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILE = PROJECT_ROOT / "app" / "server.py"

PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")


async def main():
    server = StdioServerParameters(
        command=PYTHON,
        args=[str(SERVER_FILE)],
    )

    async with Client(server) as client:
        # Discover available tools
        result = await client.list_tools()

        print("\nAvailable MCP tools:")
        for tool in result.tools:
            print(f"- {tool.name}: {tool.description}")

        # Call list_files
        print("\n--- Calling list_files ---")
        result = await client.call_tool("list_files", {})
        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)

        # Call read_file
        print("\n--- Calling read_file ---")
        result = await client.call_tool(
            "read_file",
            {"file_path": "app/server.py"},
        )
        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())

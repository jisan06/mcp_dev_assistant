
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILE = PROJECT_ROOT / "app" / "github_server.py"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

load_dotenv(PROJECT_ROOT / ".env")

GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")


async def main():
    if not GITHUB_OWNER or not GITHUB_REPO:
        raise RuntimeError(
            "Set GITHUB_OWNER and GITHUB_REPO in your .env file."
        )

    params = StdioServerParameters(
        command=str(PYTHON),
        args=[str(SERVER_FILE)],
    )

    async with Client(params) as client:
        # Discover available tools
        result = await client.list_tools()

        print("Available GitHub MCP tools:")
        for tool in result.tools:
            print(f"- {tool.name}: {tool.description}")

        # List accessible repositories
        print("\n--- Repositories ---")
        result = await client.call_tool("list_repositories", {})

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)

        # List issues from the default repository
        print(
            f"\n--- Open issues: {GITHUB_OWNER}/{GITHUB_REPO} ---"
        )

        result = await client.call_tool(
            "list_issues",
            {
                "owner": GITHUB_OWNER,
                "repo": GITHUB_REPO,
                "state": "open",
            },
        )

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)


        # Get a specific issue
        issue_number = 1

        print(f"\n--- Issue #{issue_number} ---")

        result = await client.call_tool(
            "get_issue",
            {
                "owner": GITHUB_OWNER,
                "repo": GITHUB_REPO,
                "issue_number": issue_number,
            },
        )

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)

if __name__ == "__main__":
    asyncio.run(main())

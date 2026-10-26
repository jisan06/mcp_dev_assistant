
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILE = PROJECT_ROOT / "app" / "server.py"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

load_dotenv(PROJECT_ROOT / ".env")


async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the .env file.")

    server_params = StdioServerParameters(
        command=str(PYTHON),
        args=[str(SERVER_FILE)],
    )

    async with Client(server_params) as client:

        @tool
        async def list_files() -> str:
            """List files in the developer assistant project."""
            result = await client.call_tool("list_files", {})

            if result.is_error:
                return "Tool error: " + str(result.content)

            return "\n".join(
                item.text
                for item in result.content
                if hasattr(item, "text")
            )

        @tool
        async def read_file(file_path: str) -> str:
            """Read a UTF-8 text file within the project. Provide a relative path."""
            result = await client.call_tool(
                "read_file",
                {"file_path": file_path},
            )

            if result.is_error:
                return "Tool error: " + str(result.content)

            return "\n".join(
                item.text
                for item in result.content
                if hasattr(item, "text")
            )

        model = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            temperature=0,
        )

        agent = create_react_agent(
            model,
            tools=[list_files, read_file],
            prompt=(
                "You are a developer assistant. "
                "Use the available tools when you need project files. "
                "Never claim to have read a file unless you used the tool. "
                "Do not modify files."
            ),
        )

        print("MCP Developer Assistant ready.")
        print("Type 'exit' to quit.\n")

        while True:
            question = await asyncio.to_thread(
                input, "You: "
            )

            if question.strip().lower() in {"exit", "quit"}:
                break

            if not question.strip():
                continue

            try:
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]}
                )
                print("\nAssistant:", response["messages"][-1].content, "\n")

            except Exception as exc:
                print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())

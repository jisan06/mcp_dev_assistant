
import json
import os
import re

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer


mcp = MCPServer("GitHub Assistant")

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def validate_repository(owner: str, repo: str) -> None:
    """Validate GitHub owner and repository names."""
    pattern = re.compile(r"^[A-Za-z0-9_.-]+$")

    if not pattern.fullmatch(owner) or not pattern.fullmatch(repo):
        raise ValueError("Invalid GitHub owner or repository name.")


async def github_get(endpoint: str, params: dict | None = None):
    """Make a read-only request to the GitHub API."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is missing from .env.")

    async with httpx.AsyncClient(timeout=20.0) as http:
        response = await http.get(
            f"{GITHUB_API}{endpoint}",
            headers=HEADERS,
            params=params,
        )

    if response.status_code == 401:
        raise RuntimeError("GitHub authentication failed. Check GITHUB_TOKEN.")

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub access denied or API rate limit reached. "
            "Check token permissions and rate limits."
        )

    if response.status_code == 404:
        raise RuntimeError(
            "GitHub resource not found or the token lacks access."
        )

    response.raise_for_status()
    return response.json()


@mcp.tool()
async def list_repositories() -> str:
    """List repositories accessible to the authenticated GitHub user."""
    try:
        data = await github_get(
            "/user/repos",
            params={
                "visibility": "all",
                "affiliation": "owner,collaborator,organization_member",
                "sort": "updated",
                "per_page": 30,
                "page": 1,
            },
        )

        repositories = [
            {
                "full_name": repo["full_name"],
                "private": repo["private"],
                "description": repo["description"],
                "html_url": repo["html_url"],
            }
            for repo in data
        ]

        return json.dumps(repositories, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def list_issues(
    owner: str,
    repo: str,
    state: str = "open",
) -> str:
    """List issues in a repository. State must be open, closed, or all."""
    try:
        validate_repository(owner, repo)

        if state not in {"open", "closed", "all"}:
            raise ValueError("State must be open, closed, or all.")

        data = await github_get(
            f"/repos/{owner}/{repo}/issues",
            params={
                "state": state,
                "per_page": 30,
                "page": 1,
            },
        )

        # GitHub's issues endpoint also returns pull requests.
        issues = [
            {
                "number": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "html_url": issue["html_url"],
                "user": issue["user"]["login"],
                "created_at": issue["created_at"],
            }
            for issue in data
            if "pull_request" not in issue
        ]

        return json.dumps(issues, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_issue(owner: str, repo: str, issue_number: int) -> str:
    """Get the details of a specific issue in a repository."""
    try:
        validate_repository(owner, repo)

        if issue_number < 1:
            raise ValueError("Issue number must be a positive integer.")

        issue = await github_get(
            f"/repos/{owner}/{repo}/issues/{issue_number}"
        )

        result = {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
            "state": issue["state"],
            "html_url": issue["html_url"],
            "user": issue["user"]["login"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
        }

        if "pull_request" in issue:
            result["note"] = "This item is a pull request, not an issue."

        return json.dumps(result, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    mcp.run(transport="stdio")

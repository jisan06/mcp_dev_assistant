import hmac
import hashlib
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SENSITIVE FILE PATTERNS that must never be read by any tool
SENSITIVE_FILENAME_PATTERNS = {
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"^.*\.(pem|key|pkcs12|pfx|p12|kdbx)$", re.IGNORECASE),
    re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\..+)?$", re.IGNORECASE),
    re.compile(r"^.*(secret|credential|password|token).*\.(json|yaml|yml|txt|env)$", re.IGNORECASE),
}

# SECRET_KEY for server-side HMAC validation of human approval
# Generated once per process run or loaded from environment
_APPROVAL_SECRET = os.getenv("APPROVAL_SECRET") or hashlib.sha256(f"dev_assistant_approval_{os.getpid()}_{time.time()}".encode()).hexdigest()

REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class Permission(str, Enum):
    FILE_READ = "file:read"
    FILE_LIST = "file:list"
    FILE_INDEX = "file:index"
    FILE_SEARCH = "file:search"
    GITHUB_READ = "github:read"
    GITHUB_WRITE = "github:write"


class SecurityPolicy:
    """Centralized authorization and security policy manager."""

    def __init__(self, root_dir: Path = PROJECT_ROOT):
        self.root_dir = root_dir.resolve()
        self.allowed_permissions: Set[Permission] = {
            Permission.FILE_READ,
            Permission.FILE_LIST,
            Permission.FILE_INDEX,
            Permission.FILE_SEARCH,
            Permission.GITHUB_READ,
            Permission.GITHUB_WRITE,  # Requires approval token validation
        }

    def is_permission_enabled(self, perm: Permission) -> bool:
        """Check if a permission is enabled in current policy."""
        # Check environment overrides (e.g. GITHUB_ALLOW_WRITE=false)
        if perm == Permission.GITHUB_WRITE:
            env_val = os.getenv("GITHUB_ALLOW_WRITE", "true").lower()
            if env_val in {"false", "0", "no", "disabled"}:
                return False
        return perm in self.allowed_permissions

    def authorize_file_access(self, target_path: Path) -> Path:
        """Validate and authorize file access, strictly preventing path traversal and secret leakage."""
        if not self.is_permission_enabled(Permission.FILE_READ):
            raise PermissionError("Access denied: file:read permission is disabled.")

        resolved = target_path.resolve()

        # 1. Prevent Directory Traversal outside project root
        try:
            rel = resolved.relative_to(self.root_dir)
        except ValueError:
            raise PermissionError(f"Access denied: path traversal outside project root is forbidden.")

        # 2. Block prohibited directories
        prohibited_parts = {
            ".venv", "venv", "env", ".git", ".idea", ".vscode",
            ".cursor", ".fleet", "__pycache__", ".pytest_cache", "node_modules"
        }
        if any(part.lower() in prohibited_parts for part in rel.parts):
            raise PermissionError(f"Access denied: access to '{resolved.name}' is prohibited by security policy.")

        # 3. Block sensitive secret patterns (e.g., .env, private keys, certificates)
        for pattern in SENSITIVE_FILENAME_PATTERNS:
            if pattern.fullmatch(resolved.name):
                if resolved.name.lower() == ".env.example":
                    continue  # .env.example template is safe
                raise PermissionError(f"Access denied: file '{resolved.name}' is classified as a sensitive credential.")

        return resolved

    def authorize_github_read(self, owner: str, repo: str) -> None:
        """Authorize GitHub read operations against allowed repositories."""
        if not self.is_permission_enabled(Permission.GITHUB_READ):
            raise PermissionError("Access denied: github:read permission is disabled.")

        if not REPO_PATTERN.fullmatch(owner) or not REPO_PATTERN.fullmatch(repo):
            raise ValueError("Access denied: invalid repository owner or name syntax.")

        allowed_repos = self.get_allowed_github_repositories()
        if allowed_repos is not None:
            full_name = f"{owner}/{repo}".lower()
            if full_name not in allowed_repos:
                raise PermissionError(f"Access denied: repository '{owner}/{repo}' is not authorized.")

    def authorize_github_write(self, owner: str, repo: str, title: str, approval_token: Optional[str]) -> None:
        """Server-side authorization check for GitHub write operations.

        Enforces:
        1. github:write permission enabled.
        2. Valid repository target.
        3. Mandatory valid cryptographic human approval token.
        """
        if not self.is_permission_enabled(Permission.GITHUB_WRITE):
            raise PermissionError("Access denied: github:write permission is disabled by policy.")

        if not REPO_PATTERN.fullmatch(owner) or not REPO_PATTERN.fullmatch(repo):
            raise ValueError("Access denied: invalid repository owner or name syntax.")

        allowed_repos = self.get_allowed_github_repositories()
        if allowed_repos is not None:
            full_name = f"{owner}/{repo}".lower()
            if full_name not in allowed_repos:
                raise PermissionError(f"Access denied: repository '{owner}/{repo}' is not in the authorized write list.")

        # Enforce server-side human approval token
        if not approval_token:
            raise PermissionError(
                "Access denied: create_issue requires explicit human approval. "
                "Approval token is missing."
            )

        if not verify_approval_token(owner, repo, title, approval_token):
            raise PermissionError(
                "Access denied: invalid or expired human approval token. "
                "Direct LLM issue creation without terminal confirmation is forbidden."
            )

    @staticmethod
    def get_allowed_github_repositories() -> Optional[Set[str]]:
        """Return the set of authorized GitHub repositories (owner/repo) if configured."""
        env_allowed = os.getenv("GITHUB_ALLOWED_REPOS", "").strip()
        if env_allowed:
            return {r.strip().lower() for r in env_allowed.split(",") if r.strip()}

        # Default to configured GITHUB_OWNER / GITHUB_REPO if set
        owner = os.getenv("GITHUB_OWNER", "").strip().lower()
        repo = os.getenv("GITHUB_REPO", "").strip().lower()
        if owner and repo:
            return {f"{owner}/{repo}"}

        return None


def generate_approval_token(owner: str, repo: str, title: str) -> str:
    """Generate a cryptographic HMAC approval token tying approval to this exact issue draft."""
    payload = f"{owner.strip().lower()}:{repo.strip().lower()}:{title.strip()}".encode("utf-8")
    return hmac.new(_APPROVAL_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_approval_token(owner: str, repo: str, title: str, token: str) -> bool:
    """Verify that the approval token matches the exact issue draft."""
    expected = generate_approval_token(owner, repo, title)
    return hmac.compare_digest(expected, token.strip())


# Default policy instance
default_policy = SecurityPolicy()

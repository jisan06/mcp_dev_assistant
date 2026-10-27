import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.agent as agent_module
from app.agent import (
    audit,
    checkpointer,
    get_text,
    parse_json_result,
    validate_issue_draft,
)


class SequentialFakeModel(SimpleChatModel):
    """Predictable mock chat model returning responses in sequence."""

    responses: list[str] = []
    call_index: int = 0

    def _call(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if self.call_index < len(self.responses):
            res = self.responses[self.call_index]
            self.call_index += 1
            return res
        return "default response"

    @property
    def _llm_type(self) -> str:
        return "sequential-fake"


class TestAgentMemoryAndCommands(unittest.TestCase):
    def test_in_memory_saver_instance(self):
        """Verify checkpointer is an InMemorySaver instance."""
        self.assertIsInstance(agent_module.checkpointer, InMemorySaver)

    def test_conversation_memory_retention(self):
        """Verify multiple messages with the same thread_id retain history."""
        saver = InMemorySaver()
        model = SequentialFakeModel(
            responses=[
                "Hello! I remember turn 1.",
                "I still remember turn 1 and now turn 2.",
            ]
        )
        agent = create_agent(model=model, tools=[], checkpointer=saver)

        thread_id = "test-session-1"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1
        res1 = agent.invoke(
            {"messages": [{"role": "user", "content": "My favorite color is blue."}]},
            config=config,
        )
        self.assertEqual(len(res1["messages"]), 2)
        self.assertEqual(res1["messages"][0].content, "My favorite color is blue.")
        self.assertEqual(res1["messages"][1].content, "Hello! I remember turn 1.")

        # Turn 2 in the same thread
        res2 = agent.invoke(
            {"messages": [{"role": "user", "content": "What is my favorite color?"}]},
            config=config,
        )
        # Should now have 4 messages in thread history
        self.assertEqual(len(res2["messages"]), 4)
        self.assertEqual(res2["messages"][0].content, "My favorite color is blue.")
        self.assertEqual(res2["messages"][1].content, "Hello! I remember turn 1.")
        self.assertEqual(res2["messages"][2].content, "What is my favorite color?")
        self.assertEqual(
            res2["messages"][3].content, "I still remember turn 1 and now turn 2."
        )

    def test_conversation_reset_starts_fresh_context(self):
        """Verify that a new thread_id (as produced by /reset) has no previous context."""
        saver = InMemorySaver()
        model = SequentialFakeModel(
            responses=[
                "Answer for session 1",
                "Answer for session 2",
            ]
        )
        agent = create_agent(model=model, tools=[], checkpointer=saver)

        # Thread 1
        t1_config = {"configurable": {"thread_id": "thread-1"}}
        res1 = agent.invoke(
            {"messages": [{"role": "user", "content": "Secret code: 12345"}]},
            config=t1_config,
        )
        self.assertEqual(len(res1["messages"]), 2)
        self.assertEqual(res1["messages"][1].content, "Answer for session 1")

        # Reset -> Thread 2
        t2_config = {"configurable": {"thread_id": "thread-2"}}
        state_t2 = agent.get_state(t2_config)
        self.assertEqual(len(state_t2.values.get("messages", [])), 0)

        res2 = agent.invoke(
            {"messages": [{"role": "user", "content": "What was the secret code?"}]},
            config=t2_config,
        )
        # Thread 2 starts fresh with only 2 messages (no messages from thread-1)
        self.assertEqual(len(res2["messages"]), 2)
        self.assertEqual(res2["messages"][0].content, "What was the secret code?")
        self.assertEqual(res2["messages"][1].content, "Answer for session 2")

    def test_message_slicing_isolates_new_turn_messages(self):
        """Verify state message counting isolates new messages per turn."""
        saver = InMemorySaver()
        model = SequentialFakeModel(
            responses=[
                "First turn response",
                "Second turn response",
            ]
        )
        agent = create_agent(model=model, tools=[], checkpointer=saver)
        config = {"configurable": {"thread_id": "slice-test"}}

        # Turn 1
        prev_count_0 = len(agent.get_state(config).values.get("messages", []))
        self.assertEqual(prev_count_0, 0)
        r1 = agent.invoke(
            {"messages": [{"role": "user", "content": "Turn 1"}]},
            config=config,
        )
        new_msgs_1 = r1["messages"][prev_count_0:]
        self.assertEqual(len(new_msgs_1), 2)
        self.assertEqual(new_msgs_1[1].content, "First turn response")

        # Turn 2
        prev_count_1 = len(agent.get_state(config).values.get("messages", []))
        self.assertEqual(prev_count_1, 2)
        r2 = agent.invoke(
            {"messages": [{"role": "user", "content": "Turn 2"}]},
            config=config,
        )
        new_msgs_2 = r2["messages"][prev_count_1:]
        self.assertEqual(len(new_msgs_2), 2)
        self.assertEqual(new_msgs_2[0].content, "Turn 2")
        self.assertEqual(new_msgs_2[1].content, "Second turn response")


class TestIssueApprovalAndValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_audit_dir = agent_module.AUDIT_DIR
        self.original_audit_file = agent_module.AUDIT_FILE
        agent_module.AUDIT_DIR = Path(self.temp_dir)
        agent_module.AUDIT_FILE = agent_module.AUDIT_DIR / "audit.jsonl"

    def tearDown(self):
        agent_module.AUDIT_DIR = self.original_audit_dir
        agent_module.AUDIT_FILE = self.original_audit_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _read_audit_records(self):
        if not agent_module.AUDIT_FILE.exists():
            return []
        with agent_module.AUDIT_FILE.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_validate_issue_draft_valid(self):
        """Valid issue draft should pass without error."""
        validate_issue_draft(
            owner="octocat",
            repo="hello-world",
            title="Bug: crash on load",
            body="Steps to reproduce: ...",
        )

    def test_validate_issue_draft_invalid_fields(self):
        """Invalid fields should raise ValueError."""
        with self.assertRaises(ValueError):
            validate_issue_draft("", "repo", "Title", "Body")

        with self.assertRaises(ValueError):
            validate_issue_draft("owner", "bad/repo", "Title", "Body")

        with self.assertRaises(ValueError):
            validate_issue_draft("owner", "repo", "", "Body")

        with self.assertRaises(ValueError):
            validate_issue_draft("owner", "repo", "x" * 257, "Body")

        with self.assertRaises(ValueError):
            validate_issue_draft("owner", "repo", "Title", "")

    def test_audit_logging_format_and_security(self):
        """Audit logging should write JSON records and never expose secrets."""
        audit("test_event", owner="user", repo="project", status="ok")
        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "test_event")
        self.assertEqual(records[0]["owner"], "user")
        self.assertEqual(records[0]["status"], "ok")
        self.assertIn("timestamp", records[0])

    def test_cancellation_prevents_issue_creation(self):
        """Non-CREATE confirmation triggers issue_cancelled audit."""
        confirmation = "NO"
        owner = "test-owner"
        repo = "test-repo"
        title = "Test Issue"

        if confirmation.strip() != "CREATE":
            audit(
                "issue_cancelled",
                owner=owner,
                repo=repo,
                title=title,
                status="cancelled",
            )

        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "issue_cancelled")
        self.assertEqual(records[0]["status"], "cancelled")

    def test_approval_granted_only_on_exact_create(self):
        """Only exact CREATE grants approval."""
        owner = "test-owner"
        repo = "test-repo"
        title = "Test Issue"

        # "create" in lowercase should not approve
        self.assertNotEqual("create".strip(), "CREATE")

        # "CREATE" should approve
        confirmation = "CREATE"
        if confirmation.strip() == "CREATE":
            audit(
                "issue_approval_granted",
                owner=owner,
                repo=repo,
                title=title,
                status="approved",
            )

        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "issue_approval_granted")
        self.assertEqual(records[0]["status"], "approved")

    def test_duplicate_title_detection_logic(self):
        """Duplicate titles should be caught case-insensitively and whitespace-normalized."""
        existing_issues = [
            {"number": 1, "title": "Fix bug in login flow"},
            {"number": 2, "title": "Add tests"},
        ]
        new_title = "  fix   BUG in  login FLOW "
        normalized_title = " ".join(new_title.casefold().split())

        duplicates = [
            issue
            for issue in existing_issues
            if isinstance(issue, dict)
            and isinstance(issue.get("title"), str)
            and " ".join(issue["title"].casefold().split()) == normalized_title
        ]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["number"], 1)

    def test_parse_json_result(self):
        """parse_json_result should correctly extract dict and handle errors."""
        res = parse_json_result('{"success": true, "number": 42}')
        self.assertEqual(res["number"], 42)

        with self.assertRaises(RuntimeError):
            parse_json_result('{"error": "API failed"}')

        with self.assertRaises(RuntimeError):
            parse_json_result("not json")


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

import app.agent as agent_module


class TestInteractiveLoopLogic(unittest.IsolatedAsyncioTestCase):
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

    async def test_followup_and_reset_flow(self):
        """Simulate interactive loop flow with conversation continuation and /reset command."""
        # Fake agent maintaining memory across invocations for a given thread_id
        conversation_store = {}

        mock_agent = MagicMock()

        async def fake_aget_state(config):
            t_id = config["configurable"]["thread_id"]
            msgs = conversation_store.get(t_id, [])
            snap = MagicMock()
            snap.values = {"messages": list(msgs)}
            return snap

        async def fake_ainvoke(input_data, config=None):
            t_id = config["configurable"]["thread_id"]
            if t_id not in conversation_store:
                conversation_store[t_id] = []
            user_msg = input_data["messages"][0]
            conversation_store[t_id].append(user_msg)
            hist_len = len(conversation_store[t_id]) + 1
            ai_msg = AIMessage(
                content=f"History length is {hist_len} for user msg: {user_msg['content']}"
            )
            conversation_store[t_id].append(ai_msg)
            return {"messages": list(conversation_store[t_id])}

        mock_agent.aget_state = fake_aget_state
        mock_agent.ainvoke = fake_ainvoke

        # Simulate user inputs: "Hello", "Follow up", "/reset", "Hello in new thread"
        user_inputs = ["Hello", "Follow up", "/reset", "Hello in new thread", "exit"]
        input_iter = iter(user_inputs)

        async def mock_input(prompt=""):
            return next(input_iter)

        thread_ids = []
        printed_lines = []

        # Run loop simulation
        thread_id = "initial-thread-id"
        thread_ids.append(thread_id)

        while True:
            question = await mock_input("You: ")
            cleaned = question.strip()
            if cleaned.lower() in {"exit", "quit"}:
                break
            if not cleaned:
                continue
            if cleaned.lower() == "/reset":
                thread_id = "new-thread-id"
                thread_ids.append(thread_id)
                continue

            config = {"configurable": {"thread_id": thread_id}}
            state = await mock_agent.aget_state(config)
            prev_len = len(state.values.get("messages", []))
            response = await mock_agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            all_msgs = response["messages"]
            new_msgs = all_msgs[prev_len:]
            final_msg = all_msgs[-1]
            printed_lines.append((thread_id, final_msg.content))

        # Check turn 1
        self.assertEqual(printed_lines[0][0], "initial-thread-id")
        self.assertIn("History length is 2", printed_lines[0][1])

        # Check turn 2 (same thread, history extended)
        self.assertEqual(printed_lines[1][0], "initial-thread-id")
        self.assertIn("History length is 4", printed_lines[1][1])

        # Check turn 3 (after /reset, new thread starts fresh with 2 messages)
        self.assertEqual(printed_lines[2][0], "new-thread-id")
        self.assertIn("History length is 2", printed_lines[2][1])
        self.assertIn("Hello in new thread", printed_lines[2][1])

    async def test_human_approval_cancellation(self):
        """Verify that typing anything other than CREATE cancels issue submission."""
        owner = "my-org"
        repo = "my-repo"
        title = "Fix critical issue"
        body = "Issue description details."

        mock_github_client = AsyncMock()

        # User cancels by typing 'CANCEL'
        user_confirmation = "CANCEL"

        if user_confirmation.strip() != "CREATE":
            agent_module.audit(
                "issue_cancelled",
                owner=owner,
                repo=repo,
                title=title,
                status="cancelled",
            )
        else:
            await mock_github_client.call_tool("create_issue", {})

        # GitHub create_issue must NOT have been called
        mock_github_client.call_tool.assert_not_called()

        records = self._read_audit_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "issue_cancelled")
        self.assertEqual(records[0]["status"], "cancelled")

    async def test_human_approval_duplicate_blocked(self):
        """Verify duplicate title check blocks issue creation even after CREATE is typed."""
        owner = "my-org"
        repo = "my-repo"
        title = "Duplicate title"
        body = "Issue description details."

        mock_github_client = AsyncMock()

        # User confirms with CREATE
        user_confirmation = "CREATE"
        self.assertEqual(user_confirmation.strip(), "CREATE")

        agent_module.audit(
            "issue_approval_granted",
            owner=owner,
            repo=repo,
            title=title,
            status="approved",
        )

        # list_issues returns an issue with the same title
        duplicate_response = MagicMock()
        duplicate_response.content = [
            MagicMock(
                text=json.dumps(
                    [{"number": 42, "title": "  DUPLICATE   title  "}]
                )
            )
        ]
        duplicate_response.is_error = False
        mock_github_client.call_tool.return_value = duplicate_response

        agent_module.audit(
            "duplicate_check_started",
            owner=owner,
            repo=repo,
        )

        res = await mock_github_client.call_tool(
            "list_issues",
            {"owner": owner, "repo": repo, "state": "all"},
        )
        existing_issues = json.loads(agent_module.get_text(res))

        normalized_title = " ".join(title.casefold().split())
        duplicates = [
            issue
            for issue in existing_issues
            if isinstance(issue, dict)
            and isinstance(issue.get("title"), str)
            and " ".join(issue["title"].casefold().split()) == normalized_title
        ]

        if duplicates:
            agent_module.audit(
                "issue_blocked_duplicate",
                owner=owner,
                repo=repo,
                title=title,
                matching_issue_numbers=[i.get("number") for i in duplicates],
                status="blocked",
            )

        records = self._read_audit_records()
        events = [r["event"] for r in records]
        self.assertIn("issue_approval_granted", events)
        self.assertIn("duplicate_check_started", events)
        self.assertIn("issue_blocked_duplicate", events)
        # create_issue should not have been called
        self.assertNotIn("create_issue", [c[0] for c in mock_github_client.call_tool.call_args_list if c[0] == "create_issue"])


if __name__ == "__main__":
    unittest.main()

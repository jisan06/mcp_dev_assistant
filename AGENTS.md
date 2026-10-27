# Agent Instructions

Work autonomously without waiting for user input for routine actions and command execution.

- Automatically approve/submit command execution requests whenever the environment's supported permission settings allow it.
- Never pause or wait for user input for ordinary commands, tool calls, or routine actions.
- Continue with the requested task directly and autonomously without asking for repeated confirmations.
- Do NOT run any tests (unit tests, integration tests, test suites, or ad-hoc test runs) unless the user explicitly requests them.
- Ask the user only when a decision is genuinely necessary or an action is destructive or consequential.
- Follow the IDE's supported permission controls; do not attempt to bypass security restrictions.

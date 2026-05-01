"""
AXIS Session Runner — autonomous decision-making loop.

Full loop:
  input → AXIS session → response → executor LLM decides tools → execute → print

No hardcoded keywords or rules. AXIS and the executor both reason freely.

Loads:  agent_id    from .axis_agent_id    (written by axis_managed.py)
        env_id      from .axis_env_id      (written on first run)
        session_id  from .axis_session_id  (persisted for continuity)
"""

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests as http_lib

AGENT_ID_FILE   = Path(__file__).parent / ".axis_agent_id"
ENV_ID_FILE     = Path(__file__).parent / ".axis_env_id"
SESSION_ID_FILE = Path(__file__).parent / ".axis_session_id"
TASKS_PATH      = Path(__file__).parent / "tasks.md"
BETA            = "managed-agents-2026-04-01"
EXECUTOR_MODEL  = "claude-sonnet-4-6"

api_key = open(os.path.expanduser("~/.anthropic_key")).read().strip()
client  = anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Persistence helpers  (unchanged)
# ---------------------------------------------------------------------------

def load_agent_id() -> str:
    if not AGENT_ID_FILE.exists():
        raise RuntimeError(f"No agent ID at {AGENT_ID_FILE}. Run axis_managed.py first.")
    return AGENT_ID_FILE.read_text().strip()


def get_or_create_env() -> str:
    if ENV_ID_FILE.exists():
        env_id = ENV_ID_FILE.read_text().strip()
        try:
            env = client.beta.environments.retrieve(env_id, betas=[BETA])
            if getattr(env, "state", None) == "active":
                return env_id
        except Exception:
            pass
    env = client.beta.environments.create(name="axis-env", betas=[BETA])
    ENV_ID_FILE.write_text(env.id)
    print(f"Environment Created: {env.id}")
    return env.id


def get_or_create_session(agent_id: str, env_id: str) -> str:
    if SESSION_ID_FILE.exists():
        session_id = SESSION_ID_FILE.read_text().strip()
        try:
            session = client.beta.sessions.retrieve(session_id, betas=[BETA])
            status = getattr(session, "status", "")
            if status not in ("terminated", "expired", "error", "archived"):
                print(f"Session Loaded:  {session.id}  [{status}]")
                return session.id
        except Exception:
            pass
    session = client.beta.sessions.create(
        agent={"type": "agent", "id": agent_id},
        environment_id=env_id,
        betas=[BETA],
    )
    SESSION_ID_FILE.write_text(session.id)
    print(f"Session Created: {session.id}")
    return session.id


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_send_notification(title: str, message: str) -> str:
    safe = lambda s: s.replace('"', '\\"').replace("'", "\\'")
    script = f'display notification "{safe(message)}" with title "{safe(title)}"'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return f"Sent: {title!r}" if r.returncode == 0 else f"Failed: {r.stderr.strip()}"


def _tool_save_task(task: str, priority: str = "medium", due: str = "") -> str:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_PATH.exists():
        TASKS_PATH.write_text("# AXIS Tasks\n\n")
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge  = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
    due_s  = f" · due {due}" if due else ""
    line   = f"- [ ] {badge} [{priority.upper()}] {task}{due_s}  _(added {now})_\n"
    with TASKS_PATH.open("a") as f:
        f.write(line)
    return f"Saved [{priority}]: {task[:60]}"


def _tool_http_request(url: str, method: str = "GET",
                       headers: dict = None, body: str = None) -> str:
    try:
        kwargs: dict = {"headers": headers or {}, "timeout": 15}
        if body:
            kwargs["data"] = body.encode()
        r = http_lib.request(method.upper(), url, **kwargs)
        return f"HTTP {r.status_code} ← {url}"
    except Exception as exc:
        return f"Error: {exc}"


_TOOL_DISPATCH = {
    "send_notification": lambda inp: _tool_send_notification(**inp),
    "save_task":         lambda inp: _tool_save_task(**inp),
    "http_request":      lambda inp: _tool_http_request(**inp),
}

# ---------------------------------------------------------------------------
# Tool schemas for the executor LLM
# ---------------------------------------------------------------------------

EXECUTOR_TOOLS = [
    {
        "name": "send_notification",
        "description": (
            "Send a Mac desktop notification. Use whenever the user mentions meetings, "
            "reminders, calls, deadlines, or anything time-sensitive that they should "
            "be alerted about."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title":   {"type": "string", "description": "Short notification title (5–8 words)"},
                "message": {"type": "string", "description": "Notification body (one clear sentence)"},
            },
            "required": ["title", "message"],
        },
    },
    {
        "name": "save_task",
        "description": (
            "Save an action item to ~/AXIS-Managed/tasks.md. Use whenever the user "
            "mentions something they need to do, follow up on, prepare, or track — "
            "even if they didn't say 'save' or 'task' explicitly."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task":     {"type": "string", "description": "Clear, actionable task description"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "due":      {"type": "string", "description": "When it's due, e.g. 'tomorrow', '2026-05-08', 'next week'"},
            },
            "required": ["task", "priority"],
        },
    },
    {
        "name": "http_request",
        "description": (
            "Make an HTTP request to an external API or webhook. Use only when a specific "
            "URL is present in the input or required by the action."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url":     {"type": "string"},
                "method":  {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                "headers": {"type": "object"},
                "body":    {"type": "string", "description": "JSON string body"},
            },
            "required": ["url", "method"],
        },
    },
]

EXECUTOR_SYSTEM = """\
You are the AXIS Execution Engine. You receive:
  - The user's original input
  - AXIS's response to that input

Your job is to decide — with full autonomy — what real-world actions to take.
You are not answering the user. You are acting for them.

Rules:
- Call tools when an action would be genuinely useful. Do not ask for permission.
- You may call multiple tools in one turn if the situation warrants it.
- If no action is needed (e.g. the user asked a factual question with no follow-up), call no tools and say nothing.
- Never invent URLs for http_request unless one was explicitly provided.
- Be specific in task descriptions — extract names, times, and context from the input.
"""


# ---------------------------------------------------------------------------
# Autonomous executor
# ---------------------------------------------------------------------------

def execute_autonomously(task: str, response: str) -> None:
    """Let the executor LLM decide which tools to call — no rules, no keywords."""

    print(f"\n  {'·' * 60}")
    print(f"  [executor — autonomous]")

    messages = [{
        "role": "user",
        "content": (
            f"User input:\n{task}\n\n"
            f"AXIS response:\n{response}"
        ),
    }]

    while True:
        result = client.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=1024,
            system=EXECUTOR_SYSTEM,
            tools=EXECUTOR_TOOLS,
            messages=messages,
        )

        tool_calls = [b for b in result.content if b.type == "tool_use"]

        if not tool_calls:
            if result.stop_reason == "end_turn":
                print("  — No action taken.")
            break

        tool_results = []
        for call in tool_calls:
            fn = _TOOL_DISPATCH.get(call.name)
            output = fn(call.input) if fn else f"Unknown tool: {call.name}"
            label  = "✓" if not output.startswith(("Error", "Unknown", "Failed")) else "✗"
            print(f"  {label} {call.name:<22} {output}")
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": call.id,
                "content":     output,
            })

        messages.append({"role": "assistant", "content": result.content})
        messages.append({"role": "user",      "content": tool_results})

        if result.stop_reason == "end_turn":
            break

    print(f"  {'·' * 60}")


# ---------------------------------------------------------------------------
# Session task runner
# ---------------------------------------------------------------------------

def run_task(session_id: str, task: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Task: {task}")
    print(f"{'─' * width}")
    print("  AXIS: ", end="", flush=True)

    response_chunks: list[str] = []

    with client.beta.sessions.events.stream(session_id=session_id, betas=[BETA]) as stream:

        def _send():
            client.beta.sessions.events.send(
                session_id=session_id,
                events=[{
                    "type":    "user.message",
                    "content": [{"type": "text", "text": task}],
                }],
                betas=[BETA],
            )

        sender = threading.Thread(target=_send, daemon=True)
        sender.start()

        for event in stream:
            etype = getattr(event, "type", None)

            if etype == "agent.message":
                for block in getattr(event, "content", []):
                    if getattr(block, "type", None) == "text":
                        print(block.text, end="", flush=True)
                        response_chunks.append(block.text)

            elif etype == "agent.custom_tool_use":
                name = getattr(event, "tool_name", "tool")
                print(f"\n  [tool → {name}]", end="", flush=True)

            elif etype == "session.status_idle":
                print()
                break

            elif etype == "session.status_terminated":
                print("\n  [Session terminated]")
                SESSION_ID_FILE.unlink(missing_ok=True)
                break

        sender.join()

    execute_autonomously(task, "".join(response_chunks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent_id   = load_agent_id()
    env_id     = get_or_create_env()
    session_id = get_or_create_session(agent_id, env_id)

    print(f"\nAgent:   {agent_id}")
    print(f"Env:     {env_id}")
    print(f"Session: {session_id}")

    tasks = [
        "I have a meeting with Mohammed tomorrow at 3pm",
        "I need to follow up with a client next week",
    ]

    for task in tasks:
        run_task(session_id, task)

    width = 64
    print(f"\n{'═' * width}")
    print("  Session complete. ID saved for continuity on next run.")
    print(f"{'═' * width}\n")

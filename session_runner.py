"""
AXIS True Autonomous Agent.

Loop:
  input
    → AXIS session (response + perspective)
    → Planner (decompose into sub-tasks)
    → Executor (run each sub-task autonomously)
    → Feedback (results → AXIS session → decide if more needed)
    → Suggestions (AXIS proposes next logical actions)
    → Memory (persist every decision + result)

Persistence:
  .axis_agent_id   agent (written by axis_managed.py)
  .axis_env_id     environment
  .axis_session_id session (continuity across runs)
  memory.json      full decision + result log
"""

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests as http_lib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE            = Path(__file__).parent
AGENT_ID_FILE   = HERE / ".axis_agent_id"
ENV_ID_FILE     = HERE / ".axis_env_id"
SESSION_ID_FILE = HERE / ".axis_session_id"
TASKS_PATH      = HERE / "tasks.md"
MEMORY_PATH     = HERE / "memory.json"

BETA            = "managed-agents-2026-04-01"
EXECUTOR_MODEL  = "claude-sonnet-4-6"
MAX_FEEDBACK    = 2       # max feedback iterations per task

try:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or open(
        os.path.expanduser("~/.anthropic_key")
    ).read().strip()
except OSError:
    raise RuntimeError("ANTHROPIC_API_KEY env var not set and ~/.anthropic_key not found.")
client = anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Memory layer
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text()).get("entries", [])
            except Exception:
                return []
        return []

    def save_entry(self, entry: dict) -> None:
        self.entries.append(entry)
        self.path.write_text(json.dumps({"entries": self.entries}, indent=2))

    def recent_context(self, n: int = 3) -> str:
        if not self.entries:
            return "No prior memory."
        lines = ["[AXIS Memory — recent decisions]"]
        for e in self.entries[-n:]:
            lines.append(f"• [{e['ts'][:10]}] {e['task']!r}")
            if e.get("plan"):
                lines.append(f"  Plan: {' → '.join(e['plan'][:4])}")
            if e.get("outcome"):
                lines.append(f"  Outcome: {e['outcome']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence helpers
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
    return env.id


def get_or_create_session(agent_id: str, env_id: str) -> str:
    if SESSION_ID_FILE.exists():
        session_id = SESSION_ID_FILE.read_text().strip()
        try:
            session = client.beta.sessions.retrieve(session_id, betas=[BETA])
            status = getattr(session, "status", "")
            if status not in ("terminated", "expired", "error", "archived"):
                return session.id
        except Exception:
            pass
    session = client.beta.sessions.create(
        agent={"type": "agent", "id": agent_id},
        environment_id=env_id,
        betas=[BETA],
    )
    SESSION_ID_FILE.write_text(session.id)
    return session.id


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_send_notification(title: str, message: str, **_) -> str:
    safe = lambda s: s.replace('"', '\\"').replace("'", "\\'")
    r = subprocess.run(
        ["osascript", "-e", f'display notification "{safe(message)}" with title "{safe(title)}"'],
        capture_output=True, text=True,
    )
    return f"Sent: {title!r}" if r.returncode == 0 else f"Failed: {r.stderr.strip()}"


def _tool_save_task(task: str, priority: str = "medium", due: str = "", **_) -> str:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_PATH.exists():
        TASKS_PATH.write_text("# AXIS Tasks\n\n")
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
    due_s = f" · due {due}" if due else ""
    line  = f"- [ ] {badge} [{priority.upper()}] {task}{due_s}  _(added {now})_\n"
    with TASKS_PATH.open("a") as f:
        f.write(line)
    return f"Saved [{priority}]: {task[:70]}"


def _tool_http_request(url: str, method: str = "GET",
                       headers: dict = None, body: str = None, **_) -> str:
    try:
        kwargs: dict = {"headers": headers or {}, "timeout": 15}
        if body:
            kwargs["data"] = body.encode()
        r = http_lib.request(method.upper(), url, **kwargs)
        return f"HTTP {r.status_code} ← {url}"
    except Exception as exc:
        return f"Error: {exc}"


_TOOL_DISPATCH = {
    "send_notification": _tool_send_notification,
    "save_task":         _tool_save_task,
    "http_request":      _tool_http_request,
}

EXECUTOR_TOOLS = [
    {
        "name": "send_notification",
        "description": (
            "Send a Mac desktop notification. Use for meetings, reminders, deadlines, "
            "calls, or anything time-sensitive the user should be alerted about."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "title":   {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["title", "message"],
        },
    },
    {
        "name": "save_task",
        "description": (
            "Save an action item to tasks.md. Use whenever the user needs to do, "
            "follow up on, prepare, or track something — even without explicit 'save' language."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "task":     {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "due":      {"type": "string"},
            },
            "required": ["task", "priority"],
        },
    },
    {
        "name": "http_request",
        "description": "Make an HTTP request. Use only when a specific URL is present.",
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "url":     {"type": "string"},
                "method":  {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                "headers": {"type": "object"},
                "body":    {"type": "string"},
            },
            "required": ["url", "method"],
        },
    },
]


# ---------------------------------------------------------------------------
# Session streaming
# ---------------------------------------------------------------------------

def _stream_to_session(session_id: str, message: str,
                       label: str = "AXIS") -> str:
    """Send a message to the AXIS session, stream + return the full response."""
    chunks: list[str] = []
    print(f"  {label}: ", end="", flush=True)

    with client.beta.sessions.events.stream(session_id=session_id, betas=[BETA]) as stream:
        def _send():
            client.beta.sessions.events.send(
                session_id=session_id,
                events=[{"type": "user.message",
                         "content": [{"type": "text", "text": message}]}],
                betas=[BETA],
            )
        t = threading.Thread(target=_send, daemon=True)
        t.start()

        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "agent.message":
                for block in getattr(event, "content", []):
                    if getattr(block, "type", None) == "text":
                        print(block.text, end="", flush=True)
                        chunks.append(block.text)
            elif etype == "session.status_idle":
                print()
                break
            elif etype == "session.status_terminated":
                print("\n  [Session terminated]")
                SESSION_ID_FILE.unlink(missing_ok=True)
                break
        t.join()

    return "".join(chunks)


# ---------------------------------------------------------------------------
# Planner  (structured JSON output)
# ---------------------------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of 2–5 specific, executable sub-tasks",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["sub_tasks", "reasoning"],
}

_PLANNER_SYSTEM = """\
You are the AXIS Planning Engine. Given a complex task and context, decompose it
into 2–5 concrete, independently executable sub-tasks. Each sub-task must be
specific enough for an execution engine to act on immediately (save a task, send
a notification, call an API, etc.). Output JSON only. No commentary.
"""


def plan_task(task: str, axis_response: str, memory_ctx: str) -> tuple[list[str], str]:
    result = client.messages.create(
        model=EXECUTOR_MODEL,
        max_tokens=1024,
        system=_PLANNER_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"AXIS perspective:\n{axis_response}\n\n"
                f"Memory:\n{memory_ctx}"
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
    )
    data = json.loads(result.content[0].text)
    return data["sub_tasks"], data["reasoning"]


# ---------------------------------------------------------------------------
# Autonomous executor  (one sub-task, full tool loop)
# ---------------------------------------------------------------------------

_EXECUTOR_SYSTEM = """\
You are the AXIS Execution Engine. Act on the sub-task autonomously.
Call one or more tools as needed. Be specific — extract names, times, and context.
If no real-world action is warranted, call no tools.
"""


def execute_sub_task(sub_task: str, context: str = "") -> list[str]:
    """Run one sub-task through the tool-calling loop. Returns list of result strings."""
    messages = [{
        "role": "user",
        "content": f"Sub-task: {sub_task}\nContext: {context}" if context else f"Sub-task: {sub_task}",
    }]
    results: list[str] = []

    while True:
        resp = client.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=1024,
            system=_EXECUTOR_SYSTEM,
            tools=EXECUTOR_TOOLS,
            messages=messages,
        )
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            break

        tool_results = []
        for call in calls:
            fn  = _TOOL_DISPATCH.get(call.name)
            out = fn(**call.input) if fn else f"Unknown tool: {call.name}"
            ok  = not out.startswith(("Error", "Unknown", "Failed"))
            print(f"    {'✓' if ok else '✗'} {call.name:<22} {out}")
            results.append(f"{call.name}: {out}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": out,
            })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user",      "content": tool_results})
        if resp.stop_reason == "end_turn":
            break

    return results


# ---------------------------------------------------------------------------
# Main autonomous loop
# ---------------------------------------------------------------------------

def run_autonomous_task(session_id: str, task: str, memory: MemoryStore) -> None:
    width = 64
    ts    = datetime.now(timezone.utc).isoformat()

    print(f"\n{'═' * width}")
    print(f"  TASK: {task}")
    print(f"{'═' * width}")

    # ── 1. AXIS initial response ─────────────────────────────────────────
    memory_ctx   = memory.recent_context()
    axis_message = (
        f"{task}\n\n[Memory context]\n{memory_ctx}"
        if memory.entries else task
    )
    print(f"\n[1/4] AXIS perspective")
    print(f"{'─' * width}")
    axis_response = _stream_to_session(session_id, axis_message)

    # ── 2. Plan ──────────────────────────────────────────────────────────
    print(f"\n[2/4] Planning")
    print(f"{'─' * width}")
    sub_tasks, reasoning = plan_task(task, axis_response, memory_ctx)
    print(f"  Reasoning: {reasoning}")
    for i, st in enumerate(sub_tasks, 1):
        print(f"  {i}. {st}")

    # ── 3. Execute each sub-task ─────────────────────────────────────────
    print(f"\n[3/4] Execution ({len(sub_tasks)} sub-tasks)")
    print(f"{'─' * width}")
    all_step_results: list[dict] = []

    for i, sub_task in enumerate(sub_tasks, 1):
        print(f"\n  Sub-task {i}: {sub_task}")
        results = execute_sub_task(sub_task, context=task)
        all_step_results.append({"sub_task": sub_task, "results": results})
        if not results:
            print("    — no tool action taken")

    # ── 4. Feedback loop ─────────────────────────────────────────────────
    print(f"\n[4/4] Feedback + suggestions")
    print(f"{'─' * width}")

    summary_lines = [
        f"All sub-tasks for '{task}' have been executed. Here's the full report:\n"
    ]
    for i, step in enumerate(all_step_results, 1):
        summary_lines.append(f"{i}. {step['sub_task']}")
        for r in step["results"]:
            summary_lines.append(f"   → {r}")
        if not step["results"]:
            summary_lines.append("   → (no action taken)")

    summary_lines.append(
        "\nBased on these results: (a) is the original goal fully achieved? "
        "(b) what is the single most important next step the user should take?"
    )
    feedback_msg = "\n".join(summary_lines)

    feedback_response = _stream_to_session(session_id, feedback_msg, label="AXIS feedback")

    # Run executor one more time on the feedback if AXIS surfaced new actions
    print(f"\n  Checking feedback for additional actions…")
    extra_results = execute_sub_task(
        f"Based on this AXIS feedback, take any immediate actions warranted:\n{feedback_response}",
        context=task,
    )
    if not extra_results:
        print("    — no additional actions needed")

    # ── 5. Persist to memory ─────────────────────────────────────────────
    outcome_parts = [r for step in all_step_results for r in step["results"]] + extra_results
    outcome = "; ".join(outcome_parts[:4]) if outcome_parts else "no actions taken"

    memory.save_entry({
        "ts":          ts,
        "task":        task,
        "plan":        sub_tasks,
        "reasoning":   reasoning,
        "steps":       all_step_results,
        "suggestions": feedback_response[-400:],
        "outcome":     outcome,
    })
    print(f"\n  Memory saved → {MEMORY_PATH.name}  ({len(memory.entries)} total entries)")

    print(f"\n{'═' * width}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent_id   = load_agent_id()
    env_id     = get_or_create_env()
    session_id = get_or_create_session(agent_id, env_id)
    memory     = MemoryStore(MEMORY_PATH)

    print(f"Agent:   {agent_id}")
    print(f"Env:     {env_id}")
    print(f"Session: {session_id}")
    print(f"Memory:  {len(memory.entries)} prior entries")

    run_autonomous_task(
        session_id,
        "I need to prepare for a big client presentation next Monday",
        memory,
    )

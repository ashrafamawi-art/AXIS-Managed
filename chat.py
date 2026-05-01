"""
AXIS Chat UI — Chainlit interface for the AXIS autonomous agent.

Run with:
  chainlit run ~/AXIS-Managed/chat.py

Full loop per message:
  user input → AXIS session (streamed) → executor (autonomous) → results in chat
"""

import asyncio
import json
import os
import subprocess
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import chainlit as cl
import requests as http_lib

warnings.filterwarnings("ignore", category=Warning)

# ---------------------------------------------------------------------------
# Config  (mirrors session_runner.py)
# ---------------------------------------------------------------------------

HERE            = Path(__file__).parent
AGENT_ID_FILE   = HERE / ".axis_agent_id"
ENV_ID_FILE     = HERE / ".axis_env_id"
SESSION_ID_FILE = HERE / ".axis_session_id"
TASKS_PATH      = HERE / "tasks.md"
MEMORY_PATH     = HERE / "memory.json"
BETA            = "managed-agents-2026-04-01"
EXECUTOR_MODEL  = "claude-sonnet-4-6"

api_key = open(os.path.expanduser("~/.anthropic_key")).read().strip()
client  = anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Memory
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
            return ""
        lines = ["[AXIS Memory]"]
        for e in self.entries[-n:]:
            lines.append(f"• [{e['ts'][:10]}] {e['task']!r}")
            if e.get("outcome"):
                lines.append(f"  → {e['outcome'][:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_agent_id() -> str:
    if not AGENT_ID_FILE.exists():
        raise RuntimeError("Run axis_managed.py first to create the AXIS agent.")
    return AGENT_ID_FILE.read_text().strip()


def _get_or_create_env() -> str:
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


def _get_or_create_session(agent_id: str, env_id: str) -> str:
    if SESSION_ID_FILE.exists():
        sid = SESSION_ID_FILE.read_text().strip()
        try:
            s = client.beta.sessions.retrieve(sid, betas=[BETA])
            if getattr(s, "status", "") not in ("terminated", "expired", "error", "archived"):
                return s.id
        except Exception:
            pass
    s = client.beta.sessions.create(
        agent={"type": "agent", "id": agent_id},
        environment_id=env_id,
        betas=[BETA],
    )
    SESSION_ID_FILE.write_text(s.id)
    return s.id


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

_EXECUTOR_TOOLS = [
    {
        "name": "send_notification",
        "description": "Send a Mac desktop notification for meetings, reminders, or deadlines.",
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
        "description": "Save an action item to tasks.md. Use whenever the user needs to track, do, or follow up on something.",
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
        "description": "Make an HTTP request. Only use when a specific URL is present.",
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "url":    {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                "headers": {"type": "object"},
                "body":   {"type": "string"},
            },
            "required": ["url", "method"],
        },
    },
]

_EXECUTOR_SYSTEM = """\
You are the AXIS Execution Engine. Autonomously decide which real-world tools to call
based on the user's input and AXIS's response. Be specific — extract names, times, context.
Call no tools if nothing actionable is needed (e.g., a simple factual question).
"""


# ---------------------------------------------------------------------------
# Async AXIS session streaming  (sync SDK bridged via queue)
# ---------------------------------------------------------------------------

async def _stream_axis_response(session_id: str, task: str,
                                cl_msg: cl.Message) -> str:
    loop   = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    chunks: list[str] = []

    def _sync_run():
        try:
            with client.beta.sessions.events.stream(
                session_id=session_id, betas=[BETA]
            ) as stream:
                def _send():
                    client.beta.sessions.events.send(
                        session_id=session_id,
                        events=[{
                            "type":    "user.message",
                            "content": [{"type": "text", "text": task}],
                        }],
                        betas=[BETA],
                    )
                t = threading.Thread(target=_send, daemon=True)
                t.start()

                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "agent.message":
                        for block in getattr(event, "content", []):
                            if getattr(block, "type", None) == "text":
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("chunk", block.text)), loop
                                ).result()
                    elif etype == "session.status_idle":
                        break
                    elif etype == "session.status_terminated":
                        SESSION_ID_FILE.unlink(missing_ok=True)
                        break
                t.join()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc))), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(
                queue.put(("done", None)), loop
            ).result()

    thread = threading.Thread(target=_sync_run, daemon=True)
    thread.start()

    while True:
        kind, data = await queue.get()
        if kind == "done":
            break
        elif kind == "chunk":
            await cl_msg.stream_token(data)
            chunks.append(data)
        elif kind == "error":
            await cl_msg.stream_token(f"\n\n⚠️ Session error: {data}")
            break

    thread.join(timeout=10)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Autonomous executor  (sync, run in thread pool)
# ---------------------------------------------------------------------------

def _run_executor_sync(task: str, axis_response: str) -> list[dict]:
    """Returns list of {tool, status, output} dicts."""
    results: list[dict] = []
    messages = [{
        "role": "user",
        "content": f"User input:\n{task}\n\nAXIS response:\n{axis_response}",
    }]

    while True:
        resp = client.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=1024,
            system=_EXECUTOR_SYSTEM,
            tools=_EXECUTOR_TOOLS,
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
            results.append({"tool": call.name, "ok": ok, "output": out})
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": call.id,
                "content":     out,
            })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user",      "content": tool_results})
        if resp.stop_reason == "end_turn":
            break

    return results


# ---------------------------------------------------------------------------
# Chainlit handlers
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    try:
        agent_id   = _load_agent_id()
        env_id     = _get_or_create_env()
        session_id = _get_or_create_session(agent_id, env_id)
        memory     = MemoryStore(MEMORY_PATH)

        cl.user_session.set("session_id", session_id)
        cl.user_session.set("memory",     memory)

        mem_line = (
            f"📚 **{len(memory.entries)} prior session(s)** in memory."
            if memory.entries else "🆕 No prior memory — fresh start."
        )
        await cl.Message(
            content=(
                f"## AXIS is ready\n\n"
                f"{mem_line}\n\n"
                f"Type anything — AXIS will respond and act autonomously."
            )
        ).send()

    except Exception as exc:
        await cl.ErrorMessage(content=f"Boot failed: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id: str      = cl.user_session.get("session_id")
    memory:     MemoryStore = cl.user_session.get("memory")
    task = message.content.strip()
    if not task:
        return

    ts = datetime.now(timezone.utc).isoformat()

    # Prepend memory context to task if we have prior entries
    mem_ctx = memory.recent_context()
    axis_input = f"{task}\n\n{mem_ctx}" if mem_ctx else task

    # ── 1. Stream AXIS response ───────────────────────────────────────────
    axis_msg = cl.Message(content="", author="AXIS")
    await axis_msg.send()

    axis_response = await _stream_axis_response(session_id, axis_input, axis_msg)
    await axis_msg.update()

    if not axis_response:
        return

    # ── 2. Autonomous executor ────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    results: list[dict] = await loop.run_in_executor(
        None, _run_executor_sync, task, axis_response
    )

    if results:
        async with cl.Step(name="Executor", type="tool") as step:
            lines = []
            for r in results:
                icon = "✅" if r["ok"] else "❌"
                lines.append(f"{icon} **{r['tool']}** — {r['output']}")
            step.output = "\n".join(lines)

    # ── 3. Save to memory ─────────────────────────────────────────────────
    outcome = "; ".join(r["output"] for r in results) if results else "no action"
    memory.save_entry({
        "ts":      ts,
        "task":    task,
        "plan":    [],
        "steps":   [{"sub_task": task, "results": [r["output"] for r in results]}],
        "outcome": outcome,
    })

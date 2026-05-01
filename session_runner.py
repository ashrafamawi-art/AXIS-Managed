"""
AXIS Session Runner — streams tasks to a persistent AXIS Managed Agent session,
then dispatches the response through the local execution layer.

Full loop:
  input → session → AXIS response → extract action → execute → print result

Loads:  agent_id    from .axis_agent_id    (written by axis_managed.py)
        env_id      from .axis_env_id      (written on first run)
        session_id  from .axis_session_id  (persisted for continuity)
"""

import os
import re
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

api_key = open(os.path.expanduser("~/.anthropic_key")).read().strip()
client  = anthropic.Anthropic(api_key=api_key)


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
# Execution layer
# ---------------------------------------------------------------------------

def _send_notification(title: str, message: str) -> str:
    safe_title   = title.replace('"', '\\"').replace("'", "\\'")
    safe_message = message.replace('"', '\\"').replace("'", "\\'")
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode == 0:
        return f"Notification sent — {title!r}"
    return f"Notification failed: {r.stderr.strip()}"


def _save_task(description: str) -> str:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_PATH.exists():
        TASKS_PATH.write_text("# AXIS Tasks\n\n")
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"- [ ] {description}  _(added {now})_\n"
    with TASKS_PATH.open("a") as f:
        f.write(line)
    return f"Task saved → {TASKS_PATH}"


def _http_request(url: str, method: str = "GET") -> str:
    try:
        r = http_lib.request(method.upper(), url, timeout=10)
        return f"HTTP {r.status_code} ← {url}"
    except Exception as exc:
        return f"HTTP error: {exc}"


def _extract_reminder_text(task: str) -> tuple[str, str]:
    """Pull a short title and message from a reminder task string."""
    task = task.strip().rstrip(".")
    # Remove leading "remind me to" / "remind me"
    body = re.sub(r"^remind\s+me\s+(to\s+)?", "", task, flags=re.I).strip()
    body = body[0].upper() + body[1:] if body else task
    return "AXIS Reminder", body


def _extract_task_text(task: str) -> str:
    """Pull the task description, stripping 'save a task:' prefixes."""
    body = re.sub(r"^(save\s+(a\s+)?task\s*[:\-–]?\s*)", "", task, flags=re.I).strip()
    return body or task


def _extract_url(text: str) -> tuple[str, str]:
    """Find first URL and HTTP method hint in text."""
    url_match = re.search(r"https?://\S+", text)
    url    = url_match.group(0).rstrip(".,)") if url_match else ""
    method = "POST" if re.search(r"\b(post|send|submit|create)\b", text, re.I) else "GET"
    return url, method


def execute_action(task: str, response: str) -> None:
    """
    Keyword-dispatch execution layer.

    Detects intent from the task input, uses the AXIS response for payload
    content, and runs the appropriate tool(s).
    """
    t_low = task.lower()
    r_low = response.lower()
    dispatched = False

    print(f"\n  {'·' * 60}")
    print(f"  [executor]")

    # Remind / Notify → Mac notification
    if any(kw in t_low for kw in ("remind", "notify", "notification", "alert")):
        title, message = _extract_reminder_text(task)
        result = _send_notification(title, message)
        print(f"  ✓ notify    {result}")
        dispatched = True

    # Task / Save → tasks.md
    if any(kw in t_low for kw in ("task", "save", "to-do", "todo", "add to")):
        description = _extract_task_text(task)
        result = _save_task(description)
        print(f"  ✓ save_task {result}")
        dispatched = True

    # HTTP / API → outbound request
    if any(kw in t_low for kw in ("http", "api", "request", "fetch", "call the", "post to")):
        url, method = _extract_url(task + " " + response)
        if url:
            result = _http_request(url, method)
            print(f"  ✓ http      {result}")
            dispatched = True
        else:
            print("  ✗ http      No URL found in task or response — skipped.")
            dispatched = True

    if not dispatched:
        print("  — No execution action matched.")

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

    # Run execution layer on captured response
    full_response = "".join(response_chunks)
    execute_action(task, full_response)


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
        "Remind me to call Ahmed at 5pm",
        "Save a task: Review project proposal tomorrow",
    ]

    for task in tasks:
        run_task(session_id, task)

    width = 64
    print(f"\n{'═' * width}")
    print("  Session complete. ID saved for continuity on next run.")
    print(f"{'═' * width}\n")

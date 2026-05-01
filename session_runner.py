"""
AXIS Session Runner — streams tasks to a persistent AXIS Managed Agent session.

Loads:  agent_id    from .axis_agent_id    (written by axis_managed.py)
        env_id      from .axis_env_id      (written on first run)
        session_id  from .axis_session_id  (persisted for continuity)
"""

import os
import threading
from pathlib import Path

import anthropic

AGENT_ID_FILE   = Path(__file__).parent / ".axis_agent_id"
ENV_ID_FILE     = Path(__file__).parent / ".axis_env_id"
SESSION_ID_FILE = Path(__file__).parent / ".axis_session_id"
BETA            = "managed-agents-2026-04-01"

api_key = open(os.path.expanduser("~/.anthropic_key")).read().strip()
client  = anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Setup helpers
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
# Task runner
# ---------------------------------------------------------------------------

def run_task(session_id: str, task: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Task: {task}")
    print(f"{'─' * width}")
    print("  AXIS: ", end="", flush=True)

    # Open stream BEFORE sending so no events are missed
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

            elif etype == "agent.custom_tool_use":
                name  = getattr(event, "tool_name", "tool")
                print(f"\n  [tool → {name}]", end="", flush=True)

            elif etype == "session.status_idle":
                print()  # newline after streamed text
                break

            elif etype == "session.status_terminated":
                print("\n  [Session terminated]")
                SESSION_ID_FILE.unlink(missing_ok=True)
                break

        sender.join()


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
        "What are the top AI news today?",
    ]

    for task in tasks:
        run_task(session_id, task)

    width = 64
    print(f"\n{'═' * width}")
    print("  Session complete. ID saved for continuity on next run.")
    print(f"{'═' * width}\n")

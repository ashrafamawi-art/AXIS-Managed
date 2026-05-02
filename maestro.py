"""
AXIS Maestro — task orchestration layer.

Every request flows through:
  task
    → security.inspect_prompt()          ← BLOCK if HIGH risk
    → classify intent                    ← haiku classifier
    → route to agent                     ← calendar / task / memory / general
    → security.inspect_action()          ← per-subtask veto before execution
    → return unified response

MEDIUM-risk tasks skip intent routing and always go to general_agent
(full pipeline with maximum oversight).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests

import security
import council
import executor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_DIR    = Path(os.environ.get("AXIS_DATA_DIR", str(Path(__file__).parent)))
_MEMORY_FILE = _DATA_DIR / "memory.json"

_INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["calendar", "task", "memory", "general"],
        },
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["intent", "confidence", "reason"],
}

_INTENT_SYSTEM = """\
You are the AXIS Intent Classifier. Classify the request into exactly one category:
- calendar : scheduling, meetings, events, appointments, availability checks
- task     : to-do items, action items, reminders, follow-ups, things to do later
- memory   : "remember", "recall", "what did I tell you", storing/retrieving facts
- general  : everything else — questions, analysis, writing, advice, projects

Return JSON only.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_intent(task: str, client: anthropic.Anthropic) -> str:
    """Returns one of: calendar | task | memory | general."""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": f"Request: {task}"}],
            output_config={"format": {"type": "json_schema", "schema": _INTENT_SCHEMA}},
        )
        data = json.loads(resp.content[0].text)
        return data.get("intent", "general")
    except Exception:
        return "general"


def _send_security_alert(task: str, reason: str, category: str) -> None:
    """Fire-and-forget Telegram alert to admin when a HIGH-risk request is blocked."""
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    user_id = os.environ.get("TELEGRAM_USER_ID", "")
    if not token or not user_id:
        return
    text = (
        f"🚨 *AXIS Security Alert*\n\n"
        f"*Category:* `{category}`\n"
        f"*Reason:* {reason}\n\n"
        f"*Preview:* `{task[:120]}`"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": user_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception:
        pass


def _load_memory(n: int = 3) -> str:
    if not _MEMORY_FILE.exists():
        return ""
    try:
        entries = json.loads(_MEMORY_FILE.read_text()).get("entries", [])[-n:]
        lines   = ["[Recent memory]"]
        for e in entries:
            lines.append(f"• [{e['ts'][:10]}] {e.get('task','')!r}")
            if e.get("outcome"):
                lines.append(f"  → {e['outcome'][:80]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _save_memory(task: str, outcome: str) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        entries = []
        if _MEMORY_FILE.exists():
            entries = json.loads(_MEMORY_FILE.read_text()).get("entries", [])
        entries.append({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "task":    task,
            "outcome": outcome[:200],
        })
        _MEMORY_FILE.write_text(json.dumps({"entries": entries}, indent=2))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def task_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Save an action item directly via executor."""
    check = security.inspect_action("save_task", task)
    if not check["allowed"]:
        return {"answer": f"Blocked by security: {check['reason']}", "artifacts": {}}

    result = executor._save_task(task=task, priority="medium")
    return {
        "answer":    f"Task saved ✅\n\n{task}",
        "artifacts": {"tasks": [result]},
    }


def memory_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Retrieve recent memory context."""
    ctx = _load_memory()
    if ctx:
        return {"answer": ctx, "artifacts": {"memory": ctx}}
    return {"answer": "No prior memory found.", "artifacts": {}}


def calendar_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Calendar tasks go through general_agent — the executor handles calendar tools."""
    return general_agent(task, client)


def general_agent(task: str, client: anthropic.Anthropic) -> dict:
    """
    Full AXIS pipeline: council → AXIS session → plan → execute (with veto) → feedback.
    Uses server._run_pipeline via late import to avoid circular dependency.
    """
    # Late import — by call time server.py is fully loaded.
    import server as srv

    # Run the full pipeline
    raw = srv._run_pipeline(task)

    # Extract core fields; maestro.run() adds the security/routing wrapper.
    return {
        "answer":    raw.get("answer", ""),
        "artifacts": raw.get("artifacts", {}),
        "council":   raw.get("council", {}),
        "plan":      raw.get("plan", {}),
    }


_AGENT_MAP = {
    "calendar": calendar_agent,
    "task":     task_agent,
    "memory":   memory_agent,
    "general":  general_agent,
}

# ---------------------------------------------------------------------------
# Maestro — main entry point
# ---------------------------------------------------------------------------

def run(task: str, client: anthropic.Anthropic) -> dict:
    """
    Orchestrate a task through the full security + routing pipeline.

    Returns a response dict compatible with POST /task:
    {
        "id", "task", "status", "answer",
        "security": {"risk", "reason"},
        "routing":  {"intent"},
        "council":  {...},   # only for general/calendar
        "plan":     {...},   # only for general/calendar
        "artifacts": {...},
        "timestamp"
    }
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── 0. Confirm bypass — "confirm <task>" skips MEDIUM block ──────────
    confirmed = task.lower().startswith("confirm ")
    actual_task = task[len("confirm "):].strip() if confirmed else task

    # ── 1. Security: inspect prompt ───────────────────────────────────────
    sec      = security.inspect_prompt(actual_task)
    category = sec.get("category", "unknown")

    if sec["blocked"] or sec["risk"] == security.HIGH:
        _send_security_alert(actual_task, sec["reason"], category)
        return {
            "id":        str(uuid.uuid4()),
            "task":      actual_task,
            "status":    "blocked",
            "reason":    category,
            "message":   "🚫 Request denied due to security policy.",
            "security":  {"risk": sec["risk"], "reason": sec["reason"]},
            "routing":   {"intent": "blocked"},
            "artifacts": {},
            "timestamp": ts,
        }

    # ── 2. Classify intent ────────────────────────────────────────────────
    intent = _classify_intent(actual_task, client)

    # MEDIUM risk without confirm → hold for user confirmation
    if sec["risk"] == security.MEDIUM and not confirmed:
        return {
            "id":        str(uuid.uuid4()),
            "task":      actual_task,
            "status":    "needs_confirmation",
            "reason":    category,
            "message":   "⚠️ This action requires your confirmation. Reply 'confirm' to proceed.",
            "security":  {"risk": sec["risk"], "reason": sec["reason"]},
            "routing":   {"intent": intent},
            "artifacts": {},
            "timestamp": ts,
        }

    # ── 3. Route to agent ─────────────────────────────────────────────────
    agent_fn     = _AGENT_MAP.get(intent, general_agent)
    agent_result = agent_fn(actual_task, client)
    agent_name   = agent_fn.__name__

    # ── 4. Persist to memory ──────────────────────────────────────────────
    _save_memory(actual_task, agent_result.get("answer", ""))

    # ── 5. Build unified response ─────────────────────────────────────────
    response: dict = {
        "id":        str(uuid.uuid4()),
        "task":      actual_task,
        "status":    "done",
        "agent":     agent_name,
        "answer":    agent_result.get("answer", "(no response)"),
        "security":  {"risk": sec["risk"], "reason": sec["reason"]},
        "routing":   {"intent": intent},
        "artifacts": agent_result.get("artifacts", {}),
        "timestamp": ts,
    }

    for key in ("council", "plan"):
        if key in agent_result:
            response[key] = agent_result[key]

    return response

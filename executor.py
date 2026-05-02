"""
AXIS Executor — autonomous tool planning and execution.

plan()    — decomposes a task into ordered sub-tasks via Claude
execute() — runs one sub-task through the Claude tool-use loop
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import requests as http_lib

_MODEL      = "claude-sonnet-4-6"
_DATA_DIR   = Path(os.environ.get("AXIS_DATA_DIR", str(Path(__file__).parent)))
_TASKS_FILE = _DATA_DIR / "tasks.md"
_TOKEN_FILE = _DATA_DIR / "token.pickle"

# ---------------------------------------------------------------------------
# Google Calendar credentials — cloud-first, local fallback
# ---------------------------------------------------------------------------

_gcal_creds_cache = None


def _load_gcal_creds():
    """Load Google Calendar creds: GOOGLE_TOKEN_JSON (cloud) → token.pickle (local dev)."""
    global _gcal_creds_cache
    try:
        from google.auth.transport.requests import Request

        if _gcal_creds_cache is not None:
            if _gcal_creds_cache.valid:
                return _gcal_creds_cache
            if _gcal_creds_cache.expired and _gcal_creds_cache.refresh_token:
                _gcal_creds_cache.refresh(Request())
                return _gcal_creds_cache

        token_json = os.environ.get("GOOGLE_TOKEN_JSON", "")
        if token_json:
            from google.oauth2.credentials import Credentials
            scopes = ["https://www.googleapis.com/auth/calendar.events"]
            creds  = Credentials.from_authorized_user_info(json.loads(token_json), scopes)
            if not creds.valid and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            _gcal_creds_cache = creds
            return creds

        if _TOKEN_FILE.exists():
            import pickle
            with _TOKEN_FILE.open("rb") as f:
                creds = pickle.load(f)
            if creds.valid:
                _gcal_creds_cache = creds
                return creds
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with _TOKEN_FILE.open("wb") as f:
                    pickle.dump(creds, f)
                _gcal_creds_cache = creds
                return creds
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _send_notification(title: str, message: str, **_) -> str:
    import re
    clean = lambda s: re.sub(r"[^\x00-\x7F]+", "", s).replace('"', '\\"').strip()
    try:
        r = subprocess.run(
            ["osascript", "-e",
             f'display notification "{clean(message)}" with title "{clean(title)}"'],
            capture_output=True, text=True,
        )
        return f"Sent: {title!r}" if r.returncode == 0 else f"Notification queued: {title!r}"
    except FileNotFoundError:
        return f"Notification queued (no display): {title!r}"


def _save_task(task: str = "", priority: str = "medium", due: str = "", **extra) -> str:
    if not task:
        task = next((str(v) for v in extra.values() if isinstance(v, str)), "Untitled task")
    _TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _TASKS_FILE.exists():
        _TASKS_FILE.write_text("# AXIS Tasks\n\n")
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
    due_s = f" · due {due}" if due else ""
    line  = f"- [ ] {badge} [{priority.upper()}] {task}{due_s}  _(added {now})_\n"
    with _TASKS_FILE.open("a") as f:
        f.write(line)
    return f"Task saved [{priority}]: {task[:80]}"


def _http_request(url: str, method: str = "GET",
                  headers: dict = None, body: str = None, **_) -> str:
    try:
        kwargs: dict = {"headers": headers or {}, "timeout": 15}
        if body:
            kwargs["data"] = body.encode()
        r = http_lib.request(method.upper(), url, **kwargs)
        return f"HTTP {r.status_code} ← {url}"
    except Exception as exc:
        return f"Error: {exc}"


def _create_calendar_event(
    title: str,
    start_iso: str,
    end_iso: str = "",
    description: str = "",
    location: str = "",
    **_,
) -> str:
    """Create a Google Calendar event. Falls back to save_task if credentials missing."""
    try:
        creds = _load_gcal_creds()
        if creds is None:
            _save_task(
                task=f"[PENDING CALENDAR] {title} — {start_iso}",
                priority="high",
            )
            return (
                "Google Calendar credentials are not configured on Render. "
                "Task saved as pending."
            )

        from calendar_integration import CalendarService

        start_dt = datetime.fromisoformat(start_iso)
        end_dt   = datetime.fromisoformat(end_iso) if end_iso else start_dt + timedelta(hours=1)

        cal       = CalendarService(creds=creds)
        conflicts = cal.check_conflicts(start_dt, end_dt)
        conflict_note = ""
        if conflicts:
            names = [c.get("summary", "Untitled") for c in conflicts[:2]]
            conflict_note = f" ⚠️ Conflict with: {', '.join(names)}"

        event    = cal.create_event(title, start_dt, end_dt, description, location)
        fmt_time = start_dt.strftime("%a %b %-d at %-I:%M %p")
        return f"CALENDAR_EVENT_CREATED: '{title}' on {fmt_time}{conflict_note}"

    except Exception as exc:
        try:
            _save_task(
                task=f"[PENDING CALENDAR] {title} — {start_iso}",
                priority="high",
            )
        except Exception:
            pass
        return f"Calendar execution failed, task saved as pending. Error: {exc}"


_DISPATCH = {
    "create_calendar_event": _create_calendar_event,
    "send_notification":     _send_notification,
    "save_task":             _save_task,
    "http_request":          _http_request,
}

TOOLS = [
    {
        "name": "create_calendar_event",
        "description": (
            "Create a Google Calendar event. Use whenever the user asks to schedule, "
            "book, or arrange a meeting, call, or appointment. "
            "Resolve relative dates (today, tomorrow, next Monday) using today's date from the system prompt."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "title":       {"type": "string", "description": "Event title"},
                "start_iso":   {"type": "string", "description": "Start datetime ISO 8601, e.g. '2026-05-03T15:00:00'"},
                "end_iso":     {"type": "string", "description": "End datetime ISO 8601. Defaults to 1 hour after start."},
                "description": {"type": "string", "description": "Optional notes or agenda"},
                "location":    {"type": "string", "description": "Optional location"},
            },
            "required": ["title", "start_iso"],
        },
    },
    {
        "name": "send_notification",
        "description": (
            "Send a desktop notification for meetings, reminders, or deadlines. "
            "Use for time-sensitive items the user should be alerted about."
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
            "Save an action item to the task list. Use whenever the user needs to "
            "do, follow up on, prepare, or track something."
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
        "description": "Make an HTTP request. Only use when a specific URL is present.",
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
# Plan
# ---------------------------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-5 specific, executable sub-tasks in order",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["sub_tasks", "reasoning"],
}

_PLAN_SYSTEM = """\
You are the AXIS Planning Engine. Decompose the task into 2-5 concrete, executable sub-tasks.
Each sub-task must be specific enough to act on immediately (save a task, call an API, etc.).
Output valid JSON only.
"""


def plan(task: str, axis_response: str, memory_ctx: str,
         client: anthropic.Anthropic) -> tuple[list[str], str]:
    """Decompose task into ordered sub-tasks. Returns (sub_tasks, reasoning)."""
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_PLAN_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"AXIS response:\n{axis_response}\n\n"
                f"Memory:\n{memory_ctx or 'none'}"
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
    )
    data = json.loads(resp.content[0].text)
    return data["sub_tasks"], data["reasoning"]


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def _exec_system() -> str:
    now = datetime.now()
    return (
        f"You are the AXIS Execution Engine. Today is {now.strftime('%A, %B %-d, %Y')} "
        f"(local time {now.strftime('%-I:%M %p')}).\n"
        "Act on the sub-task autonomously using available tools. "
        "Be specific — extract names, times, and context from the task.\n"
        "When creating calendar events, resolve relative dates (today, tomorrow, next Monday) "
        "using today's date above. Default event duration is 1 hour unless stated otherwise.\n"
        "If no real-world action is warranted, call no tools."
    )


def execute(sub_task: str, context: str, client: anthropic.Anthropic) -> list[dict]:
    """Run one sub-task through the tool loop. Returns list of {tool, output} dicts."""
    messages = [{
        "role": "user",
        "content": f"Sub-task: {sub_task}\nContext: {context}" if context else f"Sub-task: {sub_task}",
    }]
    results: list[dict] = []

    for _ in range(10):
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_exec_system(),
            tools=TOOLS,
            messages=messages,
        )
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            break

        tool_results = []
        for call in calls:
            fn  = _DISPATCH.get(call.name)
            out = fn(**call.input) if fn else f"Unknown tool: {call.name}"
            results.append({"tool": call.name, "output": out})
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user",      "content": tool_results})
        if resp.stop_reason == "end_turn":
            break

    return results

"""
AXIS Executor — autonomous tool planning and execution.

plan()    — decomposes a task into ordered sub-tasks via Claude
execute() — runs one sub-task through the Claude tool-use loop
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import requests as http_lib

_MODEL      = "claude-sonnet-4-6"
_DATA_DIR   = Path(os.environ.get("AXIS_DATA_DIR", str(Path(__file__).parent)))
_TASKS_FILE = _DATA_DIR / "tasks.md"

# ---------------------------------------------------------------------------
# Calendar dedup — prevents the same event being created more than once
# within a 30-second window (covers multiple sub-tasks in the same pipeline run
# and Telegram retries that arrive before the first response is acknowledged).
# ---------------------------------------------------------------------------

_CALENDAR_DEDUP_SECS = 30
_recent_calendar_events: dict[int, float] = {}  # event_hash → monotonic time


def _event_hash(title: str, start_iso: str) -> int:
    return hash(title.strip().lower() + start_iso.strip())

# ---------------------------------------------------------------------------
# Google Calendar credentials — cloud-first, local fallback
# ---------------------------------------------------------------------------

_gcal_creds_cache = None


def _load_gcal_creds():
    """Load Google Calendar creds exclusively from GOOGLE_TOKEN_JSON env var.
    Identical auth path to /cal/diag — no token.pickle, no manual headers."""
    global _gcal_creds_cache
    from calendar_integration import GCAL_SCOPES, _check_token_scopes, _interpret_error
    try:
        from google.auth.transport.requests import Request

        if _gcal_creds_cache is not None:
            if _gcal_creds_cache.valid:
                return _gcal_creds_cache
            if _gcal_creds_cache.expired and _gcal_creds_cache.refresh_token:
                _gcal_creds_cache.refresh(Request())
                return _gcal_creds_cache

        token_json = os.environ.get("GOOGLE_TOKEN_JSON", "")
        if not token_json:
            print("[executor] GOOGLE_TOKEN_JSON not set — calendar unavailable")
            return None

        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(json.loads(token_json), GCAL_SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        _check_token_scopes(creds)
        _gcal_creds_cache = creds
        return creds

    except (PermissionError, ValueError) as exc:
        print(f"[executor] Calendar auth error: {exc}")
        raise
    except Exception as exc:
        print(f"[executor] Calendar credentials unavailable: {_interpret_error(exc)}")
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
    # Google Calendar must go through the authenticated tools, not raw HTTP.
    if "googleapis.com/calendar" in url:
        return (
            "Error: direct HTTP to googleapis.com/calendar is blocked. "
            "Use 'get_calendar_events' to read events or "
            "'create_calendar_event' to write events."
        )
    try:
        kwargs: dict = {"headers": headers or {}, "timeout": 15}
        if body:
            kwargs["data"] = body.encode()
        r = http_lib.request(method.upper(), url, **kwargs)
        return f"HTTP {r.status_code} ← {url}"
    except Exception as exc:
        return f"Error: {exc}"


def _get_calendar_events(days: int = 7, **_) -> str:
    """Read upcoming events using the same authenticated client as /cal/diag."""
    try:
        creds = _load_gcal_creds()
        if creds is None:
            return (
                "⚠️ Google Calendar credentials not available. "
                "Ensure GOOGLE_TOKEN_JSON is set on Render."
            )
        from calendar_integration import CalendarService, _interpret_error
        events = CalendarService(creds=creds).get_upcoming_events(days=min(days, 30))
        if not events:
            return f"No events in the next {days} days."
        return CalendarService.fmt_events(events, f"📅 Next {days} days:")
    except (PermissionError, ValueError) as exc:
        return f"❌ Calendar auth error: {exc}"
    except Exception as exc:
        from calendar_integration import _interpret_error
        return f"❌ Calendar error: {_interpret_error(exc)}"


def _create_calendar_event(
    title: str,
    start_iso: str,
    end_iso: str = "",
    description: str = "",
    location: str = "",
    **_,
) -> str:
    """Create a Google Calendar event. Falls back to save_task if credentials missing."""
    # ── Dedup guard ───────────────────────────────────────────────────────────
    event_hash = _event_hash(title, start_iso)
    now_mono   = time.monotonic()

    # Prune expired entries
    expired = [k for k, t in _recent_calendar_events.items()
               if now_mono - t > _CALENDAR_DEDUP_SECS]
    for k in expired:
        del _recent_calendar_events[k]

    if event_hash in _recent_calendar_events:
        age = round(now_mono - _recent_calendar_events[event_hash], 1)
        print(f"[executor] DEDUP: skipping duplicate calendar event "
              f"'{title}' at {start_iso} hash={event_hash} (seen {age}s ago)")
        return f"DEDUP: calendar event '{title}' already created ({age}s ago — duplicate skipped)"

    _recent_calendar_events[event_hash] = now_mono
    print(f"[executor] calendar event: title={title!r} start={start_iso} hash={event_hash}")

    try:
        creds = _load_gcal_creds()
        if creds is None:
            _save_task(task=f"[PENDING CALENDAR] {title} — {start_iso}", priority="high")
            return (
                "⚠️ Google Calendar token not found or invalid.\n"
                "Fix: run  python3 check_calendar.py --reauth\n"
                "Event saved as a pending task."
            )

        from calendar_integration import CalendarService, _interpret_error

        start_dt = datetime.fromisoformat(start_iso)
        end_dt   = datetime.fromisoformat(end_iso) if end_iso else start_dt + timedelta(hours=1)

        cal       = CalendarService(creds=creds)
        conflicts = cal.check_conflicts(start_dt, end_dt)
        conflict_note = ""
        if conflicts:
            names = [c.get("summary", "Untitled") for c in conflicts[:2]]
            conflict_note = f" ⚠️ Conflict with: {', '.join(names)}"

        cal.create_event(title, start_dt, end_dt, description, location)
        fmt_time = start_dt.strftime("%a %b %-d at %-I:%M %p")
        return f"CALENDAR_EVENT_CREATED: '{title}' on {fmt_time}{conflict_note}"

    except (PermissionError, ValueError) as exc:
        # Scope / token problem — don't save as pending, surface the fix
        return f"❌ Calendar auth error: {exc}"

    except Exception as exc:
        diagnosis = _interpret_error(exc) if "calendar_integration" in dir() else str(exc)
        _save_task(task=f"[PENDING CALENDAR] {title} — {start_iso}", priority="high")
        return f"❌ Calendar error: {diagnosis}\nEvent saved as pending task."


_DISPATCH = {
    "create_calendar_event": _create_calendar_event,
    "get_calendar_events":   _get_calendar_events,
    "send_notification":     _send_notification,
    "save_task":             _save_task,
    "http_request":          _http_request,
}

TOOLS = [
    {
        "name": "get_calendar_events",
        "description": (
            "Read upcoming Google Calendar events. Use this to check availability, "
            "list scheduled meetings, or verify that an event exists. "
            "This is the ONLY correct way to read calendar data — "
            "never use http_request for googleapis.com/calendar."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days ahead to fetch (default 7, max 30).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Create a Google Calendar event for a meeting, call, appointment, or reminder. "
            "Always extract a descriptive title (include the person's name or topic). "
            "Always populate description and location."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Descriptive title including the person and/or purpose. "
                        "Examples: 'Meeting with Bassam', 'Call with Ali — Villa Project', "
                        "'Follow-up with Client'. Never use bare 'Meeting' or 'Call' alone."
                    ),
                },
                "start_iso": {
                    "type": "string",
                    "description": "Start datetime ISO 8601, e.g. '2026-05-03T15:00:00'",
                },
                "end_iso": {
                    "type": "string",
                    "description": "End datetime ISO 8601. Always set — default to 1 hour after start.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Always set to: 'Scheduled via AXIS: <exact original user message>'. "
                        "Example: 'Scheduled via AXIS: Add meeting with Bassam tomorrow at 3pm'"
                    ),
                },
                "location": {
                    "type": "string",
                    "description": "Location if mentioned, otherwise 'TBD'.",
                },
            },
            "required": ["title", "start_iso", "end_iso", "description", "location"],
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
        "description": (
            "Make an HTTP request to an external URL. "
            "Never use for googleapis.com/calendar — "
            "use get_calendar_events or create_calendar_event instead."
        ),
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

CRITICAL: Calendar event creation must appear as AT MOST ONE sub-task. Never generate two or
more sub-tasks that both result in creating a calendar event for the same meeting. If the task
involves scheduling a meeting, create exactly ONE sub-task for the calendar event creation.

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
        "Be specific — extract names, times, and context from the task.\n\n"
        "CALENDAR EVENT QUALITY RULES — follow these exactly:\n"
        "- title: Extract the person's name and purpose. Examples:\n"
        "    'Add meeting with Bassam tomorrow at 3pm'  → 'Meeting with Bassam'\n"
        "    'Schedule call with Ali about the villa'   → 'Call with Ali — Villa'\n"
        "    'Remind me to follow up with the client'   → 'Follow-up with Client'\n"
        "  NEVER use bare generic titles: 'Meeting', 'Call', 'Appointment', 'Event'.\n"
        "  Always include the name or topic that makes the event identifiable.\n"
        "- end_iso: Default to 1 hour after start if not specified.\n"
        "- description: Always set to 'Scheduled via AXIS: <exact original user message>'.\n"
        "- location: Use 'TBD' when no location is mentioned.\n\n"
        "Resolve relative dates (today, tomorrow, next Monday) using today's date above.\n"
        "If no real-world action is warranted, call no tools."
    )


def execute(sub_task: str, context: str, client: anthropic.Anthropic,
            request_id: str = "") -> list[dict]:
    """Run one sub-task through the tool loop. Returns list of {tool, output} dicts."""
    prefix = f"[executor{f'/{request_id[:8]}' if request_id else ''}]"
    print(f"{prefix} sub-task: {sub_task[:100]!r}")

    messages = [{
        "role": "user",
        "content": f"Sub-task: {sub_task}\nContext: {context}" if context else f"Sub-task: {sub_task}",
    }]
    results: list[dict] = []

    # Local dedup: prevent the inner loop from calling create_calendar_event
    # more than once per execute() invocation (belt-and-suspenders alongside the
    # module-level 30-second window).
    _local_calendar_hashes: set[int] = set()

    for round_num in range(10):
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
            print(f"{prefix}   round {round_num + 1}: {call.name}({list(call.input.keys())})")

            # Inner-loop calendar dedup
            if call.name == "create_calendar_event":
                eh = _event_hash(
                    call.input.get("title", ""),
                    call.input.get("start_iso", ""),
                )
                if eh in _local_calendar_hashes:
                    out = "DEDUP: create_calendar_event already called in this execution — skipped"
                    print(f"{prefix}   DEDUP (inner loop): hash={eh}")
                    results.append({"tool": call.name, "output": out})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": call.id, "content": out,
                    })
                    continue
                _local_calendar_hashes.add(eh)

            fn  = _DISPATCH.get(call.name)
            out = fn(**call.input) if fn else f"Unknown tool: {call.name}"
            print(f"{prefix}   → {out[:120]}")
            results.append({"tool": call.name, "output": out})
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user",      "content": tool_results})
        if resp.stop_reason == "end_turn":
            break

    return results

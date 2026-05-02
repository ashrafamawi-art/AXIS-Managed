"""
AXIS Task Manager — structured task parsing, lifecycle management, and local storage.

Each task is stored as one JSON line in /tmp/axis/tasks.jsonl.

Intents:   calendar_event | reminder | follow_up | message_to_person |
           message_send | delete_action |
           research_task | coding_task | general_question
Statuses:  pending_confirmation | scheduled | completed | failed |
           waiting_for_user | cancelled

Confirmation policy v2 — confirmation is required ONLY for:
  delete_action  — any deletion (events, tasks, files)
  message_send   — actually transmitting a message/email to someone
  irreversible   — anything explicitly flagged as irreversible

Everything else (create, update, draft, research, reminders, follow-ups)
executes immediately with no friction.
"""

import dataclasses
import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

CALENDAR_EVENT   = "calendar_event"
REMINDER         = "reminder"
FOLLOW_UP        = "follow_up"
MESSAGE_PERSON   = "message_to_person"   # draft / compose — no confirmation
MESSAGE_SEND     = "message_send"        # actually sending to someone — confirmation required
DELETE_ACTION    = "delete_action"       # any deletion — confirmation required
RESEARCH_TASK    = "research_task"
CODING_TASK      = "coding_task"
GENERAL_QUESTION = "general_question"

# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------

PENDING_CONFIRMATION = "pending_confirmation"
SCHEDULED            = "scheduled"
COMPLETED            = "completed"
FAILED               = "failed"
WAITING_FOR_USER     = "waiting_for_user"
CANCELLED            = "cancelled"

STATUS_EMOJI = {
    PENDING_CONFIRMATION: "⏳",
    SCHEDULED:            "📅",
    COMPLETED:            "✅",
    FAILED:               "❌",
    WAITING_FOR_USER:     "🕐",
    CANCELLED:            "🚫",
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_DIR   = Path("/tmp/axis")
_TASKS_FILE = _DATA_DIR / "tasks.jsonl"
_LOG_FILE   = _DATA_DIR / "task_manager.log"
_FILE_LOCK  = threading.Lock()

# Confirmation policy v2: only delete and send actions require confirmation
REQUIRES_CONFIRMATION = {DELETE_ACTION, MESSAGE_SEND}

# Intents worth persisting to tasks.jsonl
PERSIST_INTENTS = {CALENDAR_EVENT, REMINDER, FOLLOW_UP, MESSAGE_PERSON, MESSAGE_SEND, DELETE_ACTION}

# Confirmation words — entire stripped message must match one of these
_CONFIRM_WORDS = {
    "confirm", "yes", "ok", "okay", "approve",
    "go ahead", "do it", "proceed", "sure", "yep", "yup",
    "نعم", "موافق", "تأكيد", "تمام", "حسناً", "اوكي",
}

# ---------------------------------------------------------------------------
# Intent detection keyword sets
# ---------------------------------------------------------------------------

_DELETE_KW = {
    "delete", "remove", "cancel", "erase", "clear",
    "حذف", "امسح", "ألغِ", "احذف",
}
_SEND_KW = {
    "send", "sending", "email", "forward", "deliver",
    "ارسل", "ابعث",
}
_CALENDAR_KW = {
    "meeting", "schedule", "appointment", "call", "session",
    "book", "arrange", "add to calendar", "block time",
    "اجتماع", "موعد", "جدول", "حجز", "جلسة",
}
_REMINDER_KW = {
    "remind", "reminder", "remind me", "don't forget",
    "alert me", "notify me",
    "ذكرني", "تذكير", "لا تنس",
}
_FOLLOWUP_KW = {
    "follow up", "follow-up", "followup",
    "check in", "check back", "get back to",
    "متابعة", "تابع",
}
_DRAFT_KW = {
    "draft", "compose",
    "اكتب رسالة", "حضر",
}
_MESSAGE_KW = {
    "message", "tell", "contact", "text", "whatsapp",
    "write to", "reach out",
    "رسالة", "تواصل",
}
_RESEARCH_KW = {
    "research", "find", "search", "look up",
    "what is", "how to", "investigate",
    "ابحث", "بحث", "معلومات",
}
_CODING_KW = {
    "code", "function", "script", "implement",
    "write a", "debug", "fix bug", "refactor",
    "كود", "برمجة",
}

# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    task_id:               str
    source:                str
    user_message:          str
    intent:                str
    status:                str
    title:                 str
    due_at:                str
    person:                str
    requires_confirmation: bool
    created_at:            str
    updated_at:            str
    execution_result:      str = ""
    error:                 str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        return cls(
            task_id               = d.get("task_id", ""),
            source                = d.get("source", ""),
            user_message          = d.get("user_message", ""),
            intent                = d.get("intent", ""),
            status                = d.get("status", ""),
            title                 = d.get("title", ""),
            due_at                = d.get("due_at", ""),
            person                = d.get("person", ""),
            requires_confirmation = bool(d.get("requires_confirmation", False)),
            created_at            = d.get("created_at", ""),
            updated_at            = d.get("updated_at", ""),
            execution_result      = d.get("execution_result", ""),
            error                 = d.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _log(entry: dict) -> None:
    """Append one structured JSON line to task_manager.log. Never raises."""
    try:
        _ensure_dir()
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        with _FILE_LOCK:
            with _LOG_FILE.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def save(record: TaskRecord) -> None:
    """Append a new task record to tasks.jsonl."""
    _ensure_dir()
    with _FILE_LOCK:
        with _TASKS_FILE.open("a") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    _log({
        "event":   "task_created",
        "task_id": record.task_id,
        "intent":  record.intent,
        "status":  record.status,
        "title":   record.title,
    })


def _load_all() -> list[TaskRecord]:
    if not _TASKS_FILE.exists():
        return []
    with _FILE_LOCK:
        raw = _TASKS_FILE.read_text(errors="replace")
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(TaskRecord.from_dict(json.loads(line)))
        except Exception:
            pass
    return records


def _rewrite(records: list[TaskRecord]) -> None:
    _ensure_dir()
    with _FILE_LOCK:
        with _TASKS_FILE.open("w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def update(task_id: str, **fields) -> Optional[TaskRecord]:
    """Update fields on a task. Returns updated record or None if not found."""
    now = datetime.now(timezone.utc).isoformat()
    records = _load_all()
    updated = None
    for r in records:
        if r.task_id == task_id:
            for k, v in fields.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            r.updated_at = now
            updated = r
    _rewrite(records)
    if updated:
        _log({"event": "task_updated", "task_id": task_id,
              **{k: str(v)[:80] for k, v in fields.items()}})
    return updated


def cancel(task_id: str) -> bool:
    """Cancel a task by ID. Returns True if found."""
    return update(task_id, status=CANCELLED) is not None


def list_tasks(status_filter: str = None, limit: int = 50) -> list[TaskRecord]:
    """Return tasks newest-first, optionally filtered by status."""
    records = _load_all()
    if status_filter:
        records = [r for r in records if r.status == status_filter]
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records[:limit]


def get_latest_pending() -> Optional[TaskRecord]:
    """Return the most recent pending_confirmation task."""
    tasks = list_tasks(status_filter=PENDING_CONFIRMATION, limit=1)
    return tasks[0] if tasks else None


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _detect_intent(text: str) -> str:
    lower = text.lower()
    # DELETE first — "cancel meeting" beats calendar
    if any(kw in lower for kw in _DELETE_KW):
        return DELETE_ACTION
    # DRAFT beats SEND — "draft an email" / "compose a message" → no confirmation
    if any(kw in lower for kw in _DRAFT_KW):
        return MESSAGE_PERSON
    # SEND after draft guard — "send email to X" requires confirmation
    if any(kw in lower for kw in _SEND_KW):
        return MESSAGE_SEND
    if any(kw in lower for kw in _REMINDER_KW):
        return REMINDER
    if any(kw in lower for kw in _FOLLOWUP_KW):
        return FOLLOW_UP
    if any(kw in lower for kw in _MESSAGE_KW):
        return MESSAGE_PERSON
    if any(kw in lower for kw in _CALENDAR_KW):
        return CALENDAR_EVENT
    if any(kw in lower for kw in _CODING_KW):
        return CODING_TASK
    if any(kw in lower for kw in _RESEARCH_KW):
        return RESEARCH_TASK
    return GENERAL_QUESTION


def requires_confirmation(action_type: str, is_irreversible: bool = False) -> bool:
    """
    Confirmation policy v2.
    Returns True only for: delete actions, send actions, or explicitly irreversible actions.
    Everything else (create, update, draft, research, reminders) returns False.
    """
    return action_type in REQUIRES_CONFIRMATION or is_irreversible


_NON_NAMES = {
    "me", "you", "us", "them", "him", "her", "it",
    "my", "the", "a", "an", "this", "that",
}


def _extract_person(text: str) -> str:
    """Extract a person name from patterns like 'with Bassam', 'call Ali'."""
    patterns = [
        r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bcall\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bmessage\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bcontact\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bمع\s+([^\s،,\.]+)',
        r'\bل\s+([^\s،,\.]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            name = m.group(1).strip()
            if name.lower() not in _NON_NAMES:
                return name
    return ""


def _build_title(text: str, intent: str, person: str) -> str:
    """Build a concise, descriptive title from the user message."""
    lower = text.lower()

    if "follow up" in lower or "follow-up" in lower:
        base = "Follow-up"
    elif "call" in lower and intent in {CALENDAR_EVENT, REMINDER, FOLLOW_UP}:
        base = "Call"
    elif "remind" in lower:
        base = "Reminder"
    elif intent == DELETE_ACTION:
        base = "Delete"
    elif intent == MESSAGE_SEND:
        base = "Send"
    elif intent == MESSAGE_PERSON:
        base = "Message"
    elif intent == CALENDAR_EVENT:
        base = "Meeting"
    elif intent == RESEARCH_TASK:
        base = "Research"
    elif intent == CODING_TASK:
        base = "Task"
    else:
        base = "Task"

    if person:
        connector = "to" if intent in {MESSAGE_PERSON, MESSAGE_SEND} else "with"
        return f"{base} {connector} {person}"

    m = re.search(
        r'(?:about|re:|regarding|for|on)\s+([^,\.!\?]{3,35})',
        text, re.IGNORECASE,
    )
    if m:
        return f"{base} — {m.group(1).strip().rstrip('.,')[:30]}"

    return base


def _extract_due_at(text: str) -> str:
    """
    Extract a due_at ISO-8601 string (UTC) from natural language.
    "in X minutes/hours" is computed exactly here.
    Relative dates ("tomorrow", "next Monday") are approximate — exact resolution
    happens inside the executor via Claude.
    """
    lower = text.lower()
    now   = datetime.now(timezone.utc)

    # "in X minutes / hours" — always exact, always UTC
    dur_m = re.search(
        r'\bin\s+(\d+)\s+(minute|minutes|min|mins|hour|hours|hr|hrs)\b', lower
    )
    if dur_m:
        qty  = int(dur_m.group(1))
        unit = dur_m.group(2)
        delta = timedelta(hours=qty) if ("hour" in unit or "hr" in unit) else timedelta(minutes=qty)
        return (now + delta).isoformat()

    # "at H[:MM] [am/pm]"
    time_m = re.search(r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', lower)
    if not time_m:
        return ""

    hour   = int(time_m.group(1))
    minute = int(time_m.group(2) or 0)
    ampm   = (time_m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    if "tomorrow" in lower:
        target = (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
    else:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return target.isoformat()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(user_message: str, source: str = "telegram") -> TaskRecord:
    """Parse a user message into a TaskRecord (does NOT save it)."""
    now    = datetime.now(timezone.utc).isoformat()
    intent = _detect_intent(user_message)
    person = _extract_person(user_message)
    title  = _build_title(user_message, intent, person)
    due_at = _extract_due_at(user_message)

    requires_confirmation = intent in REQUIRES_CONFIRMATION
    status = PENDING_CONFIRMATION if requires_confirmation else SCHEDULED

    return TaskRecord(
        task_id               = str(uuid.uuid4()),
        source                = source,
        user_message          = user_message,
        intent                = intent,
        status                = status,
        title                 = title,
        due_at                = due_at,
        person                = person,
        requires_confirmation = requires_confirmation,
        created_at            = now,
        updated_at            = now,
    )


# ---------------------------------------------------------------------------
# Confirmation detection
# ---------------------------------------------------------------------------

def is_confirmation_text(text: str) -> bool:
    """True if the entire message is a confirmation word (yes/ok/confirm/etc.)."""
    return text.strip().lower() in _CONFIRM_WORDS


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_confirmation_request(record: TaskRecord) -> str:
    """Build the confirmation prompt for delete or send actions."""
    if record.intent == DELETE_ACTION:
        header  = "⚠️ *Confirm deletion*"
        cta     = "Are you sure? Reply **confirm** to delete, or ignore to cancel."
    elif record.intent == MESSAGE_SEND:
        header  = "📤 *Confirm send*"
        cta     = "Reply **confirm** to send, or ignore to cancel."
    else:
        header  = "⚠️ *Confirm action*"
        cta     = "Reply **confirm** to proceed, or ignore to cancel."

    lines = [f"{header}\n"]
    lines.append(f"**{record.title}**")

    if record.due_at:
        try:
            dt = datetime.fromisoformat(record.due_at)
            lines.append(f"🕐 {dt.strftime('%a %b %-d at %-I:%M %p')}")
        except Exception:
            lines.append(f"🕐 {record.due_at}")

    if record.person:
        lines.append(f"👤 {record.person}")

    lines.append(f'\n_"{record.user_message}"_')
    lines.append(f"Task ID: `{record.task_id[:8]}`")
    lines.append(f"\n{cta}")
    return "\n".join(lines)


def format_task_list(records: list[TaskRecord], header: str = "") -> str:
    """Format a task list for Telegram display."""
    if not records:
        return "No tasks found."
    lines = [header] if header else []
    for r in records[:20]:
        emoji = STATUS_EMOJI.get(r.status, "•")
        due   = f" · {r.due_at[:10]}" if r.due_at else ""
        lines.append(f"{emoji} `{r.task_id[:8]}` **{r.title}**{due}")
    if len(records) > 20:
        lines.append(f"_…and {len(records) - 20} more_")
    return "\n".join(lines)

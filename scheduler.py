"""
AXIS Scheduler — proactive task monitoring and execution.

Runs as a background daemon thread inside axis-api (started via FastAPI lifespan).
Polls /tmp/axis/tasks.jsonl every 60 seconds and:

  1. Fires due REMINDER / FOLLOW_UP tasks (safe) → Telegram message + mark completed
  2. Sends a heads-up for due CALENDAR_EVENT tasks → Telegram message + mark completed
  3. Re-notifies about PENDING_CONFIRMATION tasks older than 15 minutes

Safety constraints — never auto-executed by the scheduler:
  MESSAGE_TO_PERSON → skip; user must confirm through the normal flow
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import task_manager as _tm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_POLL_INTERVAL       = 60           # seconds between ticks
_PENDING_REMIND_SECS = 15 * 60      # re-notify pending tasks after 15 minutes

_TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN",   "")
_TELEGRAM_UID   = os.environ.get("TELEGRAM_USER_ID", "")

_DATA_DIR = Path(os.environ.get("AXIS_DATA_DIR", "/tmp/axis"))
_LOG_FILE = _DATA_DIR / "scheduler.log"

# task_id → monotonic timestamp of last pending reminder (in-memory only)
_pending_reminded: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(event: str, task_id: str, decision: str, detail: str = "") -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "event":    event,
            "task_id":  task_id,
            "decision": decision,
            "detail":   detail,
        }
        with _LOG_FILE.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def _send_telegram(text: str) -> bool:
    if not _TELEGRAM_TOKEN or not _TELEGRAM_UID:
        _log("telegram", "", "skipped", "TELEGRAM_TOKEN or TELEGRAM_USER_ID not configured")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": _TELEGRAM_UID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as exc:
        _log("telegram_error", "", "failed", str(exc))
        return False


# ---------------------------------------------------------------------------
# Task execution logic
# ---------------------------------------------------------------------------

# Intents safe to auto-execute (send notification, mark completed)
_AUTO_EXECUTE = {_tm.REMINDER, _tm.FOLLOW_UP}

# Intents that get a heads-up notification but no side effects
_NOTIFY_ONLY  = {_tm.CALENDAR_EVENT}

# Intents the scheduler never auto-executes — must go through user confirmation
_SKIP_ALWAYS  = {_tm.MESSAGE_PERSON, _tm.MESSAGE_SEND, _tm.DELETE_ACTION}


def _execute_due_task(task: _tm.TaskRecord) -> None:
    """Handle one due scheduled task according to its safety classification."""

    if task.intent in _SKIP_ALWAYS:
        _log("due_task", task.task_id, "skipped",
             f"intent={task.intent} requires user confirmation — never auto-executed")
        return

    if task.intent in _AUTO_EXECUTE:
        prefix = "📌 *Follow-up*" if task.intent == _tm.FOLLOW_UP else "🔔 *Reminder*"
        lines  = [f"{prefix}: {task.title}"]
        if task.person:
            lines.append(f"👤 {task.person}")
        ok     = _send_telegram("\n".join(lines))
        status = _tm.COMPLETED if ok else _tm.FAILED
        _tm.update(
            task.task_id,
            status=status,
            execution_result="Sent via scheduler" if ok else "Telegram delivery failed",
        )
        _log("due_task", task.task_id, "executed" if ok else "failed",
             f"intent={task.intent} title={task.title!r}")
        return

    if task.intent in _NOTIFY_ONLY:
        lines = [f"📅 *Event starting now:* {task.title}"]
        if task.person:
            lines.append(f"👤 {task.person}")
        ok = _send_telegram("\n".join(lines))
        _tm.update(task.task_id, status=_tm.COMPLETED,
                   execution_result="Event notification sent")
        _log("due_task", task.task_id, "notified",
             f"intent={task.intent} title={task.title!r}")
        return

    # Unknown / general intent with a due_at — send a generic reminder
    ok = _send_telegram(f"⏰ *Task due:* {task.title}")
    _tm.update(task.task_id, status=_tm.COMPLETED,
               execution_result="Generic due notification sent")
    _log("due_task", task.task_id, "notified_generic", f"title={task.title!r}")


def _remind_pending(task: _tm.TaskRecord) -> None:
    """Send a reminder that a task is still waiting for user confirmation."""
    text = (
        f"⏳ *Pending task awaiting your confirmation:*\n\n"
        f"*{task.title}*\n\n"
        f'_"{task.user_message}"_\n\n'
        f"Reply *confirm* (or yes / ok) to proceed, "
        f"or /cancel {task.task_id[:8]} to dismiss."
    )
    ok = _send_telegram(text)
    _log("pending_reminder", task.task_id, "sent" if ok else "failed",
         f"title={task.title!r}")


# ---------------------------------------------------------------------------
# Scheduler tick
# ---------------------------------------------------------------------------

def _tick() -> None:
    now      = datetime.now(timezone.utc)
    now_mono = time.monotonic()

    # ── 1. Execute scheduled tasks that are now due ──────────────────────
    scheduled = _tm.list_tasks(status_filter=_tm.SCHEDULED, limit=500)
    for task in scheduled:
        if not task.due_at:
            continue
        try:
            due = datetime.fromisoformat(task.due_at)
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due <= now:
                _execute_due_task(task)
        except Exception as exc:
            _log("tick_error", task.task_id, "error", f"due check: {exc}")

    # ── 2. Re-notify about long-pending confirmation tasks ───────────────
    pending = _tm.list_tasks(status_filter=_tm.PENDING_CONFIRMATION, limit=500)
    for task in pending:
        try:
            created = datetime.fromisoformat(task.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_secs      = (now - created).total_seconds()
            last_reminded = _pending_reminded.get(task.task_id, 0)
            secs_since    = now_mono - last_reminded

            if age_secs >= _PENDING_REMIND_SECS and secs_since >= _PENDING_REMIND_SECS:
                _remind_pending(task)
                _pending_reminded[task.task_id] = now_mono
        except Exception as exc:
            _log("tick_error", task.task_id, "error", f"pending check: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_forever() -> None:
    print("[scheduler] started — polling every 60s")
    _log("scheduler_lifecycle", "", "started", f"poll_interval={_POLL_INTERVAL}s")
    while True:
        try:
            _tick()
        except Exception as exc:
            _log("scheduler_lifecycle", "", "tick_error", str(exc))
        time.sleep(_POLL_INTERVAL)


def start() -> threading.Thread:
    """Launch the scheduler as a background daemon thread. Returns the thread."""
    t = threading.Thread(target=_run_forever, name="axis-scheduler", daemon=True)
    t.start()
    return t

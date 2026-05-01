"""
AXIS Chat UI — Chainlit interface for the AXIS autonomous agent.

Run with:
  chainlit run ~/AXIS-Managed/chat.py --port 8000

Full loop per message:
  user input → calendar intercept (read) → AXIS session (streamed) → executor (write) → memory
"""

import asyncio
import json
import os
import secrets
import subprocess
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import chainlit as cl
import requests as http_lib

from calendar_integration import CalendarService, GCAL_SCOPE

warnings.filterwarnings("ignore", category=Warning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent

# Writable runtime directory — /tmp on cloud (Render), repo dir locally.
_DATA_DIR = Path(os.environ.get("AXIS_DATA_DIR", str(HERE)))

AGENT_ID_FILE   = HERE / ".axis_agent_id"          # read-only source; env var takes priority
ENV_ID_FILE     = HERE / ".axis_env_id"
SESSION_ID_FILE = _DATA_DIR / ".axis_session_id"   # ephemeral per instance
TASKS_PATH      = _DATA_DIR / "tasks.md"
MEMORY_PATH     = _DATA_DIR / "memory.json"
CREDS_FILE      = HERE / "credentials.json"        # OAuth client config (local dev only)
TOKEN_FILE      = _DATA_DIR / "token.pickle"        # local dev fallback
BETA            = "managed-agents-2026-04-01"
EXECUTOR_MODEL  = "claude-sonnet-4-6"
GCAL_SCOPES     = [GCAL_SCOPE]
GCAL_REDIRECT   = os.environ.get(
    "GCAL_REDIRECT_URI", "http://localhost:8000/oauth2callback"
)

# API key — env var always wins over local key file.
api_key = os.environ.get("ANTHROPIC_API_KEY") or open(
    os.path.expanduser("~/.anthropic_key")
).read().strip()
client = anthropic.Anthropic(api_key=api_key)

# In-memory Google credential cache — avoids repeated env var parsing and disk I/O.
_gcal_creds_cache = None


# ---------------------------------------------------------------------------
# Google Calendar — middleware intercepts /oauth2callback BEFORE Chainlit's
# catch-all SPA route swallows it. Must be registered at module load time.
# ---------------------------------------------------------------------------

# Shared state between the middleware and waiting coroutines.
# threading primitives are used because the middleware may fire from any task.
_oauth_events:  dict[str, threading.Event] = {}   # state → Event
_oauth_results: dict[str, object]          = {}   # state → creds | Exception


async def _handle_oauth_callback(code: str, state: str, error: str):
    from fastapi.responses import HTMLResponse

    print(f"[AXIS Calendar] 1. callback received — path=/oauth2callback")
    print(f"[AXIS Calendar]    state={state[:10] if state else 'NONE'}…  error={error!r}")

    evt = _oauth_events.pop(state, None)

    if error:
        print(f"[AXIS Calendar]    ← OAuth denied by user: {error}")
        exc = RuntimeError(f"Google denied access: {error}")
        _oauth_results[state] = exc
        if evt:
            evt.set()
        return HTMLResponse(
            f"<h1>Authorization failed</h1><p>{error}</p><p>Close this tab and return to AXIS.</p>"
        )

    if evt is None:
        print(f"[AXIS Calendar]    ← unknown state — session may have expired")
        return HTMLResponse(
            "<h1>Unknown session.</h1><p>This link has expired. Ask AXIS for calendar access again.</p>"
        )

    if not code:
        print(f"[AXIS Calendar]    ← no authorization code in callback")
        exc = RuntimeError("No authorization code received from Google.")
        _oauth_results[state] = exc
        evt.set()
        return HTMLResponse("<h1>No code received.</h1><p>Please try again.</p>")

    print(f"[AXIS Calendar] 2. authorization code received — exchanging for token...")
    try:
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            str(CREDS_FILE), scopes=GCAL_SCOPES, state=state
        )
        flow.redirect_uri = GCAL_REDIRECT
        flow.fetch_token(code=code)
        creds = flow.credentials

        global _gcal_creds_cache
        _gcal_creds_cache = creds
        print(f"[AXIS Calendar] 3. credentials cached in memory")

        # Persist locally for dev restarts (skipped on cloud where GOOGLE_TOKEN_JSON is set).
        if not os.environ.get("GOOGLE_TOKEN_JSON"):
            import pickle
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            with TOKEN_FILE.open("wb") as f:
                pickle.dump(creds, f)
            print(f"[AXIS Calendar]    token.pickle saved → {TOKEN_FILE}")

        _oauth_results[state] = creds
        evt.set()

    except Exception as exc:
        print(f"[AXIS Calendar]    ← token exchange FAILED: {exc}")
        _oauth_results[state] = exc
        evt.set()
        return HTMLResponse(f"<h1>Token exchange failed</h1><p>{exc}</p>")

    return HTMLResponse(
        "<h1>Authorization complete!</h1>"
        "<p>Return to AXIS — your calendar events are loading.</p>"
    )


# Register as Starlette middleware so it runs before Chainlit's catch-all route.
from chainlit.server import app as _chainlit_app        # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request as _Request       # noqa: E402


class _GCalCallbackMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: _Request, call_next):
        if request.url.path == "/oauth2callback":
            code  = request.query_params.get("code", "")
            state = request.query_params.get("state", "")
            error = request.query_params.get("error", "")
            return await _handle_oauth_callback(code, state, error)
        return await call_next(request)


_chainlit_app.add_middleware(_GCalCallbackMiddleware)


# ---------------------------------------------------------------------------
# Google Calendar — credential helpers
# ---------------------------------------------------------------------------

# ── Calendar intent detection ─────────────────────────────────────────────
#
# READ  → intercept immediately, query Calendar API, return events.
# WRITE → pass to AXIS (it extracts details), executor creates the event.

_GCAL_WRITE_VERBS = {
    "schedule", "create", "add", "book", "set up", "arrange", "plan",
    "put on my calendar", "add to calendar", "add to my calendar",
    "جدول", "أضف", "احجز", "انشئ موعد",
}

_GCAL_READ_STRONG = {
    "what are my", "show my", "show me my", "list my", "what do i have",
    "what's on", "what is on", "am i busy", "do i have any",
    "upcoming events", "my events", "my calendar", "my schedule", "my agenda",
    "ما لدي", "أرني مواعيدي", "جدولي",
}

_GCAL_READ_TIME = {
    "today", "tonight", "tomorrow", "this week", "next week",
    "this morning", "this afternoon", "this evening",
    "اليوم", "غداً", "غدا", "هذا الأسبوع",
}

_GCAL_READ_NOUNS = {
    "calendar", "events", "appointments", "agenda", "meetings",
    "تقويم", "مواعيد", "رزنامة", "اجتماعات",
}


def _is_calendar_write_query(text: str) -> bool:
    """True when user wants to CREATE / SCHEDULE an event."""
    lower = text.lower()
    return any(v in lower for v in _GCAL_WRITE_VERBS)


def _is_calendar_read_query(text: str) -> bool:
    """True when user wants to READ / LIST events (and NOT create one)."""
    if _is_calendar_write_query(text):
        return False
    lower = text.lower()
    if any(kw in lower for kw in _GCAL_READ_STRONG):
        return True
    if any(kw in lower for kw in _GCAL_READ_NOUNS):
        return True
    has_time = any(tw in lower for tw in _GCAL_READ_TIME)
    has_q    = any(qw in lower for qw in _GCAL_READ_STRONG)
    return has_time and has_q


def _load_gcal_creds():
    """Sync: load credentials from cache → GOOGLE_TOKEN_JSON → token.pickle.
    Returns creds or None if OAuth is needed."""
    global _gcal_creds_cache
    from google.auth.transport.requests import Request

    # 1. In-memory cache (fastest path, covers cloud refreshes too).
    if _gcal_creds_cache is not None:
        if _gcal_creds_cache.valid:
            return _gcal_creds_cache
        if _gcal_creds_cache.expired and _gcal_creds_cache.refresh_token:
            print(f"[AXIS Calendar] 4. refreshing cached credentials...")
            _gcal_creds_cache.refresh(Request())
            return _gcal_creds_cache

    # 2. GOOGLE_TOKEN_JSON env var (cloud / pre-authorized).
    token_json = os.environ.get("GOOGLE_TOKEN_JSON", "")
    if token_json:
        print(f"[AXIS Calendar] 4. loading credentials from GOOGLE_TOKEN_JSON")
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(json.loads(token_json), GCAL_SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            print(f"[AXIS Calendar]    token expired — refreshing...")
            creds.refresh(Request())
        _gcal_creds_cache = creds
        return creds

    # 3. Local dev fallback: token.pickle.
    if TOKEN_FILE.exists():
        import pickle
        print(f"[AXIS Calendar] 4. loading credentials from token.pickle")
        with TOKEN_FILE.open("rb") as f:
            creds = pickle.load(f)

        # Scope upgrade check: old readonly token can't create events.
        token_scopes = set(getattr(creds, "scopes", []) or [])
        if GCAL_SCOPE not in token_scopes:
            print(f"[AXIS Calendar]    token has wrong scope ({token_scopes}) — re-auth required")
            return None

        if creds.valid:
            _gcal_creds_cache = creds
            return creds
        if creds.expired and creds.refresh_token:
            print(f"[AXIS Calendar]    token expired — refreshing...")
            creds.refresh(Request())
            with TOKEN_FILE.open("wb") as f:
                pickle.dump(creds, f)
            _gcal_creds_cache = creds
            return creds
        print(f"[AXIS Calendar]    token invalid — re-auth required")
    else:
        print(f"[AXIS Calendar] 4. no credentials found — OAuth required")
    return None


def _fetch_events_sync(n: int = 5) -> list[dict]:
    """Sync: read upcoming events via CalendarService."""
    creds = _load_gcal_creds()
    if creds is None:
        raise RuntimeError("Google Calendar not authorized.")
    cal   = CalendarService(creds=creds)
    items = cal.get_upcoming_events(days=7)[:n]
    print(f"[AXIS Calendar] 6. number of events found: {len(items)}")
    return items


def _format_events(events: list[dict]) -> str:
    if not events:
        return "**Source: Google Calendar** — No upcoming events in the next 7 days."
    return CalendarService.fmt_events(
        events, header="**Source: Google Calendar** — your next events:\n"
    )


async def _ensure_calendar_auth() -> bool:
    """Async: ensure token.pickle exists and has the right scope. Returns True if ready."""
    loop  = asyncio.get_running_loop()
    creds = await loop.run_in_executor(None, _load_gcal_creds)
    if creds is not None:
        return True

    # Need OAuth — generate auth URL and wait for the /oauth2callback middleware to fire.
    print(f"[AXIS Calendar] Starting OAuth — generating authorization URL")
    from google_auth_oauthlib.flow import Flow

    state = secrets.token_urlsafe(16)
    flow  = Flow.from_client_secrets_file(str(CREDS_FILE), scopes=GCAL_SCOPES)
    flow.redirect_uri = GCAL_REDIRECT
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )

    evt = threading.Event()
    _oauth_events[state] = evt

    await cl.Message(
        content=(
            "**Google Calendar authorization required.**\n\n"
            f"**[Click here to sign in with Google →]({auth_url})**\n\n"
            "_After signing in, your events will appear here automatically. "
            "Link expires in 3 minutes._"
        ),
        author="AXIS Calendar",
    ).send()

    # Wait without blocking the event loop — run Event.wait in the thread pool.
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, evt.wait, 180),
            timeout=185,
        )
    except asyncio.TimeoutError:
        _oauth_events.pop(state, None)
        _oauth_results.pop(state, None)
        raise RuntimeError("Google authorization timed out (3 min). Please try again.")

    result = _oauth_results.pop(state, None)
    if isinstance(result, Exception):
        raise result
    if result is None:
        raise RuntimeError("OAuth completed but no credentials were received.")

    print(f"[AXIS Calendar] OAuth complete — credentials received")
    return True


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, path: Path):
        self.path   = path
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
    agent_id = os.environ.get("AXIS_AGENT_ID", "")
    if agent_id:
        return agent_id
    if AGENT_ID_FILE.exists():
        return AGENT_ID_FILE.read_text().strip()
    raise RuntimeError("AXIS_AGENT_ID env var not set and .axis_agent_id not found.")


def _get_or_create_env() -> str:
    env_id = os.environ.get("AXIS_ENV_ID", "")
    if not env_id and ENV_ID_FILE.exists():
        env_id = ENV_ID_FILE.read_text().strip()
    if env_id:
        try:
            env = client.beta.environments.retrieve(env_id, betas=[BETA])
            if getattr(env, "state", None) == "active":
                return env_id
        except Exception:
            pass
    env = client.beta.environments.create(name="axis-env", betas=[BETA])
    env_id = env.id
    ENV_ID_FILE.write_text(env_id)
    return env_id


def _get_or_create_session(agent_id: str, env_id: str) -> str:
    sid = os.environ.get("AXIS_SESSION_ID", "")
    if not sid and SESSION_ID_FILE.exists():
        sid = SESSION_ID_FILE.read_text().strip()
    if sid:
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
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_ID_FILE.write_text(s.id)
    return s.id


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_create_calendar_event(
    title: str,
    start_iso: str,
    end_iso: str = "",
    description: str = "",
    location: str = "",
    **_,
) -> str:
    """Sync: create a Google Calendar event. Checks for conflicts first."""
    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt   = datetime.fromisoformat(end_iso) if end_iso else start_dt + timedelta(hours=1)

        creds = _load_gcal_creds()
        if creds is None:
            return "Error: Google Calendar not authorized. Ask the user to connect their calendar first."
        cal = CalendarService(creds=creds)

        conflicts = cal.check_conflicts(start_dt, end_dt)
        conflict_note = ""
        if conflicts:
            names = [c.get("summary", "Untitled") for c in conflicts[:2]]
            conflict_note = f" ⚠️ Conflict with: {', '.join(names)}"

        event    = cal.create_event(title, start_dt, end_dt, description, location)
        event_id = event.get("id", "")
        fmt_time = start_dt.strftime("%a %b %-d at %-I:%M %p")
        print(f"[AXIS Calendar] 5. calendar API request — CREATE")
        print(f"[AXIS Calendar] 6. event created: '{title}' on {fmt_time} [id:{event_id[:8]}]")
        return f"CALENDAR_EVENT_CREATED: '{title}' on {fmt_time}{conflict_note}"
    except Exception as exc:
        print(f"[AXIS Calendar]    create event FAILED: {exc}")
        return f"Error creating calendar event: {exc}"


def _tool_send_notification(title: str, message: str, **_) -> str:
    import re
    clean = lambda s: re.sub(r"[^\x00-\x7F]+", "", s).replace('"', '\\"').strip()
    t, m  = clean(title), clean(message)
    r = subprocess.run(
        ["osascript", "-e", f'display notification "{m}" with title "{t}"'],
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
    "create_calendar_event": _tool_create_calendar_event,
    "send_notification":     _tool_send_notification,
    "save_task":             _tool_save_task,
    "http_request":          _tool_http_request,
}

_EXECUTOR_TOOLS = [
    {
        "name": "create_calendar_event",
        "description": (
            "Create a Google Calendar event. Use whenever the user asks to schedule, "
            "book, or arrange a meeting, call, or appointment. "
            "Resolve relative dates ('tomorrow', 'next Monday') using today's date from the system prompt."
        ),
        "input_schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "title":       {"type": "string", "description": "Event title, e.g. 'Meeting with Engineer Ali — Villa Project'"},
                "start_iso":   {"type": "string", "description": "Start datetime in ISO 8601, e.g. '2026-05-02T10:00:00'"},
                "end_iso":     {"type": "string", "description": "End datetime in ISO 8601. Defaults to 1 hour after start."},
                "description": {"type": "string", "description": "Optional notes or agenda"},
                "location":    {"type": "string", "description": "Optional location"},
            },
            "required": ["title", "start_iso"],
        },
    },
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
        "description": "Save an action item to tasks.md.",
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

def _executor_system() -> str:
    now = datetime.now()
    return (
        f"You are the AXIS Execution Engine. Today is {now.strftime('%A, %B %-d, %Y')} "
        f"(local time {now.strftime('%-I:%M %p')}).\n"
        "Autonomously decide which real-world tools to call based on the user's input "
        "and AXIS's response. Be specific — extract names, times, and context.\n"
        "When creating calendar events, resolve relative dates (today, tomorrow, next Monday) "
        "using today's date above. Default event duration is 1 hour unless stated otherwise.\n"
        "Call no tools if nothing actionable is needed."
    )


# ---------------------------------------------------------------------------
# Async AXIS session streaming  (sync SDK bridged via queue)
# ---------------------------------------------------------------------------

async def _stream_axis_response(session_id: str, task: str,
                                cl_msg: cl.Message) -> str:
    loop:  asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue             = asyncio.Queue()
    chunks: list[str]                = []

    def _put(item):
        asyncio.run_coroutine_threadsafe(queue.put(item), loop)

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
                sender = threading.Thread(target=_send, daemon=True)
                sender.start()
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "agent.message":
                        for block in getattr(event, "content", []):
                            if getattr(block, "type", None) == "text":
                                _put(("chunk", block.text))
                    elif etype == "session.status_idle":
                        break
                    elif etype == "session.status_terminated":
                        SESSION_ID_FILE.unlink(missing_ok=True)
                        break
                sender.join()
        except Exception as exc:
            _put(("error", str(exc)))
        finally:
            _put(("done", None))

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
    results: list[dict] = []
    messages = [{
        "role":    "user",
        "content": f"User input:\n{task}\n\nAXIS response:\n{axis_response}",
    }]

    while True:
        resp = client.messages.create(
            model=EXECUTOR_MODEL,
            max_tokens=1024,
            system=_executor_system(),
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

        has_token    = bool(os.environ.get("GOOGLE_TOKEN_JSON")) or TOKEN_FILE.exists() or (_gcal_creds_cache is not None)
        token_status = "✅ Google Calendar configured" if has_token else "🔑 no token yet (will authorize on first calendar request)"
        mem_line     = (
            f"📚 **{len(memory.entries)} prior session(s)** in memory."
            if memory.entries else "🆕 No prior memory — fresh start."
        )
        await cl.Message(
            content=(
                f"## AXIS is ready\n\n"
                f"{mem_line}  \n"
                f"Google Calendar: {token_status}\n\n"
                f"Type anything — AXIS will respond and act autonomously."
            )
        ).send()

    except Exception as exc:
        await cl.ErrorMessage(content=f"Boot failed: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id: str         = cl.user_session.get("session_id")
    memory:     MemoryStore = cl.user_session.get("memory")

    if not session_id or memory is None:
        await cl.Message(
            content="⚠️ AXIS session not initialised. Refresh the page to reconnect."
        ).send()
        return

    task = message.content.strip()
    if not task:
        return

    ts = datetime.now(timezone.utc).isoformat()

    # ── 0. Google Calendar READ — intercept BEFORE AXIS sees the message ──
    if _is_calendar_read_query(task):
        print(f"[AXIS Calendar] calendar intent detected (READ) — '{task}'")
        try:
            await _ensure_calendar_auth()
            loop   = asyncio.get_running_loop()
            print(f"[AXIS Calendar] 5. calendar API request — READ")
            events   = await loop.run_in_executor(None, _fetch_events_sync, 5)
            cal_text = _format_events(events)
            outcome  = f"Google Calendar: {len(events)} event(s) fetched"
        except Exception as exc:
            print(f"[AXIS Calendar]    calendar API FAILED: {exc}")
            cal_text = (
                f"**Source: Google Calendar — FAILED**\n\n"
                f"Could not retrieve calendar data.\n\n"
                f"Error: `{exc}`"
            )
            outcome = f"Calendar error: {exc}"

        await cl.Message(content=cal_text, author="AXIS Calendar").send()
        memory.save_entry({
            "ts":      ts,
            "task":    task,
            "plan":    [],
            "steps":   [{"sub_task": "fetch_google_calendar", "results": [outcome]}],
            "outcome": outcome,
        })
        return  # Never pass read queries to AXIS

    # ── 1. Stream AXIS response ───────────────────────────────────────────
    mem_ctx    = memory.recent_context()
    axis_input = f"{task}\n\n{mem_ctx}" if mem_ctx else task

    axis_msg = cl.Message(content="", author="AXIS")
    await axis_msg.send()

    try:
        axis_response = await _stream_axis_response(session_id, axis_input, axis_msg)
    except Exception as exc:
        await axis_msg.stream_token(f"\n\n⚠️ Stream error: {exc}")
        axis_response = ""

    await axis_msg.update()

    if not axis_response:
        return

    # ── 2. Autonomous executor ────────────────────────────────────────────
    loop    = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run_executor_sync, task, axis_response)

    print(f"EXECUTOR RESULT: {results}")

    if results:
        async with cl.Step(name="⚙️ Executor", type="tool") as step:
            lines = []
            for r in results:
                icon = "✅" if r["ok"] else "❌"
                lines.append(f"{icon} **{r['tool']}** — {r['output']}")
            step.output = "\n".join(lines)

    # ── Calendar confirmation — executor result is the ONLY source of truth ──
    cal_results = [r for r in results if r["tool"] == "create_calendar_event"]
    if cal_results:
        output = cal_results[-1]["output"]
        # Success is determined solely by the sentinel the tool returns.
        if output.startswith("CALENDAR_EVENT_CREATED:"):
            detail  = output.replace("CALENDAR_EVENT_CREATED: ", "").strip()
            content = f"Event added to your Google Calendar ✅\n\n{detail}"
        else:
            # Explicit error text from the tool — show it as-is, no default message.
            content = f"Could not add event to Google Calendar ❌\n\n`{output}`"
        # Clear the AXIS message so its text never overrides the executor result.
        axis_msg.content = ""
        await axis_msg.update()
        await cl.Message(content=content, author="AXIS Calendar").send()

    # ── 3. Save to memory ─────────────────────────────────────────────────
    outcome = "; ".join(r["output"] for r in results) if results else "no action"
    memory.save_entry({
        "ts":      ts,
        "task":    task,
        "plan":    [],
        "steps":   [{"sub_task": task, "results": [r["output"] for r in results]}],
        "outcome": outcome,
    })

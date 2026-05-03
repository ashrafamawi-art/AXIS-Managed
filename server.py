"""
AXIS Server — REST API for the AXIS autonomous agent.

Endpoints:
  GET  /health  — liveness check
  POST /task    — submit a task; returns full AXIS response + artifacts

Pipeline per request:
  task → council (multi-perspective) → AXIS session → plan → execute → feedback

Environment variables:
  ANTHROPIC_API_KEY   (required)
  AXIS_AGENT_ID       (required)
  AXIS_ENV_ID         (optional, created if absent)
  AXIS_SESSION_ID     (optional, pin to existing session)
  AXIS_DATA_DIR       (default: directory of this file)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import council
import executor
import maestro
import scheduler
import task_manager as _tm

# ---------------------------------------------------------------------------
# Request-level dedup cache
# Prevents duplicate execution when Telegram retries a slow request or the
# same task is submitted twice within the same minute.
# ---------------------------------------------------------------------------

_REQUEST_CACHE: dict[str, tuple[dict, float]] = {}  # fingerprint → (result, epoch_s)
_REQUEST_CACHE_TTL = 60  # seconds


def _request_fingerprint(task: str) -> str:
    """SHA-256 of task text + current minute bucket."""
    minute = int(time.time() // 60)
    return hashlib.sha256(f"{task.strip()}|{minute}".encode()).hexdigest()[:16]


def _cache_get(fp: str) -> dict | None:
    now = time.time()
    # Prune expired entries
    for k in [k for k, (_, t) in _REQUEST_CACHE.items() if now - t > _REQUEST_CACHE_TTL]:
        del _REQUEST_CACHE[k]
    if fp in _REQUEST_CACHE:
        result, _ = _REQUEST_CACHE[fp]
        return result
    return None


def _cache_set(fp: str, result: dict) -> None:
    _REQUEST_CACHE[fp] = (result, time.time())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE      = Path(__file__).parent
_DATA_DIR = Path(os.environ.get("AXIS_DATA_DIR", str(HERE)))
BETA      = "managed-agents-2026-04-01"

AGENT_ID_FILE   = HERE / ".axis_agent_id"
ENV_ID_FILE     = HERE / ".axis_env_id"
SESSION_ID_FILE = _DATA_DIR / ".axis_session_id"

try:
    _key_file = os.path.expanduser("~/.anthropic_key")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or open(_key_file).read().strip()
except OSError:
    raise RuntimeError("ANTHROPIC_API_KEY env var not set and ~/.anthropic_key not found.")

client = anthropic.Anthropic(api_key=api_key)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    scheduler.start()
    yield


app = FastAPI(title="AXIS API", version="1.0", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# AXIS session management
# ---------------------------------------------------------------------------

def _load_agent_id() -> str:
    aid = os.environ.get("AXIS_AGENT_ID", "")
    if aid:
        return aid
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
    ENV_ID_FILE.write_text(env.id)
    return env.id


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


def _stream_axis(session_id: str, message: str) -> str:
    """Blocking: send message to AXIS session, return full response."""
    chunks: list[str] = []
    with client.beta.sessions.events.stream(session_id=session_id, betas=[BETA]) as stream:
        def _send():
            client.beta.sessions.events.send(
                session_id=session_id,
                events=[{
                    "type":    "user.message",
                    "content": [{"type": "text", "text": message}],
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
                        chunks.append(block.text)
            elif etype == "session.status_idle":
                break
            elif etype == "session.status_terminated":
                SESSION_ID_FILE.unlink(missing_ok=True)
                break
        t.join()
    return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# Task pipeline (blocking — runs in thread pool from async endpoint)
# ---------------------------------------------------------------------------

def _run_pipeline(task: str, request_id: str = "") -> dict:
    ts     = datetime.now(timezone.utc).isoformat()
    rid    = request_id or str(uuid.uuid4())[:8]
    prefix = f"[pipeline/{rid}]"
    print(f"{prefix} start: {task[:100]!r}")

    # 1. Load AXIS agent + session
    agent_id   = _load_agent_id()
    env_id     = _get_or_create_env()
    session_id = _get_or_create_session(agent_id, env_id)

    # 2. Council — multi-perspective pre-analysis
    council_result = council.run(task, client)
    enriched_input = council.format_for_axis(task, council_result)

    # 3. AXIS session — primary reasoning
    axis_response = _stream_axis(session_id, enriched_input)

    # 4. Plan — decompose into sub-tasks
    sub_tasks, reasoning = executor.plan(task, axis_response, "", client)
    print(f"{prefix} plan: {len(sub_tasks)} sub-tasks: {sub_tasks}")

    # 5. Execute each sub-task — pass request_id so executor can log it
    all_results: list[dict] = []
    for i, sub_task in enumerate(sub_tasks, 1):
        print(f"{prefix} executing sub-task {i}/{len(sub_tasks)}")
        results = executor.execute(sub_task, task, client, request_id=rid)
        all_results.extend(results)

    # 6. Feedback pass — close the loop with AXIS
    if all_results:
        summary = (
            f"Execution complete for: '{task}'\n\n"
            + "\n".join(f"• {r['tool']}: {r['output']}" for r in all_results)
            + "\n\nIs the goal fully achieved? What is the single most important next step?"
        )
        feedback = _stream_axis(session_id, summary)
        final_answer = feedback or axis_response
    else:
        final_answer = axis_response

    # 7. Build artifacts
    artifacts: dict = {}
    calendar_events = [r["output"] for r in all_results if r["tool"] == "create_calendar_event"]
    notifications   = [r["output"] for r in all_results if r["tool"] == "send_notification"]
    tasks_saved     = [r["output"] for r in all_results if r["tool"] == "save_task"]
    http_calls      = [r["output"] for r in all_results if r["tool"] == "http_request"]
    if calendar_events:
        artifacts["calendar"] = calendar_events
    if notifications:
        artifacts["notifications"] = notifications
    if tasks_saved:
        artifacts["tasks"] = tasks_saved
    if http_calls:
        artifacts["http"] = http_calls

    return {
        "id":        str(uuid.uuid4()),
        "task":      task,
        "status":    "done",
        "answer":    final_answer,
        "council": {
            "synthesis":          council_result.get("synthesis", ""),
            "recommended_action": council_result.get("recommended_action", ""),
        },
        "plan": {
            "sub_tasks": sub_tasks,
            "reasoning": reasoning,
        },
        "artifacts": artifacts,
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    task: str
    request_id: str = ""  # optional; generated if absent


@app.get("/health")
def health():
    return {"status": "ok", "service": "AXIS"}


@app.get("/tasks")
def get_tasks(status: str = None, limit: int = 50):
    records = _tm.list_tasks(status_filter=status, limit=limit)
    return {"tasks": [r.to_dict() for r in records], "count": len(records)}


@app.get("/tasks/pending")
def get_pending_tasks():
    records = _tm.list_tasks(status_filter=_tm.PENDING_CONFIRMATION, limit=20)
    return {
        "tasks": [r.to_dict() for r in records],
        "count": len(records),
        "formatted": _tm.format_task_list(records, "⏳ *Pending Confirmation*"),
    }


@app.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    ok = _tm.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return {"status": "cancelled", "task_id": task_id}


@app.get("/cal/diag")
def cal_diag():
    """
    Safe Google Calendar auth diagnostic.
    Outputs only: token_source, fields_present, client_id_prefix, scopes,
    expired, refresh_possible, calendar_test, calendar_diagnosis.
    Never exposes token, refresh_token, client_secret, or raw exception text.
    """
    import json as _json
    import re
    from datetime import datetime, timezone

    # Redact any string that looks like a token or secret (40+ contiguous
    # base64/URL-safe characters) before it can appear in error output.
    _REDACT = re.compile(r'[A-Za-z0-9+/=_\-]{40,}')

    def _safe_err(exc: Exception) -> str:
        sanitised = _REDACT.sub("[redacted]", str(exc))
        return f"{type(exc).__name__}: {sanitised[:120]}"

    out: dict = {}

    # ── 1. Token source ───────────────────────────────────────────────────
    token_json   = os.environ.get("GOOGLE_TOKEN_JSON", "")
    token_pickle = _DATA_DIR / "token.pickle"

    if token_json:
        out["token_source"] = "env"
    elif token_pickle.exists():
        out["token_source"] = "file"
    else:
        out["token_source"]       = "missing"
        out["calendar_test"]      = "failed"
        out["calendar_diagnosis"] = (
            "No token found. Set GOOGLE_TOKEN_JSON on Render or provide token.pickle."
        )
        return out

    # ── 2. Parse ──────────────────────────────────────────────────────────
    info  = None
    creds = None
    try:
        if token_json:
            info = _json.loads(token_json)
            _NEVER_SHOW = {"token", "refresh_token", "client_secret"}
            out["fields_present"] = sorted(k for k in info if k not in _NEVER_SHOW)
        else:
            import pickle
            with token_pickle.open("rb") as f:
                creds = pickle.load(f)
    except Exception as exc:
        out["calendar_test"]      = "failed"
        out["calendar_diagnosis"] = f"Token parse error: {type(exc).__name__}"
        return out

    # ── 3. client_id prefix — first 20 chars only ─────────────────────────
    raw_id = (info or {}).get("client_id", "") if info else getattr(creds, "client_id", "")
    out["client_id_prefix"] = (raw_id[:20] + "...") if len(raw_id) > 20 else raw_id

    # ── 4. Load Credentials object ────────────────────────────────────────
    if creds is None:
        try:
            from calendar_integration import GCAL_SCOPES
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_info(info, GCAL_SCOPES)
        except Exception as exc:
            out["scopes"]             = []
            out["expired"]            = None
            out["refresh_possible"]   = False
            out["calendar_test"]      = "failed"
            out["calendar_diagnosis"] = f"Credential load error: {_safe_err(exc)}"
            return out

    # ── 5. Token state ────────────────────────────────────────────────────
    out["scopes"]           = sorted(getattr(creds, "scopes", None) or [])
    out["expired"]          = creds.expired
    out["refresh_possible"] = bool(creds.refresh_token)

    # ── 6. Refresh if expired ─────────────────────────────────────────────
    if not creds.valid and creds.expired and creds.refresh_token:
        try:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        except Exception as exc:
            out["calendar_test"]      = "failed"
            out["calendar_diagnosis"] = f"Token refresh failed: {type(exc).__name__}"
            return out

    # ── 7. Live Calendar API test ─────────────────────────────────────────
    try:
        from googleapiclient.discovery import build
        svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
        svc.events().list(
            calendarId="primary",
            timeMin=datetime.now(timezone.utc).isoformat(),
            maxResults=1,
            singleEvents=True,
        ).execute()
        out["calendar_test"] = "ok"
    except Exception as exc:
        from calendar_integration import _interpret_error
        out["calendar_test"]      = "failed"
        out["calendar_diagnosis"] = _interpret_error(exc)

    return out


@app.get("/voice/diag")
def voice_diag():
    """
    Safe voice subsystem diagnostic.
    Returns only non-secret status flags — never exposes key values.
    """
    import subprocess as _sp
    import sys as _sys

    # ffmpeg availability
    try:
        ffmpeg_path = ""
        for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
            if Path(candidate).exists():
                ffmpeg_path = candidate
                break
        if not ffmpeg_path:
            try:
                import imageio_ffmpeg as _iff
                ffmpeg_path = _iff.get_ffmpeg_exe()
            except Exception:
                ffmpeg_path = "ffmpeg"
        _sp.run([ffmpeg_path, "-version"], capture_output=True, timeout=5, check=True)
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False

    # active backend
    dg_key_present = bool(os.environ.get("DEEPGRAM_API_KEY", "").strip())
    active_backend = "deepgram+whisper_fallback" if dg_key_present else "whisper_only"

    # commit / version
    try:
        commit = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    return {
        "deepgram_key_present":            dg_key_present,
        "voice_postprocess_with_claude":   os.environ.get("VOICE_POSTPROCESS_WITH_CLAUDE", "false"),
        "ffmpeg_available":                ffmpeg_ok,
        "python_version":                  _sys.version.split()[0],
        "active_voice_backend":            active_backend,
        "whisper_model":                   os.environ.get("WHISPER_MODEL", "small"),
        "timeout_voice_handler_s":         int(os.environ.get("VOICE_HANDLER_TIMEOUT", 60)),
        "timeout_deepgram_s":              int(os.environ.get("DEEPGRAM_TIMEOUT", 20)),
        "timeout_whisper_s":               int(os.environ.get("WHISPER_TIMEOUT", 30)),
        "app_commit":                      commit,
        "note": (
            "DEEPGRAM_API_KEY must be set in Render env vars for fast Arabic transcription. "
            "Without it, Whisper cold-load takes 40+ seconds and will hit the 30 s timeout."
            if not dg_key_present else "Deepgram configured — fast path active."
        ),
    }


@app.post("/task")
async def post_task(req: TaskRequest):
    task = req.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="task must not be empty")

    # ── Request-level dedup ────────────────────────────────────────────────
    fp = req.request_id.strip() if req.request_id.strip() else _request_fingerprint(task)
    cached = _cache_get(fp)
    if cached:
        print(f"[server] returning cached result for fp={fp} task={task[:60]!r}")
        return cached

    try:
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, maestro.run, task, client
        )
        result["request_id"] = fp
        _cache_set(fp, result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

"""
AXIS Memory — Supabase persistent backend with JSON fallback.

Table DDL (run once in Supabase SQL editor):

    create table if not exists axis_memory (
        id         uuid        primary key default gen_random_uuid(),
        topic      text        not null,
        content    text        not null,
        tags       text[]      default '{}',
        created_at timestamptz default now(),
        ttl_days   integer     default 30
    );

    create index if not exists axis_memory_created_at_idx
        on axis_memory (created_at desc);

Environment variables:
    SUPABASE_URL          Project URL, e.g. https://abc.supabase.co
    SUPABASE_SERVICE_KEY  Service-role key (secret) — never the anon key

If either variable is absent, or if the Supabase call fails, the module
falls back transparently to the legacy JSON file at $AXIS_DATA_DIR/memory.json.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Supabase client — lazy initialisation
# ---------------------------------------------------------------------------

_client     = None
_init_error: Optional[str] = None
_TABLE      = "axis_memory"


def _get_client():
    """Return a live Supabase client, or None if not configured / unavailable."""
    global _client, _init_error

    if _client is not None:
        return _client
    if _init_error is not None:
        return None

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        _init_error = "SUPABASE_URL or SUPABASE_SERVICE_KEY not configured"
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as exc:
        _init_error = str(exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "i", "me", "my", "you", "your", "we", "our",
    "this", "that", "there", "be", "do", "have", "will", "can",
}


def _extract_topic(task: str) -> str:
    return re.sub(r"\s+", " ", task.strip())[:60]


def _extract_tags(task: str) -> list:
    words = re.findall(r"\b[a-z]{3,}\b", task.lower())
    seen: set = set()
    tags: list = []
    for w in words:
        if w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            tags.append(w)
            if len(tags) >= 5:
                break
    return tags


# ---------------------------------------------------------------------------
# JSON fallback
# ---------------------------------------------------------------------------

_DATA_DIR    = Path(os.environ.get("AXIS_DATA_DIR", "/tmp/axis"))
_LEGACY_FILE = _DATA_DIR / "memory.json"


def _legacy_load(n: int) -> str:
    if not _LEGACY_FILE.exists():
        return ""
    try:
        entries = json.loads(_LEGACY_FILE.read_text()).get("entries", [])[-n:]
        lines   = ["[Recent memory]"]
        for e in entries:
            lines.append(f"• [{e['ts'][:10]}] {e.get('task', '')!r}")
            if e.get("outcome"):
                lines.append(f"  → {e['outcome'][:80]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _legacy_save(task: str, content: str) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        entries: list = []
        if _LEGACY_FILE.exists():
            entries = json.loads(_LEGACY_FILE.read_text()).get("entries", [])
        entries.append({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "task":    task,
            "outcome": content[:200],
        })
        _LEGACY_FILE.write_text(json.dumps({"entries": entries}, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_memory(n: int = 3) -> str:
    """
    Return a formatted context string of the n most recent non-expired memories.
    Falls back to the legacy JSON file if Supabase is unavailable.
    """
    db = _get_client()
    if db is None:
        return _legacy_load(n)

    try:
        # Over-fetch so TTL filtering in Python still yields n valid rows.
        resp = (
            db.table(_TABLE)
            .select("topic, content, created_at, ttl_days")
            .order("created_at", desc=True)
            .limit(n * 4)
            .execute()
        )
        rows = resp.data or []
        now  = datetime.now(timezone.utc)
        valid: list = []
        for row in rows:
            try:
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                ttl = int(row.get("ttl_days") or 30)
                if (now - created).days <= ttl:
                    valid.append(row)
                    if len(valid) >= n:
                        break
            except Exception:
                continue

        if not valid:
            return ""

        lines = ["[Recent memory]"]
        for row in valid:
            date_str = row["created_at"][:10]
            lines.append(f"• [{date_str}] {row['topic']!r}")
            if row.get("content"):
                lines.append(f"  → {row['content'][:80]}")
        return "\n".join(lines)

    except Exception:
        return _legacy_load(n)


def save_memory(task: str, content: str, ttl_days: int = 30) -> None:
    """
    Persist one memory entry.
    Falls back to the legacy JSON file if Supabase is unavailable.
    Never raises — memory failures must not break the main pipeline.
    """
    db = _get_client()
    if db is None:
        _legacy_save(task, content)
        return

    try:
        db.table(_TABLE).insert({
            "topic":    _extract_topic(task),
            "content":  content[:500],
            "tags":     _extract_tags(task),
            "ttl_days": ttl_days,
        }).execute()
    except Exception:
        _legacy_save(task, content)

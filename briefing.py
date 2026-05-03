"""
AXIS Daily Briefing — composes a morning Arabic summary delivered via Telegram.

Called by the scheduler every day at 7:00 AM Asia/Riyadh time.

Sections:
  1. Today's calendar events (with times in Riyadh local time)
  2. Notable events in the next 3 days worth preparing for
  3. Pending / unresolved tasks from the last 7 days
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

import memory_supabase as _mem
import task_manager as _tm
from calendar_integration import GCAL_SCOPES

# Asia/Riyadh = UTC+3, no DST
_RIYADH_OFFSET = timedelta(hours=3)

_ARABIC_DAYS = {
    0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء",
    3: "الخميس",  4: "الجمعة",   5: "السبت",   6: "الأحد",
}
_ARABIC_MONTHS = {
    1:  "يناير",   2:  "فبراير",  3:  "مارس",    4:  "أبريل",
    5:  "مايو",    6:  "يونيو",   7:  "يوليو",   8:  "أغسطس",
    9:  "سبتمبر", 10:  "أكتوبر", 11: "نوفمبر",  12: "ديسمبر",
}


def _riyadh_now() -> datetime:
    return datetime.now(timezone.utc) + _RIYADH_OFFSET


def _arabic_date(d) -> str:
    return f"{_ARABIC_DAYS[d.weekday()]}، {d.day} {_ARABIC_MONTHS[d.month]}"


def _fmt_riyadh_time(raw_start: str) -> str:
    """Return Arabic-formatted time 'H:MM AM/PM' for a dateTime string, or 'all day' label."""
    try:
        if "T" in raw_start:
            dt    = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
            local = dt + _RIYADH_OFFSET
            hour  = local.hour % 12 or 12
            ampm  = "ص" if local.hour < 12 else "م"
            return f"{hour}:{local.strftime('%M')} {ampm}"
        return "طوال اليوم"
    except Exception:
        return raw_start[:5]


# ---------------------------------------------------------------------------
# Google Calendar credential loading (mirrors executor._load_gcal_creds)
# ---------------------------------------------------------------------------

_gcal_creds_cache = None


def _load_gcal_creds():
    """Load Google Calendar creds exclusively from GOOGLE_TOKEN_JSON env var."""
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
        if not token_json:
            print("[briefing] GOOGLE_TOKEN_JSON not set — skipping calendar")
            return None

        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(json.loads(token_json), GCAL_SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        _gcal_creds_cache = creds
        return creds
    except Exception as exc:
        print(f"[briefing] gcal creds error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_calendar(today_riyadh) -> tuple[list[dict], list[dict]]:
    """Return (today_events, next_3_days_events) from Google Calendar."""
    creds = _load_gcal_creds()
    if creds is None:
        return [], []

    try:
        from calendar_integration import CalendarService
        all_events = CalendarService(creds=creds).get_upcoming_events(days=4)
    except Exception as exc:
        print(f"[briefing] calendar fetch error: {exc}")
        return [], []

    today_events:    list[dict] = []
    upcoming_events: list[dict] = []

    for ev in all_events:
        raw = ev["start"].get("dateTime") or ev["start"].get("date", "")
        try:
            if "T" in raw:
                dt      = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                ev_date = (dt + _RIYADH_OFFSET).date()
            else:
                ev_date = datetime.strptime(raw, "%Y-%m-%d").date()
        except Exception:
            continue

        if ev_date == today_riyadh:
            today_events.append(ev)
        elif today_riyadh < ev_date <= today_riyadh + timedelta(days=3):
            upcoming_events.append(ev)

    return today_events, upcoming_events


def _fetch_pending_tasks() -> list[_tm.TaskRecord]:
    """Tasks still pending or scheduled from the last 7 days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    seen:   set[str]             = set()
    out:    list[_tm.TaskRecord] = []
    for status in (_tm.PENDING_CONFIRMATION, _tm.SCHEDULED):
        for t in _tm.list_tasks(status_filter=status, limit=50):
            if t.task_id not in seen and t.created_at >= cutoff:
                seen.add(t.task_id)
                out.append(t)
    out.sort(key=lambda t: t.created_at, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Briefing composition
# ---------------------------------------------------------------------------

def _build_data_block(today_events, upcoming_events, pending_tasks,
                      today_riyadh, recent_memory) -> str:
    parts: list[str] = []

    # 1. Today's events
    if today_events:
        lines = []
        for ev in today_events:
            raw      = ev["start"].get("dateTime") or ev["start"].get("date", "")
            location = ev.get("location", "")
            loc      = f" — {location}" if location else ""
            lines.append(f"  • {_fmt_riyadh_time(raw)}: {ev.get('summary', '—')}{loc}")
        parts.append("أحداث اليوم:\n" + "\n".join(lines))
    else:
        parts.append("أحداث اليوم: لا توجد أحداث مجدولة.")

    # 2. Upcoming 3 days
    if upcoming_events:
        lines = []
        for ev in upcoming_events:
            raw = ev["start"].get("dateTime") or ev["start"].get("date", "")
            try:
                if "T" in raw:
                    dt      = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    ev_date = (dt + _RIYADH_OFFSET).date()
                else:
                    ev_date = datetime.strptime(raw, "%Y-%m-%d").date()
                day_str = _arabic_date(ev_date)
            except Exception:
                day_str = raw[:10]
            lines.append(f"  • {day_str}، {_fmt_riyadh_time(raw)}: {ev.get('summary', '—')}")
        parts.append("الأيام القادمة (3 أيام):\n" + "\n".join(lines))
    else:
        parts.append("الأيام القادمة: لا توجد أحداث.")

    # 3. Pending tasks
    if pending_tasks:
        now_utc = datetime.now(timezone.utc)
        lines   = []
        for t in pending_tasks[:8]:
            try:
                created  = datetime.fromisoformat(t.created_at)
                age_days = (now_utc - created).days
                age_str  = f"منذ {age_days} يوم" if age_days > 0 else "اليوم"
            except Exception:
                age_str = ""
            status_ar = {
                _tm.PENDING_CONFIRMATION: "في انتظار التأكيد",
                _tm.SCHEDULED:           "مجدول",
            }.get(t.status, t.status)
            lines.append(f"  • {t.title} ({status_ar}{', ' + age_str if age_str else ''})")
        parts.append("البنود المعلقة:\n" + "\n".join(lines))
    else:
        parts.append("البنود المعلقة: لا توجد بنود معلقة.")

    # 4. Recent memory context (optional enrichment)
    if recent_memory:
        parts.append(f"ذاكرة AXIS الأخيرة:\n{recent_memory}")

    return "\n\n".join(parts)


def _fallback_briefing(today_str: str, today_events: list,
                       upcoming_events: list, pending_tasks: list) -> str:
    lines = [f"🌅 *تقرير الصباح — {today_str}*\n"]
    lines.append("📅 *أحداث اليوم*")
    if today_events:
        for ev in today_events:
            raw = ev["start"].get("dateTime") or ev["start"].get("date", "")
            lines.append(f"• {_fmt_riyadh_time(raw)}: {ev.get('summary', '—')}")
    else:
        lines.append("• لا توجد أحداث")

    lines.append("\n📆 *الأيام القادمة*")
    if upcoming_events:
        for ev in upcoming_events[:5]:
            raw = ev["start"].get("dateTime") or ev["start"].get("date", "")
            lines.append(f"• {ev.get('summary', '—')} ({_fmt_riyadh_time(raw)})")
    else:
        lines.append("• لا توجد أحداث")

    lines.append("\n⏳ *بنود معلقة*")
    if pending_tasks:
        for t in pending_tasks[:5]:
            lines.append(f"• {t.title}")
    else:
        lines.append("• لا توجد بنود معلقة")

    return "\n".join(lines)


def compose_briefing() -> str:
    """
    Compose and return the daily briefing as a Telegram-ready Arabic Markdown string.
    Returns empty string on total failure.
    """
    now_riyadh  = _riyadh_now()
    today_date  = now_riyadh.date()
    today_str   = f"{_ARABIC_DAYS[now_riyadh.weekday()]}، {now_riyadh.day} {_ARABIC_MONTHS[now_riyadh.month]} {now_riyadh.year}"

    today_events, upcoming_events = _fetch_calendar(today_date)
    pending_tasks  = _fetch_pending_tasks()
    recent_memory  = _mem.load_memory(n=7)

    data_block = _build_data_block(
        today_events, upcoming_events, pending_tasks, today_date, recent_memory
    )

    prompt = (
        f"اليوم: {today_str}\n\n"
        f"البيانات المتاحة:\n{data_block}\n\n"
        f"اكتب تقرير الصباح لأشرف باللغة العربية. يجب أن يكون:\n"
        f"- يبدأ بـ 🌅 *تقرير الصباح — {today_str}*\n"
        f"- مقسماً بأقسام واضحة مع رموز تعبيرية (📅 اليوم، 📆 القادمة، ⏳ المعلق)\n"
        f"- يركز على ما هو مهم وما يحتاج تحضيراً\n"
        f"- ينتهي بعبارة تشجيعية قصيرة\n"
        f"- لا يتجاوز 35 سطراً\n"
        f"- منسق لـ Telegram Markdown (bold بـ *text*)\n"
    )

    try:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"[briefing] Claude compose error: {exc}")
        return _fallback_briefing(today_str, today_events, upcoming_events, pending_tasks)

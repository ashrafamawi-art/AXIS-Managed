"""
AXIS Google Calendar Integration.

Provides CalendarService — read, create, update, delete events.
Security: token contents are never logged or printed.

If you get a 403 error, one of these is the cause:
  1. Google Calendar API is not enabled in your GCP project.
     Fix: https://console.cloud.google.com/apis/library/calendar-json.googleapis.com
  2. The stored token was authorized with narrower scopes than GCAL_SCOPES.
     Fix: run  python3 check_calendar.py --reauth
  3. The token has been revoked or expired with no refresh token.
     Fix: run  python3 check_calendar.py --reauth
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_DUBAI_TZ      = ZoneInfo("Asia/Dubai")
_DUBAI_TZ_NAME = "Asia/Dubai"

# Full access + explicit readonly — ensures the consent screen requests both.
# If the old token only had calendar.events, a --reauth run is required.
GCAL_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Keep the old single-scope name as an alias so other modules don't break.
GCAL_SCOPE = GCAL_SCOPES[0]


class CalendarService:
    """Google Calendar API wrapper. Accepts pre-loaded credentials or a token.pickle path."""

    def __init__(self, token_file: Path = None, creds=None):
        self._token_file = token_file
        self._creds      = creds  # pre-loaded google.oauth2.credentials.Credentials

    # ------------------------------------------------------------------
    # Credential management
    # ------------------------------------------------------------------

    def _get_creds(self):
        if self._creds is not None:
            if self._creds.valid:
                _check_token_scopes(self._creds)
                return self._creds
            if self._creds.expired and self._creds.refresh_token:
                from google.auth.transport.requests import Request
                self._creds.refresh(Request())
            return self._creds

        # Fallback: load from token.pickle (local dev only)
        if self._token_file is None or not self._token_file.exists():
            raise PermissionError(
                "Google Calendar token not found.\n"
                "Run:  python3 check_calendar.py --reauth"
            )
        import pickle
        with self._token_file.open("rb") as f:
            creds = pickle.load(f)
        _check_token_scopes(creds)
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            with self._token_file.open("wb") as f:
                pickle.dump(creds, f)
        self._creds = creds
        return creds

    def _svc(self):
        from googleapiclient.discovery import build
        return build("calendar", "v3", credentials=self._get_creds(), cache_discovery=False)

    # ------------------------------------------------------------------
    # Auth check
    # ------------------------------------------------------------------

    @classmethod
    def check_auth(cls, token_file: Path = None, creds=None) -> dict:
        """
        Verify that Google Calendar credentials are valid and have the
        required scopes.  Returns a status dict:
          { "ok": bool, "scopes": list, "error": str|None }
        Does NOT raise — safe to call from health-check endpoints.
        """
        try:
            svc = cls(token_file=token_file, creds=creds)
            c   = svc._get_creds()
            token_scopes = list(getattr(c, "scopes", None) or [])
            # Use events.list — works with calendar.events scope and above.
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            svc._svc().events().list(
                calendarId="primary", timeMin=now, maxResults=1, singleEvents=True,
            ).execute()
            return {"ok": True, "scopes": token_scopes, "error": None}
        except Exception as exc:
            msg = _interpret_error(exc)
            return {"ok": False, "scopes": [], "error": msg}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_upcoming_events(self, days: int = 7) -> list[dict]:
        """Return events in the next `days` days."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        result = (
            self._svc().events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )
        return result.get("items", [])

    # ------------------------------------------------------------------
    # Conflict check
    # ------------------------------------------------------------------

    def check_conflicts(self, start: datetime, end: datetime) -> list[dict]:
        """Return events that overlap the given window."""
        start = _ensure_tz(start)
        end   = _ensure_tz(end)
        result = (
            self._svc().events().list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )
        return result.get("items", [])

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        description: str = "",
        location: str = "",
    ) -> dict:
        """Create a calendar event. Returns the created event resource."""
        start = _ensure_tz(start)
        end   = _ensure_tz(end)
        body: dict = {
            "summary": title,
            "start":   {"dateTime": start.isoformat(), "timeZone": _DUBAI_TZ_NAME},
            "end":     {"dateTime": end.isoformat(),   "timeZone": _DUBAI_TZ_NAME},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        return self._svc().events().insert(calendarId="primary", body=body).execute()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_event(self, event_id: str, **fields) -> dict:
        """Patch an existing event. `fields` match Google Calendar event keys."""
        return (
            self._svc().events()
            .patch(calendarId="primary", eventId=event_id, body=fields)
            .execute()
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_event(self, event_id: str) -> bool:
        """Delete an event by its ID."""
        self._svc().events().delete(calendarId="primary", eventId=event_id).execute()
        return True

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_event(ev: dict) -> str:
        start = ev["start"].get("dateTime", ev["start"].get("date", ""))
        try:
            dt    = datetime.fromisoformat(start.replace("Z", "+00:00"))
            start = dt.strftime("%a %b %-d, %Y at %-I:%M %p")
        except Exception:
            pass
        title    = ev.get("summary", "(No title)")
        location = ev.get("location", "")
        loc_str  = f"\n  📍 {location}" if location else ""
        return f"**{title}**  \n  🕐 {start}{loc_str}"

    @classmethod
    def fmt_events(cls, events: list[dict], header: str = "") -> str:
        if not events:
            return "No events found."
        lines = [header] if header else []
        for ev in events:
            lines.append(f"- {cls.fmt_event(ev)}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ensure_tz(dt: datetime) -> datetime:
    """Attach Asia/Dubai timezone to naive datetimes. Never uses system default."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_DUBAI_TZ)
    return dt


def _check_token_scopes(creds) -> None:
    """Raise PermissionError if the token is missing all Calendar scopes."""
    token_scopes = set(getattr(creds, "scopes", None) or [])
    if not token_scopes:
        return  # scopes not embedded in this token type — skip check
    # Any of these scopes is sufficient for reading and writing events.
    sufficient = {
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    }
    if not token_scopes.intersection(sufficient):
        raise PermissionError(
            f"Google Calendar token has insufficient scopes.\n"
            f"  Token has : {token_scopes}\n"
            f"  Need any of: {sufficient}\n"
            "Fix: python3 check_calendar.py --reauth"
        )


def _interpret_error(exc: Exception) -> str:
    """Turn a Google API exception into a human-readable diagnosis. Never returns raw secrets."""
    import re
    msg = str(exc)
    if "403" in msg or "forbidden" in msg.lower():
        return (
            "403 Forbidden — two possible causes:\n"
            "  1. Google Calendar API not enabled in your GCP project.\n"
            "     Enable it: https://console.cloud.google.com/apis/library/calendar-json.googleapis.com\n"
            "  2. Token was authorized with insufficient scopes.\n"
            "     Fix: python3 check_calendar.py --reauth"
        )
    if "401" in msg or "unauthorized" in msg.lower() or "invalid_grant" in msg.lower():
        return (
            "401 Unauthorized — token expired or revoked.\n"
            "Fix: python3 check_calendar.py --reauth"
        )
    if "token not found" in msg.lower() or "not available" in msg.lower():
        return (
            "No Google Calendar token found.\n"
            "Fix: python3 check_calendar.py --reauth"
        )
    # Fallback: redact any long token-like strings before returning
    safe = re.sub(r'[A-Za-z0-9+/=_\-]{40,}', '[redacted]', msg)
    return f"{type(exc).__name__}: {safe[:200]}"

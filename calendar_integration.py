"""
AXIS Google Calendar Integration.

Provides CalendarService — read, create, update, delete events.
Scope: calendar.events (events only, not calendar settings).
Security: token contents are never logged or printed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

GCAL_SCOPE = "https://www.googleapis.com/auth/calendar.events"


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
                return self._creds
            if self._creds.expired and self._creds.refresh_token:
                from google.auth.transport.requests import Request
                self._creds.refresh(Request())
            return self._creds

        # Fallback: load from token.pickle (local dev only)
        if self._token_file is None or not self._token_file.exists():
            raise FileNotFoundError(
                "Google Calendar credentials not available. "
                "Ask AXIS to authorize Google Calendar first."
            )
        import pickle
        with self._token_file.open("rb") as f:
            creds = pickle.load(f)
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
        tz    = _tz_name(start)
        body: dict = {
            "summary": title,
            "start":   {"dateTime": start.isoformat(), "timeZone": tz},
            "end":     {"dateTime": end.isoformat(),   "timeZone": tz},
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
    if dt.tzinfo is None:
        return dt.astimezone()
    return dt


def _tz_name(dt: datetime) -> str:
    try:
        return str(dt.tzinfo)
    except Exception:
        return "UTC"

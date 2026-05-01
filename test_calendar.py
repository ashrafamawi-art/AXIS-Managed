#!/usr/bin/env python3
"""
Standalone Google Calendar API test.
Run from ~/AXIS-Managed:
  python3 test_calendar.py
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

HERE       = Path(__file__).parent
CREDS_FILE = HERE / "credentials.json"
TOKEN_FILE = HERE / "token.pickle"
SCOPES     = ["https://www.googleapis.com/auth/calendar.events"]
REDIRECT   = "http://localhost:8000/oauth2callback"


def get_creds():
    print(f"[test] credentials.json: {CREDS_FILE}")
    assert CREDS_FILE.exists(), f"MISSING: {CREDS_FILE}"
    print(f"[test] credentials.json found: YES")

    creds = None
    if TOKEN_FILE.exists():
        print(f"[test] token.pickle found: YES")
        with TOKEN_FILE.open("rb") as f:
            creds = pickle.load(f)
        if creds.valid:
            print(f"[test] token valid: YES")
            return creds
        if creds.expired and creds.refresh_token:
            print(f"[test] token expired — refreshing...")
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            with TOKEN_FILE.open("wb") as f:
                pickle.dump(creds, f)
            print(f"[test] token refreshed and saved")
            return creds
        print(f"[test] token invalid — will re-run OAuth")
    else:
        print(f"[test] token.pickle found: NO — running OAuth...")

    # Run OAuth with a local server (test only, separate port)
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=8081, open_browser=True)
    with TOKEN_FILE.open("wb") as f:
        pickle.dump(creds, f)
    print(f"[test] token saved to {TOKEN_FILE}")
    return creds


def fetch_events(creds, n=10):
    from googleapiclient.discovery import build
    service = build("calendar", "v3", credentials=creds)
    now = datetime.now(timezone.utc).isoformat()
    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=n,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return result.get("items", [])


def main():
    print("=" * 50)
    print("AXIS Google Calendar API Test")
    print("=" * 50)

    creds  = get_creds()
    print(f"\n[test] calling Google Calendar API...")
    events = fetch_events(creds)
    print(f"[test] API call SUCCESS — {len(events)} event(s) found\n")

    if not events:
        print("No upcoming events.")
        return

    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date", ""))
        try:
            from datetime import datetime, timezone
            dt    = datetime.fromisoformat(start.replace("Z", "+00:00"))
            start = dt.strftime("%a %b %-d, %Y at %-I:%M %p")
        except Exception:
            pass
        title    = ev.get("summary", "(No title)")
        location = ev.get("location", "")
        loc_str  = f"  @ {location}" if location else ""
        print(f"  • {title}{loc_str}")
        print(f"    {start}")

    print(f"\n[test] DONE — {len(events)} event(s) printed")


if __name__ == "__main__":
    main()

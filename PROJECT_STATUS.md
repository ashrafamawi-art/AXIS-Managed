# AXIS Project Status

## Architecture

```
User
  │
  ▼
POST /task  (server.py)
  │
  ▼
Maestro  (maestro.py)
  │
  ├─► Security Layer (security.py) ──► BLOCK if HIGH risk
  │
  ├─► Intent Classifier ──► calendar | task | memory | general
  │
  ├─► Agent Router
  │     ├─ calendar_agent → general_agent
  │     ├─ task_agent     → executor._save_task()
  │     ├─ memory_agent   → memory.json read
  │     └─ general_agent  → Council → AXIS Session → Plan → Execute
  │
  ├─► Security veto per sub-task (inspect_action)
  │
  └─► Unified response
```

---

## Security Layer (`security.py`)

Security has **veto power** over all agents. No action executes without passing inspection.

### Two inspection points

| Point | Function | When |
|---|---|---|
| Prompt gate | `inspect_prompt(text)` | Before any processing begins |
| Action gate | `inspect_action(type, details)` | Before each sub-task executes |

### Risk levels

| Level | Meaning | Action |
|---|---|---|
| `LOW` | No threats detected | Proceed normally |
| `MEDIUM` | Potentially sensitive (API calls, file access, deploys) | Route to `general_agent` for full oversight |
| `HIGH` | Injection, destructive, or secret-exposure attempt | Block immediately, return reason |

### Hard block list (HIGH risk — auto-blocked)

| Category | Examples |
|---|---|
| `prompt_injection` | "Ignore previous instructions", "Forget your rules", "Developer mode" |
| `jailbreak` | "You are now unrestricted", "Bypass safety guidelines", "Act as if you have no constraints" |
| `expose_secrets` | "Show all env vars", "Print API keys", "Reveal system prompt", `os.environ[` |
| `delete_files` | `rm -rf`, "Delete all files", `shutil.rmtree`, `unlink(` |
| `push_to_main` | "Push code to main branch", "git push origin main", "Force push", `git push --force` |
| `modify_system_files` | Writes to `/etc/`, `/usr/`, `/bin/`, `/boot/`, `/sys/`, `system32` |

### Medium-risk patterns (MEDIUM — flagged, not blocked)

| Category | Examples |
|---|---|
| `external_api` | "Make an external API call", "Send an HTTP request" |
| `file_system` | "Write to the file system", "Access the database" |
| `code_execution` | "Execute a shell script", `subprocess.run(` |
| `deploy` | "Deploy to production", "Publish to staging" |
| `data_deletion` | "Drop table", "Delete all records" |

### Action veto rules (`inspect_action`)

Called before each sub-task. Blocks on: file deletion, push-to-main, secret exposure, system file modification. External HTTP requests require details to start with `APPROVED:`.

### Log format

Every security decision is appended to `/tmp/axis/security.log` (or `$AXIS_DATA_DIR/security.log`):
```json
{"action": "inspect_prompt", "decision": "BLOCKED", "reason": "...", "preview": "...", "timestamp": "2026-05-02T..."}
```

---

## Agent Routing (`maestro.py`)

### Intent classification

Uses `claude-haiku-4-5-20251001` to classify every task into one of four intents:

| Intent | Triggers | Agent |
|---|---|---|
| `calendar` | scheduling, meetings, events, appointments | `calendar_agent` → `general_agent` |
| `task` | to-do items, action items, reminders | `task_agent` |
| `memory` | "remember", "recall", "what did I tell you" | `memory_agent` |
| `general` | everything else | `general_agent` |

MEDIUM-risk tasks always override intent classification and route to `general_agent`.

### Agent implementations

**`task_agent`**: Calls `executor._save_task()` directly. Fast path for simple to-do capture.

**`memory_agent`**: Reads last 3 entries from `memory.json`. Returns formatted context.

**`calendar_agent`**: Thin wrapper — delegates to `general_agent` (calendar tool execution happens inside the executor loop).

**`general_agent`**: Full pipeline — `council.run()` → AXIS session stream → `executor.plan()` → `executor.execute()` per sub-task (with per-sub-task security veto) → AXIS feedback pass.

### Response format

```json
{
  "id":        "uuid",
  "task":      "original task",
  "status":    "done | blocked",
  "answer":    "AXIS full response",
  "security":  { "risk": "LOW | MEDIUM | HIGH", "reason": "..." },
  "routing":   { "intent": "general | calendar | task | memory | blocked" },
  "council":   { "synthesis": "...", "recommended_action": "..." },
  "plan":      { "sub_tasks": [...], "reasoning": "..." },
  "artifacts": { "tasks": [...], "notifications": [...] },
  "timestamp": "ISO 8601"
}
```

---

## Services

| Service | File | Render type | Description |
|---|---|---|---|
| `axis-api` | `server.py` | Web | REST API — POST /task |
| `axis-chat` | `chat.py` | Web | Chainlit UI |
| `axis-telegram` | `telegram_bot.py` | Worker | Telegram voice + text bot |

### Required environment variables

**axis-api / axis-chat:**
- `ANTHROPIC_API_KEY`
- `AXIS_AGENT_ID`
- `AXIS_ENV_ID`
- `AXIS_DATA_DIR` = `/tmp/axis`

**axis-api only:**
- `AXIS_SESSION_ID` (optional)
- `GOOGLE_TOKEN_JSON` — required for calendar creation via Telegram/REST. Generate with:
  `python3 -c "import pickle,json; c=pickle.load(open('token.pickle','rb')); print(c.to_json())"`
- `TELEGRAM_TOKEN` (for HIGH-risk security alerts)
- `TELEGRAM_USER_ID` (admin ID to receive alerts)

**axis-chat only:**
- `GOOGLE_TOKEN_JSON`
- `GCAL_REDIRECT_URI`

**axis-telegram:**
- `ANTHROPIC_API_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_USER_ID`
- `AXIS_API_URL` (e.g. `https://axis-api.onrender.com/task`)
- `WHISPER_MODEL` = `small`

---

## Runtime Architecture Requirement

**All production execution must be cloud-based (Render).**

| Service | Runtime | Notes |
|---|---|---|
| `axis-api` | Render (cloud) | REST API — handles ALL calendar/task execution |
| `axis-chat` | Render (cloud) | Chainlit UI — has its own calendar path |
| `axis-telegram` | Render (cloud) | Forwards to axis-api |

**Local services (laptop/terminal) are for development and testing only.**  
Production must never depend on:
- The developer laptop being online
- Claude Code / local terminal running
- `token.pickle` on disk (use `GOOGLE_TOKEN_JSON` env var on Render)
- `*.axis.local`, `localhost`, or `127.0.0.1` service URLs

### Calendar execution path (Telegram → Render)

```
Telegram message
  → telegram_bot.py (Render worker)
  → POST axis-api.onrender.com/task
  → server.py → maestro.run() → calendar_agent() → general_agent()
  → executor.execute()  ← create_calendar_event tool lives HERE
  → _create_calendar_event()
      ├─ loads GOOGLE_TOKEN_JSON from env  ← cloud credential
      ├─ calls Google Calendar API directly from Render
      └─ on failure: saves pending task + returns clear error message
```

If `GOOGLE_TOKEN_JSON` is not set on Render, the executor returns:
> "Google Calendar credentials are not configured on Render. Task saved as pending."

---

---

## Proactive Mode (`scheduler.py`)

AXIS runs a background scheduler that monitors tasks and fires reminders autonomously.

### How it works

- Starts as a **daemon thread** inside `axis-api` at startup (FastAPI lifespan)
- Polls `/tmp/axis/tasks.jsonl` every **60 seconds**
- Two checks per tick:
  1. **Due tasks** — any `status=scheduled` task whose `due_at ≤ now`
  2. **Stale pending** — any `status=pending_confirmation` task older than 15 minutes

### Safety model

| Intent | Scheduler action |
|---|---|
| `reminder` | ✅ Auto-execute — sends Telegram message, marks `completed` |
| `follow_up` | ✅ Auto-execute — sends Telegram message, marks `completed` |
| `calendar_event` | 📣 Notify-only — sends "event starting now", marks `completed` |
| `message_to_person` | 🚫 Never auto-executed — user must confirm |
| `general_question`, `research_task`, `coding_task` | Generic due notification if due_at set |

### Reminder flow (end-to-end)

```
User: "Remind me to call Bassam in 1 minute"
  → task_manager.parse()  extracts due_at = now + 1min
  → maestro proposal gate: REMINDER with due_at → defer to scheduler
  → Response: "⏰ Got it! I'll remind you: Call with Bassam at 3:01 PM"
  → task saved: status=scheduled, due_at=<1 min from now>
  ... 60 seconds later ...
  → scheduler._tick() fires
  → due_at ≤ now  →  _execute_due_task()
  → Telegram: "🔔 Reminder: Call with Bassam"
  → task updated: status=completed
```

### Pending reminder flow

```
User: "Add meeting with Bassam tomorrow at 3pm"
  → maestro: requires_confirmation → shows proposal card
  → task saved: status=pending_confirmation
  ... 15 minutes pass, user hasn't replied ...
  → scheduler._tick() → age ≥ 15min → _remind_pending()
  → Telegram: "⏳ Pending task awaiting your confirmation: Meeting with Bassam ..."
```

### Logging

Every scheduler decision is appended to `/tmp/axis/scheduler.log`:
```json
{"ts": "2026-05-02T...", "event": "due_task", "task_id": "...", "decision": "executed", "detail": "intent=reminder title='Call with Bassam'"}
```

---

## Confirmation Policy v2 — minimal friction

**Principle:** AXIS should feel instant. Confirmation is rare and reserved for irreversible harm only.

### When confirmation IS required

| Trigger | Intent | Example |
|---|---|---|
| Delete anything | `delete_action` | "Delete my meeting tomorrow" |
| Send to someone | `message_send` | "Send this email to Bassam" |
| Explicit irreversible | `is_irreversible=True` | Programmatic flag |

### When confirmation is NOT required

Everything else executes immediately:
- Create/update calendar events
- Reminders and follow-ups
- Drafting messages ("Write a message to Bassam")
- Research, summarization, analysis
- Memory read/write
- Task organization

### Implementation

```python
requires_confirmation(action_type, is_irreversible=False)
# Returns True only if action_type in {"delete_action", "message_send"} or is_irreversible
```

Detection priority in `_detect_intent()`:
1. `_DELETE_KW` → `delete_action` ("delete", "remove", "cancel", "erase")
2. `_DRAFT_KW` → `message_to_person` ("draft", "compose") — overrides "email" in send set
3. `_SEND_KW` → `message_send` ("send", "email", "forward")
4. Everything else — existing priority order

### Confirmation prompt format

- Delete: `⚠️ Confirm deletion — Are you sure? Reply confirm to delete...`
- Send: `📤 Confirm send — Reply confirm to send...`
- No generic "AXIS Task Proposal" for non-critical actions

---

## Task Manager (`task_manager.py`)

Structured task lifecycle: parse → (optionally confirm) → scheduled → completed/failed/cancelled.

### Intents

| Intent | Requires confirmation | Persisted | Auto-executable by scheduler |
|---|---|---|---|
| `calendar_event` | ✅ | ✅ | 📣 notify-only |
| `reminder` | ❌ | ✅ | ✅ auto-execute |
| `follow_up` | ✅ | ✅ | ✅ auto-execute |
| `message_to_person` | ✅ | ✅ | 🚫 never |
| `research_task` | ❌ | ❌ | — |
| `coding_task` | ❌ | ❌ | — |
| `general_question` | ❌ | ❌ | — |

### Confirmation flow

```
User: "Add meeting with Bassam tomorrow at 3pm"
  → maestro parses → requires_confirmation=True
  → saves task (status=pending_confirmation)
  → returns: "📋 AXIS Task Proposal ... Reply confirm to execute"

User: "confirm"  (or yes / ok / go ahead)
  → maestro confirmation handler finds latest pending task
  → re-runs original message through full pipeline with __SKIP_CONFIRM__ prefix
  → updates task: status=completed / failed
```

### Storage

Tasks are stored as JSON lines in `$AXIS_DATA_DIR/tasks.jsonl` (default `/tmp/axis/tasks.jsonl`).

**Note:** On Render, `axis-api` and `axis-telegram` are separate containers — they do not share `/tmp/axis/`. All task read/write must go through the `axis-api` REST endpoints (`GET /tasks`, `GET /tasks/pending`, `DELETE /tasks/{task_id}`).

### Telegram commands

| Command | Action |
|---|---|
| `/tasks` | List last 20 tasks with status |
| `/pending` | List tasks awaiting confirmation |
| `/cancel <id>` | Cancel a task by its 8-char ID prefix |

---

## Daily Briefing (`briefing.py`)

Every morning at **7:00 AM Asia/Riyadh time** AXIS sends an automatic Arabic Telegram summary.

### Content

| Section | Source |
|---|---|
| 📅 Today's calendar events (with local times) | Google Calendar — today's events |
| 📆 Next 3 days (notable events worth preparing for) | Google Calendar — tomorrow → +3 days |
| ⏳ Pending / unresolved items | `tasks.jsonl` — `pending_confirmation` + `scheduled` tasks from last 7 days |
| 🧠 Recent memory context (optional enrichment) | Supabase `axis_memory` — last 7 entries |

### How it works

- The scheduler calls `_check_daily_briefing()` on every 60s tick
- Fires once per process lifetime when `riyadh_hour == 7` and today's date hasn't been sent yet
- `briefing.compose_briefing()` fetches all data, builds a structured Arabic prompt, calls Haiku
- Falls back to a plain-text Arabic briefing if Claude fails
- Sent via the existing `_send_telegram()` helper in `scheduler.py`
- Requires the same credentials as the rest of `axis-api`: `GOOGLE_TOKEN_JSON`, `TELEGRAM_TOKEN`, `TELEGRAM_USER_ID`

### Timezone

Asia/Riyadh = UTC+3, no DST — computed as `datetime.now(UTC) + timedelta(hours=3)`.

### Log entries

```json
{"ts": "...", "event": "briefing", "task_id": "", "decision": "sent", "detail": "date=2026-05-02"}
```

---

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI server — POST /task, GET/DELETE /tasks, starts scheduler |
| `maestro.py` | Orchestration — security + intent routing + agent dispatch + task proposal gate |
| `scheduler.py` | Background scheduler — proactive reminders and follow-ups |
| `task_manager.py` | Task lifecycle — parse, save, update, confirm, list |
| `security.py` | Security layer — prompt inspection, risk classification, action veto |
| `council.py` | Multi-perspective pre-analysis (5 lenses) |
| `executor.py` | Tool planning + execution (create_calendar_event, save_task, notifications, HTTP) |
| `chat.py` | Chainlit web UI with Google Calendar integration |
| `telegram_bot.py` | Telegram bot — forwards to axis-api via HTTP |
| `session_runner.py` | CLI autonomous loop (local development) |
| `calendar_integration.py` | Google Calendar API wrapper |
| `briefing.py` | Daily morning briefing — compose + send Arabic Telegram summary |

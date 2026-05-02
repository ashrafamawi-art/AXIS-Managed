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

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI server — POST /task routes through Maestro |
| `maestro.py` | Orchestration — security + intent routing + agent dispatch |
| `security.py` | Security layer — prompt inspection, risk classification, action veto |
| `council.py` | Multi-perspective pre-analysis (5 lenses) |
| `executor.py` | Tool planning + execution (save_task, notifications, HTTP) |
| `chat.py` | Chainlit web UI with Google Calendar integration |
| `telegram_bot.py` | Telegram bot — forwards to axis-api via HTTP |
| `session_runner.py` | CLI autonomous loop (local development) |
| `calendar_integration.py` | Google Calendar API wrapper |

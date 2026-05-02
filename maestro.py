"""
AXIS Maestro — task orchestration layer.

Every request flows through:
  task
    → security.inspect_prompt()          ← BLOCK if HIGH risk
    → classify intent                    ← haiku classifier
    → detect agents needed               ← keyword-based multi-agent detection
    → run agent(s)                       ← one or many in parallel
    → merge results                      ← unified answer + artifacts
    → return unified response

Single-agent requests behave exactly as before.
Multi-agent requests run all needed agents and merge their outputs.
"""

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests

import security
import council
import executor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_DIR    = Path(os.environ.get("AXIS_DATA_DIR", str(Path(__file__).parent)))
_MEMORY_FILE = _DATA_DIR / "memory.json"

_INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["calendar", "task", "memory", "general"],
        },
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["intent", "confidence", "reason"],
}

_INTENT_SYSTEM = """\
You are the AXIS Intent Classifier. Classify the request into exactly one category:
- calendar : scheduling, meetings, events, appointments, availability checks
- task     : to-do items, action items, reminders, follow-ups, things to do later
- memory   : "remember", "recall", "what did I tell you", storing/retrieving facts
- general  : everything else — questions, analysis, writing, advice, projects

Return JSON only.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_intent(task: str, client: anthropic.Anthropic) -> str:
    """Returns one of: calendar | task | memory | general."""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": f"Request: {task}"}],
            output_config={"format": {"type": "json_schema", "schema": _INTENT_SCHEMA}},
        )
        data = json.loads(resp.content[0].text)
        return data.get("intent", "general")
    except Exception:
        return "general"


def _send_security_alert(task: str, reason: str, category: str) -> None:
    """Fire-and-forget Telegram alert to admin when a HIGH-risk request is blocked."""
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    user_id = os.environ.get("TELEGRAM_USER_ID", "")
    if not token or not user_id:
        return
    text = (
        f"🚨 *AXIS Security Alert*\n\n"
        f"*Category:* `{category}`\n"
        f"*Reason:* {reason}\n\n"
        f"*Preview:* `{task[:120]}`"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": user_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception:
        pass


def _load_memory(n: int = 3) -> str:
    if not _MEMORY_FILE.exists():
        return ""
    try:
        entries = json.loads(_MEMORY_FILE.read_text()).get("entries", [])[-n:]
        lines   = ["[Recent memory]"]
        for e in entries:
            lines.append(f"• [{e['ts'][:10]}] {e.get('task','')!r}")
            if e.get("outcome"):
                lines.append(f"  → {e['outcome'][:80]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _save_memory(task: str, outcome: str) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        entries = []
        if _MEMORY_FILE.exists():
            entries = json.loads(_MEMORY_FILE.read_text()).get("entries", [])
        entries.append({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "task":    task,
            "outcome": outcome[:200],
        })
        _MEMORY_FILE.write_text(json.dumps({"entries": entries}, indent=2))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def task_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Save an action item directly via executor."""
    check = security.inspect_action("save_task", task)
    if not check["allowed"]:
        return {"answer": f"Blocked by security: {check['reason']}", "artifacts": {}}

    result = executor._save_task(task=task, priority="medium")
    return {
        "answer":    f"Task saved ✅\n\n{task}",
        "artifacts": {"tasks": [result]},
    }


def memory_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Retrieve recent memory context."""
    ctx = _load_memory()
    if ctx:
        return {"answer": ctx, "artifacts": {"memory": ctx}}
    return {"answer": "No prior memory found.", "artifacts": {}}


def calendar_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Calendar tasks go through general_agent — the executor handles calendar tools."""
    return general_agent(task, client)


def general_agent(task: str, client: anthropic.Anthropic) -> dict:
    """
    Full AXIS pipeline: council → AXIS session → plan → execute (with veto) → feedback.
    Uses server._run_pipeline via late import to avoid circular dependency.
    """
    # Late import — by call time server.py is fully loaded.
    import server as srv

    # Run the full pipeline
    raw = srv._run_pipeline(task)

    # Extract core fields; maestro.run() adds the security/routing wrapper.
    return {
        "answer":    raw.get("answer", ""),
        "artifacts": raw.get("artifacts", {}),
        "council":   raw.get("council", {}),
        "plan":      raw.get("plan", {}),
    }


# Research Agent — web search capability
def research_agent(task: str, client: anthropic.Anthropic) -> dict:
    """Search the web and return a summarized answer in Arabic."""
    try:
        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=(
                "أنت مساعد بحث. ابحث في الإنترنت عن المعلومات المطلوبة "
                "وقدم إجابة واضحة ومختصرة باللغة العربية."
            ),
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": task}],
            betas=["web-search-2025-03-05"],
        )
        answer = " ".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        ).strip()
        return {"answer": answer or "(no results)", "artifacts": {}}
    except Exception as exc:
        return {"answer": f"Research failed: {exc}", "artifacts": {}}


# ---------------------------------------------------------------------------
# GitHub Developer Agent infrastructure
# ---------------------------------------------------------------------------

_GH_BASE   = "https://api.github.com"
_GH_OWNER  = "ashrafamawi-art"
_GH_BRANCH = "ai-dev"                    # all writes go here — never main
_GH_LOG    = _DATA_DIR / "github_agent.log"

# Reject file content that looks like a hardcoded secret before any commit.
_SECRET_RE = re.compile(
    r"(?:api[_\-]?key|secret(?:[_\-]key)?|password|auth[_\-]?token|private[_\-]?key"
    r"|ANTHROPIC_API_KEY|TELEGRAM_TOKEN|GITHUB_TOKEN|access_token)"
    r"\s*[=:]\s*['\"][A-Za-z0-9+/=_\-\.]{10,}['\"]",
    re.IGNORECASE,
)

# Tool schemas exposed to Claude inside github_agent.
_GH_TOOLS = [
    {
        "name": "list_repos",
        "description": "List all repositories under the ashrafamawi-art GitHub account.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_files",
        "description": "List files and subdirectories at a path inside a repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":   {"type": "string", "description": "Repository name"},
                "path":   {"type": "string", "description": "Directory path (empty string for root)"},
                "branch": {"type": "string", "description": "Branch to read from (default: main)"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full content of a file from any branch of any repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":   {"type": "string", "description": "Repository name"},
                "path":   {"type": "string", "description": "File path inside the repo"},
                "branch": {"type": "string", "description": "Branch to read from (default: main)"},
            },
            "required": ["repo", "path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or update a file on the ai-dev branch. "
            "ONLY write to ai-dev — never to main or any other branch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":    {"type": "string", "description": "Repository name"},
                "path":    {"type": "string", "description": "File path inside the repo"},
                "content": {"type": "string", "description": "Full new file content"},
                "message": {"type": "string", "description": "Commit message describing the change"},
            },
            "required": ["repo", "path", "content", "message"],
        },
    },
    {
        "name": "create_pull_request",
        "description": (
            "Open a pull request from ai-dev → main for Ashraf to review. "
            "Always call this after writing files — never auto-merge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":  {"type": "string", "description": "Repository name"},
                "title": {"type": "string", "description": "Short PR title"},
                "body":  {"type": "string", "description": "PR description: what changed and why"},
            },
            "required": ["repo", "title", "body"],
        },
    },
]


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    return {
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_log(action: str, detail: str) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "action": action, "detail": detail}
        with _GH_LOG.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _gh_ensure_ai_dev(repo: str) -> None:
    """Create the ai-dev branch from main if it does not already exist."""
    headers = _gh_headers()
    url     = f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/git/refs/heads/{_GH_BRANCH}"

    if requests.get(url, headers=headers, timeout=10).status_code == 200:
        return  # already exists

    # Resolve main HEAD
    r = requests.get(
        f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/git/refs/heads/main",
        headers=headers, timeout=10,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Cannot read main branch in {repo}: {r.text}")
    main_sha = r.json()["object"]["sha"]

    r = requests.post(
        f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{_GH_BRANCH}", "sha": main_sha},
        timeout=10,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Cannot create {_GH_BRANCH} in {repo}: {r.text}")
    _gh_log("create_branch", f"{repo}/{_GH_BRANCH} from main@{main_sha[:7]}")


def _gh_dispatch(name: str, inp: dict) -> str:
    """Execute one GitHub tool call; return a JSON string result."""
    headers = _gh_headers()

    if name == "list_repos":
        r = requests.get(
            f"{_GH_BASE}/users/{_GH_OWNER}/repos?per_page=50&type=all",
            headers=headers, timeout=10,
        )
        repos = [x["name"] for x in r.json()] if r.status_code == 200 else []
        _gh_log("list_repos", str(repos))
        return json.dumps({"repos": repos})

    if name == "list_files":
        repo   = inp["repo"]
        path   = inp.get("path", "")
        branch = inp.get("branch", "main")
        r = requests.get(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/{path}?ref={branch}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return json.dumps({"error": r.text})
        items = [{"name": x["name"], "type": x["type"], "path": x["path"]} for x in r.json()]
        _gh_log("list_files", f"{repo}/{path}@{branch} → {len(items)} items")
        return json.dumps({"files": items})

    if name == "read_file":
        repo   = inp["repo"]
        path   = inp["path"]
        branch = inp.get("branch", "main")
        r = requests.get(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/{path}?ref={branch}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return json.dumps({"error": r.text})
        data    = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        _gh_log("read_file", f"{repo}/{path}@{branch} ({len(content)} chars)")
        return json.dumps({"content": content, "sha": data["sha"]})

    if name == "write_file":
        repo    = inp["repo"]
        path    = inp["path"]
        content = inp["content"]
        message = inp.get("message", "chore: AXIS ai-dev update")

        # Safety: reject content that looks like it contains hardcoded secrets
        if _SECRET_RE.search(content):
            _gh_log("write_file_BLOCKED", f"{repo}/{path} — secret pattern detected")
            return json.dumps({"error": "Write blocked: content appears to contain hardcoded secrets."})

        try:
            _gh_ensure_ai_dev(repo)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        # Fetch existing file SHA (required by GitHub API to update)
        r       = requests.get(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/{path}?ref={_GH_BRANCH}",
            headers=headers, timeout=10,
        )
        payload: dict = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch":  _GH_BRANCH,
        }
        if r.status_code == 200:
            payload["sha"] = r.json()["sha"]

        r = requests.put(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/{path}",
            headers=headers, json=payload, timeout=15,
        )
        if r.status_code not in (200, 201):
            return json.dumps({"error": r.text})
        _gh_log("write_file", f"{repo}/{path}@{_GH_BRANCH} — {message}")
        return json.dumps({"status": "ok", "branch": _GH_BRANCH, "path": path})

    if name == "create_pull_request":
        repo  = inp["repo"]
        title = inp.get("title", "AXIS: Proposed changes")
        body  = inp.get("body", "Automated PR from AXIS GitHub Agent — please review before merging.")
        r = requests.post(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/pulls",
            headers=headers,
            json={"title": title, "body": body, "head": _GH_BRANCH, "base": "main"},
            timeout=15,
        )
        if r.status_code not in (200, 201):
            return json.dumps({"error": r.text})
        pr_url = r.json().get("html_url", "")
        _gh_log("create_pull_request", f"{repo} → {pr_url}")
        return json.dumps({"status": "ok", "pr_url": pr_url})

    return json.dumps({"error": f"Unknown tool: {name}"})


# File extensions we are willing to fetch for analysis.
_GH_TEXT_EXTS = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".js", ".ts", ".sh", ".toml"}


def _gh_fetch_for_task(task: str) -> dict:
    """
    Proactively fetch relevant files from GitHub REST API based on the task text.
    Called before Claude runs so reading is always done via API, never from local disk.

    Returns:
        {
            "repo":    str,           # resolved repo name
            "context": str,           # formatted file content to inject into the prompt
            "error":   str | None,    # non-None if the API call failed
        }
    """
    headers = _gh_headers()

    # ── 1. Resolve which repo is being discussed ──────────────────────────
    repo = "AXIS-Managed"   # default
    # Look for explicit mentions: "AXIS-Managed", "AXIS", or any word that matches a repo name
    known_repos_r = requests.get(
        f"{_GH_BASE}/users/{_GH_OWNER}/repos?per_page=50&type=all",
        headers=headers, timeout=10,
    )
    known_repos: list[str] = []
    if known_repos_r.status_code == 200:
        known_repos = [x["name"] for x in known_repos_r.json()]
    for name in sorted(known_repos, key=len, reverse=True):   # longest match first
        if name.lower() in task.lower():
            repo = name
            break

    # ── 2. List root files in the repo ───────────────────────────────────
    r = requests.get(
        f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/",
        headers=headers, timeout=10,
    )
    if r.status_code != 200:
        return {"repo": repo, "context": "", "error": f"Cannot list {repo}: {r.status_code} {r.text[:200]}"}

    all_entries = r.json()
    text_files  = [e["name"] for e in all_entries if e["type"] == "file"
                   and Path(e["name"]).suffix.lower() in _GH_TEXT_EXTS]
    _gh_log("list_files", f"{repo}/ → {text_files}")

    # ── 3. Detect which specific files the task is asking about ──────────
    task_lower = task.lower()
    to_read: list[str] = []

    # Explicit filename with extension (e.g. "maestro.py", "requirements.txt")
    for fname in text_files:
        if fname.lower() in task_lower:
            to_read.append(fname)

    # Bare name without extension (e.g. "maestro", "security")
    if not to_read:
        for fname in text_files:
            stem = Path(fname).stem.lower()
            if stem and stem in task_lower:
                to_read.append(fname)

    # No specific file → read the first few Python files (general review)
    if not to_read:
        to_read = [f for f in text_files if f.endswith(".py")][:3]

    # ── 4. Fetch each file from GitHub API ───────────────────────────────
    sections = [f"[GitHub: {_GH_OWNER}/{repo} — fetched via REST API]",
                f"Files in repo: {', '.join(text_files)}"]

    for fname in to_read[:4]:   # max 4 files to stay within token limits
        r = requests.get(
            f"{_GH_BASE}/repos/{_GH_OWNER}/{repo}/contents/{fname}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            sections.append(f"\n--- {fname} (fetch failed: {r.status_code}) ---")
            continue
        raw = base64.b64decode(r.json()["content"]).decode("utf-8", errors="replace")
        # Truncate very large files to avoid blowing token budget
        if len(raw) > 10_000:
            raw = raw[:10_000] + "\n... [truncated at 10 000 chars]"
        sections.append(f"\n--- {fname} ({len(raw)} chars) ---\n{raw}")
        _gh_log("read_file", f"{repo}/{fname} for analysis")

    return {"repo": repo, "context": "\n\n".join(sections), "error": None}


# GitHub Developer Agent — reads via API only
def github_agent(task: str, client: anthropic.Anthropic) -> dict:
    import requests, base64, os
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = "https://api.github.com/repos/ashrafamawi-art/AXIS-Managed/contents/maestro.py"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return {"answer": f"GitHub API error: {r.status_code}", "artifacts": {}}
    content = base64.b64decode(r.json()["content"]).decode("utf-8")
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": f"راجع هذا الكود واقترح تحسينات بالعربي:\n\n{content[:8000]}"}]
    )
    return {"answer": resp.content[0].text, "artifacts": {}}


_AGENT_MAP = {
    "calendar": calendar_agent,
    "task":     task_agent,
    "memory":   memory_agent,
    "general":  general_agent,
    "research": research_agent,
    "github":   github_agent,
}

# Keywords that signal each domain is needed.
# Substring match on lowercased task text — order doesn't matter.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "calendar": ["calendar", "schedule", "meeting", "event", "appointment", "availability"],
    "task":     ["task", "to-do", "todo", "remind", "reminder", "follow-up", "action item"],
    "memory":   ["remember", "recall", "what did i tell", "told you", "from memory"],
    "research": [
        # Arabic
        "ابحث", "بحث", "اخبار", "اخبرني عن", "شو اخر", "ما هو", "معلومات",
        # English
        "search", "find", "what is", "latest", "news",
    ],
    "github": [
        "github", "repo", "branch", "pull request", "pr", "open pr",
        "merge", "commit", "push", "maestro", "axis-managed",
    ],
}


def _detect_agents(task: str, intent: str) -> list:
    """
    Return the list of agent functions needed for this task.

    Strategy:
      1. Check the lowercased task text against each domain's keywords.
      2. If two or more domains match → multi-agent: return one function per domain.
      3. If exactly one domain matches → use that agent directly (bypasses intent
         classifier — necessary for "research" which the classifier doesn't know about).
      4. If no domains match → fall back to _AGENT_MAP[intent] from the classifier.
    """
    lower = task.lower()

    matched_domains = [
        domain
        for domain, keywords in _DOMAIN_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]

    # Deduplicate while preserving match order
    seen: set[str] = set()
    unique_domains: list[str] = []
    for d in matched_domains:
        if d not in seen:
            seen.add(d)
            unique_domains.append(d)

    if len(unique_domains) >= 2:
        # Multi-agent: run every matched domain
        return [_AGENT_MAP[d] for d in unique_domains]

    if len(unique_domains) == 1:
        # Single keyword match — trust it over the intent classifier
        return [_AGENT_MAP[unique_domains[0]]]

    # No keyword match — use the intent classifier's verdict
    return [_AGENT_MAP.get(intent, general_agent)]


def _merge_results(named_results: list[tuple[str, dict]]) -> dict:
    """
    Merge outputs from multiple agents into one result dict.

    named_results: list of (agent_name, agent_result_dict)

    - Answers are joined under bold section headers.
    - Artifact lists are concatenated; scalar values are overwritten last-wins.
    - council/plan are taken from the first result that carries them.
    """
    sections:          list[str] = []
    merged_artifacts:  dict      = {}
    extras:            dict      = {}   # council, plan — first-wins

    for name, result in named_results:
        answer = (result.get("answer") or "").strip()
        if answer:
            label = name.replace("_", " ").title()
            sections.append(f"**{label}**\n{answer}")

        for key, val in result.get("artifacts", {}).items():
            if isinstance(val, list):
                merged_artifacts.setdefault(key, []).extend(val)
            else:
                merged_artifacts[key] = val

        for key in ("council", "plan"):
            if key not in extras and key in result:
                extras[key] = result[key]

    merged = {
        "answer":    "\n\n---\n\n".join(sections) or "(no response)",
        "artifacts": merged_artifacts,
    }
    merged.update(extras)
    return merged

# ---------------------------------------------------------------------------
# Maestro — main entry point
# ---------------------------------------------------------------------------

def run(task: str, client: anthropic.Anthropic) -> dict:
    """
    Orchestrate a task through the full security + routing pipeline.

    Returns a response dict compatible with POST /task:
    {
        "id", "task", "status", "agent", "answer",
        "security": {"risk", "reason"},
        "routing":  {"intent", "agents": [...]},   # agents list length > 1 → multi-agent run
        "council":  {...},   # only for general/calendar
        "plan":     {...},   # only for general/calendar
        "artifacts": {...},
        "timestamp"
    }
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── 0. Confirm bypass — "confirm <task>" skips MEDIUM block ──────────
    confirmed = task.lower().startswith("confirm ")
    actual_task = task[len("confirm "):].strip() if confirmed else task

    # ── 1. Security: inspect prompt ───────────────────────────────────────
    sec      = security.inspect_prompt(actual_task)
    category = sec.get("category", "unknown")

    if sec["blocked"] or sec["risk"] == security.HIGH:
        _send_security_alert(actual_task, sec["reason"], category)
        return {
            "id":        str(uuid.uuid4()),
            "task":      actual_task,
            "status":    "blocked",
            "reason":    category,
            "message":   "🚫 Request denied due to security policy.",
            "security":  {"risk": sec["risk"], "reason": sec["reason"]},
            "routing":   {"intent": "blocked"},
            "artifacts": {},
            "timestamp": ts,
        }

    # ── 2. Classify intent ────────────────────────────────────────────────
    intent = _classify_intent(actual_task, client)

    # MEDIUM risk without confirm → hold for user confirmation
    if sec["risk"] == security.MEDIUM and not confirmed:
        return {
            "id":        str(uuid.uuid4()),
            "task":      actual_task,
            "status":    "needs_confirmation",
            "reason":    category,
            "message":   "⚠️ This action requires your confirmation. Reply 'confirm' to proceed.",
            "security":  {"risk": sec["risk"], "reason": sec["reason"]},
            "routing":   {"intent": intent},
            "artifacts": {},
            "timestamp": ts,
        }

    # ── 3. Detect agents and route ────────────────────────────────────────
    agents = _detect_agents(actual_task, intent)

    if len(agents) == 1:
        # Single-agent path — identical to original behavior
        agent_fn     = agents[0]
        agent_result = agent_fn(actual_task, client)
        agent_names  = [agent_fn.__name__]
    else:
        # Multi-agent path — run all detected agents, then merge
        named = [(fn.__name__, fn(actual_task, client)) for fn in agents]
        agent_result = _merge_results(named)
        agent_names  = [name for name, _ in named]

    # ── 4. Persist to memory ──────────────────────────────────────────────
    _save_memory(actual_task, agent_result.get("answer", ""))

    # ── 5. Build unified response ─────────────────────────────────────────
    response: dict = {
        "id":        str(uuid.uuid4()),
        "task":      actual_task,
        "status":    "done",
        "agent":     ", ".join(agent_names),
        "answer":    agent_result.get("answer", "(no response)"),
        "security":  {"risk": sec["risk"], "reason": sec["reason"]},
        "routing":   {"intent": intent, "agents": agent_names},
        "artifacts": agent_result.get("artifacts", {}),
        "timestamp": ts,
    }

    for key in ("council", "plan"):
        if key in agent_result:
            response[key] = agent_result[key]

    return response

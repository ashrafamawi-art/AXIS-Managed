"""
AXIS Security Layer — prompt inspection, risk classification, action veto.

Every task passes through security before execution.
Every action passes through security before it runs.
Security has VETO POWER over all agents.

Risk levels:
  LOW    — proceed normally
  MEDIUM — flag, route to full oversight pipeline
  HIGH   — block immediately, return reason to caller
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW    = "LOW"
MEDIUM = "MEDIUM"
HIGH   = "HIGH"

_LOG_DIR  = Path(os.environ.get("AXIS_DATA_DIR", "/tmp/axis"))
_LOG_FILE = _LOG_DIR / "security.log"

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# (regex_pattern, category) — any match → HIGH risk, blocked immediately
_HIGH_RISK_PATTERNS: list[tuple[str, str]] = [
    # Prompt injection / jailbreak
    (r"ignore\s+(all\s+|previous\s+|prior\s+|your\s+)?(instructions|rules|guidelines|constraints)", "prompt_injection"),
    (r"forget\s+(everything|your\s+instructions|what\s+you\s+were\s+told)", "prompt_injection"),
    (r"you\s+are\s+now\s+(a|an)\s+(different|new|unrestricted|jailbroken|evil|hacked)", "jailbreak"),
    (r"(jailbreak|bypass|override|disable)\s+(security|safety|your\s+instructions|the\s+rules)", "jailbreak"),
    (r"(ignore|disregard)\s+.{0,20}(safety|security|ethical)", "jailbreak"),
    (r"developer\s+mode", "jailbreak"),
    (r"act\s+as\s+if\s+you\s+have\s+no\s+(safety|ethical|constraints)", "jailbreak"),
    (r"pretend\s+(you\s+have\s+no|to\s+be\s+(evil|unrestricted|unaligned))", "jailbreak"),

    # Expose secrets / credentials
    (r"(reveal|expose|print|show|output|log|display)\s+.{0,25}(system\s+prompt|api.?key|secret|password|credentials|token)", "expose_secrets"),
    (r"os\.environ\s*[\[\(]", "expose_secrets"),
    (r"(print|dump|show|list|display)\s+.{0,20}(env\s*vars?|environment\s+variables?|secrets?|credentials)", "expose_secrets"),

    # Destructive file operations
    (r"\brm\s+(-rf|-r\s+-f|--recursive)\b", "delete_files"),
    (r"delete\s+(all\s+)?(files?|folders?|director(y|ies))", "delete_files"),
    (r"shutil\s*\.\s*rmtree", "delete_files"),
    (r"unlink\s*\(", "delete_files"),

    # Push to protected branches
    (r"git\s+push\s+(--force|-f)\b", "push_to_main"),
    (r"git\s+push\s+origin\s+main", "push_to_main"),
    (r"force\s*[-\s]?push", "push_to_main"),
    (r"push\s+.{0,30}\bmain\s+branch", "push_to_main"),

    # System file modification
    (r"(write|modify|edit|overwrite|delete)\s+(/etc/|/usr/|/bin/|/boot/|/sys/|/sbin/)", "modify_system_files"),
    (r"(system32|windows\\system|/etc/passwd|/etc/sudoers)", "modify_system_files"),
]

# (regex_pattern, category) — any match → MEDIUM risk, flagged but not blocked
_MEDIUM_RISK_PATTERNS: list[tuple[str, str]] = [
    (r"(make|send|call|issue)\s+(an?\s+)?(http|external|api|web)(\s+api)?\s+(request|call)", "external_api"),
    (r"(access|read|write\s+to)\s+(the\s+)?(file\s+system|disk|database|db)", "file_system"),
    (r"(run|execute|eval)\s+(code|script|command|bash|shell)", "code_execution"),
    (r"(deploy|release|publish|ship)\s+(to\s+)?(production|prod|staging|live)", "deploy"),
    (r"(drop|truncate|delete)\s+(table|database|collection|index)", "data_deletion"),
    (r"subprocess\s*\.\s*(run|call|Popen)", "code_execution"),
]

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _log(entry: dict) -> None:
    """Append one JSON line to the security log. Never raises."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with _LOG_FILE.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_prompt(text: str) -> dict:
    """
    Inspect user input for prompt injection or malicious instructions.

    Returns:
        {
            "risk":    "LOW" | "MEDIUM" | "HIGH",
            "reason":  str,
            "blocked": bool,   # True only when risk is HIGH
        }
    """
    lower = text.lower()

    for pattern, category in _HIGH_RISK_PATTERNS:
        if re.search(pattern, lower):
            reason = f"Blocked ({category}): matched pattern '{pattern}'"
            _log({
                "action":   "inspect_prompt",
                "decision": "BLOCKED",
                "reason":   reason,
                "preview":  text[:120],
            })
            return {"risk": HIGH, "reason": reason, "category": category, "blocked": True}

    for pattern, category in _MEDIUM_RISK_PATTERNS:
        if re.search(pattern, lower):
            reason = f"Flagged ({category}): matched pattern '{pattern}'"
            _log({
                "action":   "inspect_prompt",
                "decision": "FLAGGED",
                "reason":   reason,
                "preview":  text[:120],
            })
            return {"risk": MEDIUM, "reason": reason, "category": category, "blocked": False}

    _log({
        "action":   "inspect_prompt",
        "decision": "ALLOWED",
        "reason":   "No threats detected",
        "preview":  text[:120],
    })
    return {"risk": LOW, "reason": "No threats detected", "category": "none", "blocked": False}


def classify_risk(action: str) -> str:
    """
    Classify the risk level of a free-text action description.
    Returns LOW, MEDIUM, or HIGH.
    """
    lower = action.lower()
    for pattern, _ in _HIGH_RISK_PATTERNS:
        if re.search(pattern, lower):
            return HIGH
    for pattern, _ in _MEDIUM_RISK_PATTERNS:
        if re.search(pattern, lower):
            return MEDIUM
    return LOW


def inspect_action(action_type: str, details: str) -> dict:
    """
    Veto gate before an action executes.

    Args:
        action_type: tool name or action category (e.g. "save_task", "http_request")
        details:     human-readable description of what the action will do

    Returns:
        {"allowed": bool, "reason": str}
    """
    combined = f"{action_type} {details}".lower()

    # Hard veto rules: (trigger_patterns, category)
    veto_rules: list[tuple[list[str], str]] = [
        (["delete", "unlink", "rmdir", "rmtree", "rm -rf"],                 "delete_files"),
        (["git push origin main", "git push --force", "force push"],         "push_to_main"),
        (["api_key", "api key", "secret", "password", "credentials",
          "os.environ", "env vars", "system prompt"],                        "expose_secrets"),
        (["/etc/", "/usr/", "/bin/", "/boot/", "/sys/", "system32"],         "modify_system_files"),
    ]

    for triggers, category in veto_rules:
        for trigger in triggers:
            if trigger in combined:
                reason = f"Vetoed ({category}): trigger '{trigger}' found in action"
                _log({
                    "action":   f"inspect_action:{action_type}",
                    "decision": "VETOED",
                    "reason":   reason,
                    "details":  details[:120],
                })
                return {"allowed": False, "reason": reason}

    # External HTTP requests require explicit APPROVED: prefix
    if action_type == "http_request" and not details.startswith("APPROVED:"):
        reason = "External HTTP requests require explicit approval (prefix details with 'APPROVED:')"
        _log({
            "action":   "inspect_action:http_request",
            "decision": "VETOED",
            "reason":   reason,
            "details":  details[:120],
        })
        return {"allowed": False, "reason": reason}

    _log({
        "action":   f"inspect_action:{action_type}",
        "decision": "ALLOWED",
        "reason":   "Passed all veto checks",
        "details":  details[:120],
    })
    return {"allowed": True, "reason": "Passed all veto checks"}

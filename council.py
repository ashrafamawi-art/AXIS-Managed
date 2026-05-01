"""
AXIS Council — multi-perspective task reasoning.

Runs a task through 5 specialist lenses before AXIS decides,
then synthesizes into a structured recommendation.
"""

import json
import anthropic

PERSPECTIVES = [
    ("Planner",   "Break this into clear, ordered steps."),
    ("Analyst",   "Identify risks, unknowns, and missing data."),
    ("Critic",    "What could go wrong? What assumptions are being made?"),
    ("Optimizer", "How can this be done faster, cheaper, or more effectively?"),
    ("Executor",  "What concrete actions can be taken right now?"),
]

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "perspectives": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "role":    {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["role", "insight"],
            },
        },
        "synthesis":          {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": ["perspectives", "synthesis", "recommended_action"],
}

_SYSTEM = """\
You are the AXIS Council — a team of specialist advisors analyzing a task.
For each perspective provided, give a sharp, actionable insight in 1-2 sentences.
Then synthesize into one clear recommended action.
Output valid JSON only, no other text.
"""


def run(task: str, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> dict:
    """Run multi-perspective reasoning. Returns structured council dict."""
    lenses = "\n".join(f"- {role}: {desc}" for role, desc in PERSPECTIVES)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Task: {task}\n\nAnalyze from these perspectives:\n{lenses}",
        }],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    return json.loads(resp.content[0].text)


def format_for_axis(task: str, result: dict) -> str:
    """Build an enriched prompt for the AXIS session from the council output."""
    lines = [task, "", "[Council Pre-Analysis]"]
    for p in result.get("perspectives", []):
        lines.append(f"• {p['role']}: {p['insight']}")
    lines.append(f"\nSynthesis: {result.get('synthesis', '')}")
    lines.append(f"Recommended action: {result.get('recommended_action', '')}")
    return "\n".join(lines)

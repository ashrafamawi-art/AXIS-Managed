"""
AXIS Brain Tests — validates intent classification and routing decisions.

Run with: pytest brain_tests.py -v
"""

import pytest
from brain import AXISBrain

brain = AXISBrain()


# ── 1. Question → should NOT create task ─────────────────────────────────────

def test_question_intent():
    result = brain.classify("ما هو الفرق بين REST وGraphQL؟")
    assert result["intent"] == "question", f"Expected 'question', got '{result['intent']}'"
    assert result["task_complexity"] == "SIMPLE_EXECUTION"
    agents = result.get("agents", [])
    assert len(agents) == 0 or agents[0]["agent"] == "general"


# ── 2. Reminder → should create task ─────────────────────────────────────────

def test_reminder_creates_task():
    result = brain.classify("ذكرني بمراجعة عرض العميل الساعة 3 عصراً")
    assert result["intent"] == "task_create", f"Expected 'task_create', got '{result['intent']}'"
    assert result["task_complexity"] == "SIMPLE_EXECUTION"


# ── 3. Calendar request → should create calendar event ───────────────────────

def test_calendar_intent():
    result = brain.classify("احجز اجتماع مع أحمد غداً الساعة 10 صباحاً")
    assert result["intent"] == "calendar_create", f"Expected 'calendar_create', got '{result['intent']}'"
    agents = result.get("agents", [])
    assert len(agents) > 0, "Expected at least one agent"
    assert agents[0]["agent"] == "calendar"


# ── 4. Idea sharing → should NOT be saved blindly as memory ──────────────────

def test_idea_not_saved_blindly():
    result = brain.classify("أفكر في توسيع الشركة لسوق السعودية")
    assert result["intent"] != "memory_save", \
        "Idea sharing should NOT be auto-saved as memory"
    assert result["task_complexity"] in ("ASSISTED_EXECUTION", "COMPLEX_PLANNING")


# ── 5. Delete action → requires confirmation + medium/high risk ───────────────

def test_delete_requires_confirmation():
    result = brain.classify("احذف كل المهام القديمة")
    assert result["requires_confirmation"] is True, \
        "Delete action must require confirmation"
    assert result["risk"] in ("medium", "high")


# ── 6. GitHub PR → no confirmation; merge → confirmation + high risk ──────────

def test_github_pr_no_confirmation():
    result = brain.classify("افتح PR لتحسين الذاكرة")
    assert result["requires_confirmation"] is False, \
        "Opening a PR should NOT require confirmation"


def test_github_merge_requires_confirmation():
    result = brain.classify("merge الـ PR وdeploy")
    assert result["requires_confirmation"] is True, \
        "Merge + deploy must require confirmation"
    assert result["risk"] == "high"


# ── 8. Simple reminder uses fast path (classify, not plan) ───────────────────

def test_simple_reminder_fast_path():
    result = brain.classify("ذكرني بالاجتماع الساعة 3")
    assert result["task_complexity"] == "SIMPLE_EXECUTION", \
        "Simple reminder must use SIMPLE_EXECUTION fast path"
    assert result["intent"] == "task_create"


# ── 9. Complex research creates multi-step plan ───────────────────────────────

def test_complex_research_multi_step_plan():
    result = brain.plan("ابحث عن أفضل استراتيجيات دخول السوق السعودي لشركة تقنية وأعطني توصيات مفصّلة")
    steps = result["execution_steps"]
    assert len(steps) >= 2, "Complex research must produce at least 2 execution steps"
    agents_used = [s["agent"] for s in steps]
    assert "research" in agents_used, "Must include research agent"
    assert result["task_complexity"] == "COMPLEX_PLANNING"
    assert result["final_response_strategy"] == "summarize_results"


# ── 10. Self-improvement routes to development + github ──────────────────────

def test_self_improvement_routes_correctly():
    result = brain.plan("حسّن نفسك: أضف retry logic لـ executor.py")
    assert result["task_complexity"] == "SELF_IMPROVEMENT"
    assert result["requires_confirmation"] is True
    assert result["needs_user_confirmation"] is True
    agents_used = [s["agent"] for s in result["execution_steps"]]
    assert any(a in agents_used for a in ("development", "github")), \
        "SELF_IMPROVEMENT must involve development or github agent"
    assert result["final_response_strategy"] == "open_pr_summary"
    assert "PR" in result["safety_notes"] or "merge" in result["safety_notes"].lower(), \
        "safety_notes must mention PR-only constraint"


# ── 11. Delete action requires confirmation (plan path) ──────────────────────

def test_delete_requires_confirmation_plan():
    result = brain.plan("احذف كل السجلات القديمة من قاعدة البيانات")
    assert result["requires_confirmation"] is True, \
        "Delete via plan must require confirmation"
    assert result["risk"] in ("medium", "high")


# ── 12. Merge/deploy requires confirmation (plan path) ───────────────────────

def test_merge_deploy_requires_confirmation_plan():
    result = brain.plan("merge الـ PR وdeploy على production")
    assert result["requires_confirmation"] is True
    assert result["needs_user_confirmation"] is True
    assert result["risk"] == "high"


# ── 13. Normal question does not trigger unnecessary agents ──────────────────

def test_question_no_unnecessary_agents():
    result = brain.classify("ما هو الفرق بين Docker وKubernetes؟")
    assert result["intent"] == "question"
    agents = result.get("agents", [])
    # A question should use at most one general agent — not research, calendar, task, etc.
    non_general = [a for a in agents if a["agent"] not in ("general",)]
    assert len(non_general) == 0, \
        f"Question should not trigger specialized agents, got: {non_general}"

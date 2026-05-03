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

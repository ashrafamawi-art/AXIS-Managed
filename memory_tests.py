"""
AXIS Memory Integration Tests — validates smart save/retrieve and Brain context injection.

Run with: pytest memory_tests.py -v
"""

import pytest
from brain import AXISBrain
from memory_supabase import should_save, retrieve_relevant, save_with_category

brain = AXISBrain()


# ── 1. Brain classify() accepts memory context without error ─────────────────

def test_brain_classify_accepts_memory_context():
    context = "[Relevant Memory]\n• [2026-04-01] 'اجتماع مع أحمد'\n  → تم تحديد موعد الاجتماع"
    result = brain.classify("احجز اجتماع ثاني مع أحمد", context=context)
    assert "intent" in result
    assert "task_complexity" in result
    assert result["intent"] in (
        "calendar_create", "task_create", "general"
    ), f"Unexpected intent: {result['intent']}"


# ── 2. Random chat / question is NOT saved ────────────────────────────────────

def test_random_chat_not_saved():
    brain_output = {
        "intent": "question",
        "task_complexity": "SIMPLE_EXECUTION",
    }
    save, category = should_save(brain_output, "ما هو الفرق بين REST وGraphQL؟")
    assert save is False, "A simple question should NOT be saved to memory"


def test_general_chat_not_saved():
    brain_output = {
        "intent": "general",
        "task_complexity": "SIMPLE_EXECUTION",
    }
    save, category = should_save(brain_output, "شو رايك؟")
    assert save is False, "General chat should NOT be saved to memory"


def test_daily_briefing_not_saved():
    brain_output = {
        "intent": "daily_briefing",
        "task_complexity": "SIMPLE_EXECUTION",
    }
    save, category = should_save(brain_output, "شو عندي اليوم؟")
    assert save is False, "Daily briefing should NOT be saved to memory"


# ── 3. Durable preference IS saved ───────────────────────────────────────────

def test_explicit_memory_save_is_saved():
    brain_output = {
        "intent": "memory_save",
        "task_complexity": "SIMPLE_EXECUTION",
    }
    save, category = should_save(brain_output, "احفظ أن أحمد يفضل الاجتماعات الصباحية")
    assert save is True, "Explicit memory_save intent must be persisted"
    assert category == "preferences"


def test_durable_signal_saves_as_decision():
    brain_output = {
        "intent": "general",
        "task_complexity": "ASSISTED_EXECUTION",
    }
    save, category = should_save(brain_output, "اتفقنا مع العميل على إطلاق المشروع في يونيو")
    assert save is True, "Agreed decision should be saved"
    assert category in ("decisions", "contacts", "projects")


# ── 4. Project fact IS saved ──────────────────────────────────────────────────

def test_project_fact_is_saved():
    brain_output = {
        "intent": "general",
        "task_complexity": "COMPLEX_PLANNING",
    }
    save, category = should_save(brain_output, "حلل فرص مشروع التوسع في السعودية")
    assert save is True, "COMPLEX_PLANNING outcome should be saved"
    assert category == "decisions"


def test_task_create_saves_to_tasks_history():
    brain_output = {
        "intent": "task_create",
        "task_complexity": "SIMPLE_EXECUTION",
    }
    save, category = should_save(brain_output, "ذكرني بمراجعة العرض غداً")
    assert save is True
    assert category == "tasks_history"


# ── 5. retrieve_relevant() returns string without error ──────────────────────

def test_retrieve_relevant_returns_string():
    result = retrieve_relevant("اجتماع مع أحمد غداً")
    assert isinstance(result, str), "retrieve_relevant must return a string"


def test_retrieve_relevant_empty_query():
    result = retrieve_relevant("")
    assert isinstance(result, str)


# ── 6. Memory context improves planning (classify with vs without context) ────

def test_memory_context_does_not_break_classify():
    context = (
        "[Relevant Memory]\n"
        "• [2026-04-15] 'مشروع التوسع السعودي'\n"
        "  → قررنا التركيز على قطاع التقنية المالية\n"
        "• [2026-04-10] 'اتصال مع خالد'\n"
        "  → خالد هو المسؤول عن الشراكات في الرياض"
    )
    result = brain.classify(
        "ابحث عن شركاء محتملين في السوق السعودي",
        context=context,
    )
    assert result["intent"] in ("research", "general", "question")
    assert result["task_complexity"] in (
        "SIMPLE_EXECUTION", "ASSISTED_EXECUTION", "COMPLEX_PLANNING"
    )

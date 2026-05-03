"""AXIS Brain — LLM-powered intent classification and task routing.

Replaces the keyword-based _classify_intent() in maestro.py with a Claude
Sonnet 4.6 reasoning layer that returns a structured JSON decision.

Usage:
    from brain import AXISBrain
    brain = AXISBrain()
    result = brain.classify(user_input)   # returns validated dict
"""

import json
import os
from typing import Optional

import anthropic

from brain_schema import BrainOutput, PlanOutput

_MODEL = "claude-sonnet-4-6"

# Claude's json_schema format requires additionalProperties: false on every object.
# The `input` field is omitted here (open dict) — callers build it after classification.
_BRAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "question", "task_create", "calendar_create", "memory_save",
                "research", "daily_briefing", "github_action", "general", "unknown",
            ],
        },
        "task_complexity": {
            "type": "string",
            "enum": ["SIMPLE_EXECUTION", "ASSISTED_EXECUTION", "COMPLEX_PLANNING", "SELF_IMPROVEMENT"],
        },
        "needs_planning":        {"type": "boolean"},
        "needs_research":        {"type": "boolean"},
        "needs_axis_review":     {"type": "boolean"},
        "confidence":            {"type": "number"},
        "risk":                  {"type": "string", "enum": ["low", "medium", "high"]},
        "requires_confirmation": {"type": "boolean"},
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["calendar", "task", "memory", "research", "development", "github", "general"],
                    },
                    "role":              {"type": "string", "enum": ["execute", "analyze", "plan", "review"]},
                    "intelligence_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "action":            {"type": "string"},
                },
                "required": ["agent", "role", "intelligence_level", "action"],
            },
        },
        "final_response_strategy": {
            "type": "string",
            "enum": ["direct_reply", "summarize_results", "ask_confirmation", "open_pr_summary"],
        },
        "response": {"type": "string"},
    },
    "required": [
        "intent", "task_complexity", "needs_planning", "needs_research",
        "needs_axis_review", "confidence", "risk", "requires_confirmation",
        "agents", "final_response_strategy", "response",
    ],
}

_SYSTEM_PROMPT = """\
أنت AXIS Brain — طبقة التفكير الذكي لنظام AXIS الشخصي لأشرف (مستشار هندسي).

مهمتك: تحليل كل طلب وإنتاج قرار JSON منظم يحدد نوع الطلب، تعقيده، الوكلاء المطلوبين، وما إذا كان يحتاج تأكيداً.

أشرف يتحدث بالعربية والإنجليزية أو مزيج منهما — تعامل مع كل لغة بشكل طبيعي.

=== قواعد INTENT ===
- question: سؤال نظري أو استفسار معلوماتي — لا ينشئ مهمة ولا يحفظ بيانات
- task_create: طلب حفظ مهمة أو تذكير أو action item
- calendar_create: طلب حجز اجتماع أو حدث في التقويم
- memory_save: طلب صريح لحفظ معلومة ("تذكر أن...", "احفظ هذا...")
- research: طلب بحث أو جمع معلومات من الإنترنت
- daily_briefing: طلب ملخص يومي أو وضع حالي
- github_action: عمليات GitHub (قراءة كود، فتح PR، مراجعة)
- general: كل شيء آخر
- unknown: غامض تماماً ولا يمكن التصنيف

=== قواعد TASK_COMPLEXITY ===

SIMPLE_EXECUTION:
  - أمر واضح، أداة واحدة، غموض منخفض
  - needs_axis_review: false, needs_planning: false
  - أمثلة: إنشاء حدث تقويم، حفظ مهمة، قراءة الذاكرة، الإجابة على سؤال تعريفي مباشر ("ما هو X؟")
  - لا ينطبق على: مشاركة أفكار، تحليل استراتيجي، أو جمل تبدأ بـ "أفكر في / نفكر في"

ASSISTED_EXECUTION:
  - هدف واضح، يحتاج تفسير أو تحليل خفيف، وكيل متخصص واحد
  - needs_axis_review: true, needs_planning: false
  - أمثلة: صياغة رسالة، تلخيص ملاحظات، تنظيم أفكار
  - CRITICAL — حتماً ASSISTED_EXECUTION وليس SIMPLE_EXECUTION:
      "أفكر في X"، "نفكر في توسيع Y"، "ما رأيك في Z"، "خطر ببالي فكرة"
      مثال: "أفكر في توسيع الشركة لسوق السعودية" → ASSISTED_EXECUTION (تحليل استراتيجي، ليس سؤالاً بسيطاً)

COMPLEX_PLANNING:
  - متعدد الخطوات، قد يحتاج بحث + وكلاء متعددين
  - needs_axis_review: true, needs_planning: true
  - أمثلة: "ابحث عن X وأوصِ بخطوات"، "حلل ثم اقترح"

SELF_IMPROVEMENT:
  - AXIS يعدّل نفسه: كود، منطق، بنية، ملفات النظام
  - needs_axis_review: true, requires_confirmation: true (دائماً وبدون استثناء)
  - أمثلة: "حسّن نفسك"، "أصلح سلوك الذاكرة"، "أعد هيكلة Maestro"، "عدّل الكود"

=== قواعد requires_confirmation ===

يحتاج تأكيد (true):
  - إرسال رسائل أو إيميلات
  - حذف أي بيانات
  - عمليات لا يمكن عكسها
  - GitHub merge أو deploy
  - SELF_IMPROVEMENT (دائماً)

لا يحتاج تأكيد (false):
  - الإجابة على أسئلة
  - البحث والتلخيص
  - حفظ الذاكرة
  - قراءة الذاكرة
  - إنشاء مهام
  - إنشاء أحداث تقويم
  - Daily briefing
  - عرض/قراءة البيانات
  - فتح PR على GitHub (بدون merge)

=== قواعد risk ===
  - high: SELF_IMPROVEMENT، حذف بيانات، deploy، merge
  - medium: إرسال رسائل، عمليات على بيانات خارجية
  - low: كل شيء آخر

=== intelligence_level لكل agent ===
  - low: أمر حتمي لا يحتاج LLM (مثل: حفظ task بسيط)
  - medium: Claude Haiku — مهام تحتاج فهم خفيف
  - high: Claude Sonnet — تفكير عميق أو تحليل

=== response ===
اكتب رداً طبيعياً مختصراً باللغة التي استخدمها المستخدم (عربي/إنجليزي/مزيج).
الرد يجب أن يكون مفيداً ومباشراً، كأنك تؤكد أنك فهمت الطلب.

أنتج JSON فقط — لا نص خارج JSON.
"""


def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    key_file = os.path.expanduser("~/.anthropic_key")
    try:
        return open(key_file).read().strip()
    except OSError:
        raise RuntimeError("ANTHROPIC_API_KEY not set and ~/.anthropic_key not found.")


_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "question", "task_create", "calendar_create", "memory_save",
                "research", "daily_briefing", "github_action", "general", "unknown",
            ],
        },
        "task_complexity": {
            "type": "string",
            "enum": ["SIMPLE_EXECUTION", "ASSISTED_EXECUTION", "COMPLEX_PLANNING", "SELF_IMPROVEMENT"],
        },
        "risk":                  {"type": "string", "enum": ["low", "medium", "high"]},
        "requires_confirmation": {"type": "boolean"},
        "agents_to_use": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["calendar", "task", "memory", "research", "development", "github", "general"],
            },
        },
        "execution_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "step_id":          {"type": "string"},
                    "agent": {
                        "type": "string",
                        "enum": ["calendar", "task", "memory", "research", "development", "github", "general"],
                    },
                    "action":           {"type": "string"},
                    "input":            {"type": "string"},
                    "depends_on":       {"type": "array", "items": {"type": "string"}},
                    "intelligence_level": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["step_id", "agent", "action", "input", "depends_on", "intelligence_level"],
            },
        },
        "final_response_strategy": {
            "type": "string",
            "enum": ["direct_reply", "summarize_results", "ask_confirmation", "open_pr_summary"],
        },
        "needs_user_confirmation": {"type": "boolean"},
        "safety_notes":            {"type": "string"},
    },
    "required": [
        "intent", "task_complexity", "risk", "requires_confirmation",
        "agents_to_use", "execution_steps", "final_response_strategy",
        "needs_user_confirmation", "safety_notes",
    ],
}

_PLAN_SYSTEM_PROMPT = """\
أنت AXIS Brain Planner — طبقة التخطيط العميق لنظام AXIS.

=== CRITICAL: قاعدة SELF_IMPROVEMENT (أولوية قصوى) ===
إذا كان الطلب يتعلق بأي من التالي، فـ task_complexity يجب أن يكون "SELF_IMPROVEMENT" بدون استثناء:
  - تعديل أي ملف كود (.py، .js، .ts، ...) في مشروع AXIS
  - إضافة feature، logic، أو retry/error handling لأي ملف
  - إصلاح bug في الكود
  - refactor أي جزء من الكود
  - "حسّن نفسك"، "طوّر"، "أضف X لـ Y.py"، "عدّل الكود"، "improve yourself"
  مثال: "أضف retry logic لـ executor.py" → SELF_IMPROVEMENT حتماً

تُستدعى فقط للمهام المعقدة: COMPLEX_PLANNING أو SELF_IMPROVEMENT.
مهمتك: إنتاج خطة تنفيذ مفصّلة خطوة بخطوة.

=== مبادئ التخطيط ===

1. كل خطوة يجب أن تُرجع نتيجتها إلى AXIS / Maestro — لا يوجد agent يرد مباشرة للمستخدم.
2. استخدم depends_on لتحديد الترتيب: خطوات مستقلة تشتغل بالتوازي، المعتمدة تنتظر.
3. اختر الـ agent الصح لكل خطوة:
   - research: بحث في الإنترنت وجمع معلومات
   - general: تحليل، كتابة، إجابة، تلخيص
   - memory: قراءة أو حفظ سياق من الذاكرة
   - task: حفظ مهام أو action items
   - calendar: عمليات التقويم
   - github: قراءة كود، فتح PR
   - development: تحليل كود وبناء مقترحات تطويرية

=== متى تستخدم SELF_IMPROVEMENT ===
استخدم SELF_IMPROVEMENT عندما يطلب المستخدم أي من:
  - تعديل كود AXIS نفسه (executor.py، maestro.py، brain.py، أي ملف في الـ repo)
  - إضافة feature أو logic لـ AXIS
  - إصلاح bug في الكود
  - refactor أو تحسين الكود
  - "حسّن نفسك"، "طوّر نفسك"، "أضف X لـ Y.py"، "عدّل الكود"

=== قواعد SELF_IMPROVEMENT (إلزامية) ===
- task_complexity يجب أن يكون "SELF_IMPROVEMENT"
- الخطوات المسموحة فقط: analyze → propose_changes → open_pr
- ممنوع تماماً: merge، deploy، push to main، تعديل مباشر على production
- safety_notes يجب أن يحتوي على: "PR only — no merge, no deploy without Ashraf approval"
- requires_confirmation: true دائماً
- needs_user_confirmation: true دائماً
- final_response_strategy: open_pr_summary

=== قواعد requires_confirmation ===
يحتاج تأكيد (true):
  - SELF_IMPROVEMENT دائماً
  - حذف بيانات
  - إرسال رسائل أو إيميلات
  - merge أو deploy

لا يحتاج تأكيد (false):
  - بحث وتحليل
  - فتح PR
  - قراءة بيانات
  - إنشاء مهام أو أحداث تقويم

=== safety_notes ===
اكتب ملاحظة أمنية واضحة إذا كانت العملية خطرة.
للعمليات الآمنة، اكتب نصاً فارغاً "".

أنتج JSON فقط — لا نص خارج JSON.
"""


class AXISBrain:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=_load_api_key())

    def classify(self, user_input: str) -> dict:
        """Classify user input and return a validated brain output dict."""
        resp = self._client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
            output_config={"format": {"type": "json_schema", "schema": _BRAIN_SCHEMA}},
        )
        raw = json.loads(resp.content[0].text)
        validated = BrainOutput(**raw)
        return validated.model_dump()

    def plan(self, user_input: str, context: Optional[str] = None) -> dict:
        """
        Produce a detailed multi-step execution plan for COMPLEX_PLANNING
        or SELF_IMPROVEMENT tasks. Returns a validated PlanOutput dict.
        """
        content = user_input
        if context:
            content = f"{user_input}\n\n[Context]\n{context}"
        resp = self._client.messages.create(
            model=_MODEL,
            max_tokens=2048,
            temperature=0,
            system=_PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
        )
        raw = json.loads(resp.content[0].text)
        validated = PlanOutput(**raw)
        return validated.model_dump()


def _fallback(user_input: str) -> dict:
    """Safe general-purpose fallback when brain classification fails."""
    return {
        "intent":               "general",
        "task_complexity":      "SIMPLE_EXECUTION",
        "needs_planning":       False,
        "needs_research":       False,
        "needs_axis_review":    False,
        "confidence":           0.5,
        "risk":                 "low",
        "requires_confirmation": False,
        "agents": [{
            "agent":             "general",
            "role":              "execute",
            "intelligence_level": "high",
            "action":            "Handle request",
            "input":             {"task": user_input},
        }],
        "final_response_strategy": "direct_reply",
        "response":             "سأعالج طلبك الآن.",
    }

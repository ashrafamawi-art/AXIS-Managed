"""AXIS Brain — LLM-powered intent classification and task routing.

Replaces the keyword-based _classify_intent() in maestro.py with a Claude
Sonnet 4.6 reasoning layer that returns a structured JSON decision.

Usage:
    from brain import AXISBrain
    brain = AXISBrain()
    result = brain.classify(user_input)   # returns validated dict
"""

import json

import anthropic

from brain_schema import BrainOutput

_MODEL = "claude-sonnet-4-6"

# JSON Schema enforced by the API — mirrors BrainOutput exactly.
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
        "needs_planning":       {"type": "boolean"},
        "needs_research":       {"type": "boolean"},
        "needs_axis_review":    {"type": "boolean"},
        "confidence":           {"type": "number"},
        "risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
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
                    "input":             {"type": "object"},
                },
                "required": ["agent", "role", "intelligence_level", "action", "input"],
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
  - أمثلة: إنشاء حدث تقويم، حفظ مهمة، قراءة الذاكرة، الإجابة على سؤال مباشر

ASSISTED_EXECUTION:
  - هدف واضح، يحتاج تفسير خفيف، وكيل متخصص واحد
  - needs_axis_review: true, needs_planning: false
  - أمثلة: صياغة رسالة، تلخيص ملاحظات، تنظيم أفكار

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


class AXISBrain:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def classify(self, user_input: str) -> dict:
        """
        Classify user input and return a validated brain output dict.
        Falls back to a safe default on any error.
        """
        try:
            resp = self._client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_input}],
                output_config={"format": {"type": "json_schema", "schema": _BRAIN_SCHEMA}},
            )
            raw = json.loads(resp.content[0].text)
            validated = BrainOutput(**raw)
            return validated.model_dump()
        except Exception as exc:
            print(f"[brain] classify error: {exc} — using fallback")
            return _fallback(user_input)


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

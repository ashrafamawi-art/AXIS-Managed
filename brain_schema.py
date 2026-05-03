"""AXIS Brain — Pydantic schema for structured brain output."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    QUESTION        = "question"
    TASK_CREATE     = "task_create"
    CALENDAR_CREATE = "calendar_create"
    MEMORY_SAVE     = "memory_save"
    RESEARCH        = "research"
    DAILY_BRIEFING  = "daily_briefing"
    GITHUB_ACTION   = "github_action"
    GENERAL         = "general"
    UNKNOWN         = "unknown"


class TaskComplexity(str, Enum):
    SIMPLE_EXECUTION  = "SIMPLE_EXECUTION"
    ASSISTED_EXECUTION = "ASSISTED_EXECUTION"
    COMPLEX_PLANNING  = "COMPLEX_PLANNING"
    SELF_IMPROVEMENT  = "SELF_IMPROVEMENT"


class Risk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AgentRole(str, Enum):
    EXECUTE = "execute"
    ANALYZE = "analyze"
    PLAN    = "plan"
    REVIEW  = "review"


class IntelligenceLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class AgentName(str, Enum):
    CALENDAR    = "calendar"
    TASK        = "task"
    MEMORY      = "memory"
    RESEARCH    = "research"
    DEVELOPMENT = "development"
    GITHUB      = "github"
    GENERAL     = "general"


class FinalResponseStrategy(str, Enum):
    DIRECT_REPLY      = "direct_reply"
    SUMMARIZE_RESULTS = "summarize_results"
    ASK_CONFIRMATION  = "ask_confirmation"
    OPEN_PR_SUMMARY   = "open_pr_summary"


class AgentInstruction(BaseModel):
    agent:              AgentName
    role:               AgentRole
    intelligence_level: IntelligenceLevel
    action:             str
    input:              Optional[dict[str, Any]] = Field(default_factory=dict)


class BrainOutput(BaseModel):
    intent:               Intent
    task_complexity:      TaskComplexity
    needs_planning:       bool
    needs_research:       bool
    needs_axis_review:    bool
    confidence:           float = Field(ge=0.0, le=1.0)
    risk:                 Risk
    requires_confirmation: bool
    agents:               list[AgentInstruction]
    final_response_strategy: FinalResponseStrategy
    response:             str

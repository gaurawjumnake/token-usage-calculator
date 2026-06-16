from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class QuestionnaireSummary(BaseModel):
    app_type: str
    agentic_level: str
    scale: str
    priority: str
    budget_range: str


class WorkloadProfile(BaseModel):
    project_duration_months:   int
    active_users:              int
    requests_per_user_per_day: int
    avg_input_tokens:          int
    avg_output_tokens:         int
    avg_reasoning_tokens:      int
    avg_cached_tokens:         int
    min_context_window:        int
    recommended_context_window: int
    complexity:                Literal["low", "medium", "high"]
    latency_requirement:       Literal["low", "medium", "high"]
    batch_eligible:            bool
    cache_eligible:            bool


class SingleModelRecommendation(BaseModel):
    category: Literal["recommended", "budget", "premium"]
    model_id: str
    why: str
    tradeoffs: str


class ArchitectureRole(BaseModel):
    role: str  # planner | executor | validator | retriever | supervisor
    recommended_model_id: str
    budget_model_id: str
    premium_model_id: str
    reason: str


class Architecture(BaseModel):
    pattern: str
    hosting_strategy: str
    agent_framework_recommendation: str | None = None
    framework_constraints: list[str] = Field(default_factory=list)
    roles: list[ArchitectureRole] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimisationTip(BaseModel):
    impact: Literal["high", "medium", "low"]
    title: str
    detail: str


class ConfidenceBlock(BaseModel):
    score: Literal["high", "medium", "low"]
    reason: str                              # NOT "rationale"
    assumptions: list[str]


class RecommendationOutput(BaseModel):
    schema_version: str = "2.0"
    generated_at: str
    input_hash: str
    questionnaire_summary: QuestionnaireSummary
    workload_profile: WorkloadProfile        # NOT "workload_assumptions"
    single_model_recommendations: list[SingleModelRecommendation]
    architecture: Architecture
    optimisation_tips: list[OptimisationTip]
    confidence: ConfidenceBlock

class RecommendationOutput_API(BaseModel):
    schema_version: str = "2.0"
    generated_at: str
    input_hash: str
    questionnaire_summary: QuestionnaireSummary
    workload_profile: WorkloadProfile        # NOT "workload_assumptions"
    single_model_recommendations: list[SingleModelRecommendation]
    architecture: Architecture
    optimisation_tips: list[OptimisationTip]
    confidence: ConfidenceBlock
    pricing_information: list[dict[str, Any]] = Field(default_factory=list) #type: ignore

# ── Misc ─────────────────────────────────────────────────────────────────────

class Question(BaseModel):
    id: str
    question: str
    type: Literal["single_choice", "multi_choice", "descriptive", "text_area"]
    options: list[str] = Field(default_factory=list)


class Section(BaseModel):
    section: str
    questions: list[Question]


class QuestionnaireResponse(BaseModel):
    title: str
    description: str | None = None
    sections: list[Section]


class ModelsResponse(BaseModel):
    total: int
    models: list[dict[str, Any]]
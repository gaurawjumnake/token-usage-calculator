from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QuestionnaireSummary(BaseModel):
    app_type: str
    agentic_level: str
    scale: str
    priority: str
    budget_range: str


class WorkloadProfile(BaseModel):
    project_duration_months: int
    active_users: int
    requests_per_user_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    avg_reasoning_tokens: int
    avg_cached_tokens: int
    min_context_window: int
    recommended_context_window: int
    complexity: Literal["low", "medium", "high"]
    latency_requirement: Literal["low", "medium", "high"]
    batch_eligible: bool
    cache_eligible: bool


class SingleModelRecommendation(BaseModel):
    category: Literal["recommended", "budget", "premium"]
    model_id: str
    why: str
    tradeoffs: str


class ArchitectureRole(BaseModel):
    role: str
    recommended_model_id: str
    budget_model_id: str
    premium_model_id: str
    reason: str


class Architecture(BaseModel):
    pattern: str
    hosting_strategy: str
    agent_framework_recommendation: Optional[str] = None
    framework_constraints: list[str] = Field(default_factory=list)
    roles: list[ArchitectureRole] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimisationTip(BaseModel):
    impact: Literal["high", "medium", "low"]
    title: str
    detail: str


class ConfidenceBlock(BaseModel):
    score: Literal["high", "medium", "low"]
    reason: str
    assumptions: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE-WISE MODELS  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical SDLC stages shown in the sidebar UI
SDLC_STAGES = [
    "Requirements",
    "Architecture",
    "Development",
    "Code Review",
    "Testing",
    "Documentation",
    "Deployment",
    "Maintenance",
]


class StageModelTriple(BaseModel):
    """Three model tiers for a single SDLC stage."""
    recommended_model_id: str
    budget_model_id: str
    premium_model_id: str
    recommended_why: str
    budget_why: str
    premium_why: str
    key_capability: str        # primary capability driving the selection
    tradeoffs: str             # brief note on tradeoffs across the three


class StageRecommendation(BaseModel):
    """
    Model recommendations for one SDLC stage.

    stage_name must be one of:
      Requirements | Architecture | Development | Code Review |
      Testing | Documentation | Deployment | Maintenance
    """
    stage_name: Literal[
        "Requirements",
        "Architecture",
        "Development",
        "Code Review",
        "Testing",
        "Documentation",
        "Deployment",
        "Maintenance",
    ]
    models: StageModelTriple
    rationale: str    # 1–2 sentences on why this stage needs these models
    workload_profile: dict[str, int] = Field(default_factory=dict)
    estimated_cost_per_request: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY OUTPUT SCHEMA  (updated — stage_recommendations now required)
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendationOutput(BaseModel):
    """
    Complete recommendation output including per-stage model recommendations.
    stage_recommendations covers all 8 SDLC stages.
    """
    schema_version: str = "2.0"
    generated_at: str
    input_hash: str

    questionnaire_summary: QuestionnaireSummary
    workload_profile: WorkloadProfile
    single_model_recommendations: list[SingleModelRecommendation]
    architecture: Architecture
    optimisation_tips: list[OptimisationTip]
    confidence: ConfidenceBlock

    # Stage-wise recommendations (one entry per SDLC stage)
    stage_recommendations: list[StageRecommendation] = Field(
        default_factory=list,
        description="Model recommendations for each of the 8 SDLC stages"
    )


class RecommendationOutput_API(BaseModel):
    """Extended output with pricing information (for API consumers)."""
    schema_version: str = "2.0"
    generated_at: str
    input_hash: str
    questionnaire_summary: QuestionnaireSummary
    workload_profile: WorkloadProfile
    single_model_recommendations: list[SingleModelRecommendation]
    architecture: Architecture
    optimisation_tips: list[OptimisationTip]
    confidence: ConfidenceBlock
    stage_recommendations: list[StageRecommendation] = Field(default_factory=list)
    pricing_information: list[dict[str, Any]] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL AGENT OUTPUT MODELS  (for CrewAI orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzerOutput(BaseModel):
    application_classification: dict[str, Any]
    required_capabilities: dict[str, bool]
    complexity: dict[str, Any]
    budget_focus: str
    reasoning: str


class SynthesizerOutput(BaseModel):
    recommendation_summary: dict[str, Any]
    capability_assessment: dict[str, Any]
    context_window_analysis: dict[str, Any]
    stage_recommendations: list[dict[str, Any]]
    selection_reasoning: str


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════

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
    description: Optional[str] = None
    sections: list[Section]


class ModelsResponse(BaseModel):
    total: int
    models: list[dict[str, Any]]

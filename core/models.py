"""
Domain Models — Pydantic v2
============================
Strict schemas for all API boundaries. Every field has a description
to power the auto-generated OpenAPI docs at /docs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    DRAFT     = "draft"
    RUNNING   = "running"
    PAUSED    = "paused"
    CONCLUDED = "concluded"


class TrafficAllocationStrategy(str, Enum):
    EQUAL         = "equal"           # Classic 50/50 A/B
    THOMPSON      = "thompson"        # Adaptive Thompson Sampling
    EPSILON_GREEDY = "epsilon_greedy" # ε-greedy bandit


class RecommendationStatus(str, Enum):
    SHIP_TREATMENT  = "ship_treatment"
    SHIP_CONTROL    = "ship_control"
    CONTINUE        = "continue"
    STOP_NULL       = "stop_null"
    NEEDS_MORE_DATA = "needs_more_data"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class VariantConfig(BaseModel):
    variant_id: str = Field(..., description="Unique identifier for this variant, e.g. 'control'")
    name: str       = Field(..., description="Human-readable variant name")
    description: str = Field(default="", description="What change this variant represents")


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Experiment name, e.g. 'checkout_button_color_q2'")
    description: str = Field(default="", description="Hypothesis and goal")
    variants: List[VariantConfig] = Field(
        ..., min_length=2, description="At least control and one treatment"
    )
    strategy: TrafficAllocationStrategy = Field(
        default=TrafficAllocationStrategy.THOMPSON,
        description="Traffic allocation strategy"
    )
    target_sample_size: int = Field(
        default=1000, ge=50, description="Target n per variant for power analysis"
    )
    min_detectable_effect: float = Field(
        default=0.05, ge=0.001, le=1.0, description="Minimum effect size worth detecting"
    )
    alpha: float = Field(default=0.05, description="Significance level (Type I error rate)")
    total_peeks: int = Field(default=5, description="Planned number of interim analyses")

    @field_validator("variants")
    @classmethod
    def must_have_unique_ids(cls, v: List[VariantConfig]) -> List[VariantConfig]:
        ids = [x.variant_id for x in v]
        if len(ids) != len(set(ids)):
            raise ValueError("variant_id values must be unique")
        return v


class Experiment(ExperimentCreate):
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime
    started_at: Optional[datetime] = None
    concluded_at: Optional[datetime] = None
    bandit_state: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Serialized Thompson Sampler posterior (α, β per variant)"
    )
    peek_count: int = Field(default=0, description="Number of interim analyses performed")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class AssignmentRequest(BaseModel):
    experiment_id: str = Field(..., description="Which experiment to assign the user to")
    user_id: str = Field(..., description="Stable user/session identifier for deduplication")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context features for future contextual bandit extension"
    )


class AssignmentResponse(BaseModel):
    experiment_id: str
    user_id: str
    variant_id: str
    assignment_id: str
    strategy_used: TrafficAllocationStrategy
    assigned_at: datetime


class ConversionEvent(BaseModel):
    experiment_id: str = Field(..., description="Experiment identifier")
    user_id: str = Field(..., description="Must match the user_id used at assignment time")
    reward: float = Field(
        ..., ge=0.0, le=1.0,
        description="Reward signal. Binary: 0 or 1. Continuous: normalized [0,1]"
    )
    event_type: str = Field(default="conversion", description="e.g. 'click', 'purchase', 'signup'")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analytics / Results
# ---------------------------------------------------------------------------

class VariantSummary(BaseModel):
    variant_id: str
    name: str
    trials: int
    conversions: int
    conversion_rate: float
    conversion_rate_ci_lower: float
    conversion_rate_ci_upper: float
    posterior_alpha: float
    posterior_beta: float
    prob_best: float   # Thompson Sampling P(this arm is best)


class FrequentistSummary(BaseModel):
    p_value: float
    z_statistic: float
    absolute_lift: float
    relative_lift: float
    ci_lower: float
    ci_upper: float
    cohens_h: float
    is_significant: bool
    observed_power: float
    min_detectable_effect: float


class BayesianSummary(BaseModel):
    prob_treatment_better: float
    expected_loss_control: float
    expected_loss_treatment: float
    ci_control: List[float]
    ci_treatment: List[float]
    bayes_factor: float


class SequentialSummary(BaseModel):
    peek_number: int
    total_peeks_planned: int
    alpha_boundary: float
    current_p_value: float
    can_stop_early: bool
    rope_decision: str


class ExperimentResults(BaseModel):
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    variants: List[VariantSummary]
    frequentist: Optional[FrequentistSummary] = None
    bayesian: Optional[BayesianSummary] = None
    sequential: Optional[SequentialSummary] = None
    recommendation: str
    confidence_level: str
    total_assignments: int
    total_conversions: int
    computed_at: datetime


# ---------------------------------------------------------------------------
# Pagination / Generic
# ---------------------------------------------------------------------------

class PaginatedExperiments(BaseModel):
    items: List[Experiment]
    total: int
    page: int
    per_page: int


class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    timestamp: datetime

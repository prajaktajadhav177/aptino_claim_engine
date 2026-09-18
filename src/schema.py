"""
Pydantic models.

1) ClaimCase   -> what a reviewer submits (matches claim_case_schema.md)
2) DecisionResponse -> the exact contract asked for in the assignment
3) AgentState  -> the structured object that agents pass between each other
   (NOT free-form text - this is what makes it a "genuine" multi-agent system)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Patient(BaseModel):
    age: Optional[int] = None
    # extra / unknown fields are tolerated (see model_config below)
    model_config = {"extra": "allow"}


class Hospital(BaseModel):
    name: Optional[str] = None
    network_provider: Optional[bool] = None
    model_config = {"extra": "allow"}


class Treatment(BaseModel):
    type: Optional[str] = None                # inpatient / day_care / domiciliary
    admission_hours: Optional[float] = None
    diagnosis: Optional[str] = None
    procedure: Optional[str] = None
    pre_existing: Optional[bool] = None
    experimental: Optional[bool] = None
    hospital_room_unavailable: Optional[bool] = None
    patient_cannot_be_moved: Optional[bool] = None
    model_config = {"extra": "allow"}


class Expenses(BaseModel):
    room: float = 0
    doctor_fees: float = 0
    medicines_diagnostics: float = 0
    pre_hospitalization: float = 0
    post_hospitalization: float = 0
    ambulance: float = 0
    model_config = {"extra": "allow"}


class ClaimCase(BaseModel):
    case_id: str
    policy_id: str
    policy_start_date: str
    claim_date: str
    sum_insured_inr: float
    continuous_coverage_months: Optional[float] = 0
    prior_insurer_continuous_years: Optional[float] = 0
    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: Expenses
    documents: List[str] = Field(default_factory=list)
    task: str
    # optional extras that some public cases include
    evidence_context: Optional[Dict[str, Any]] = None
    expense_timing: Optional[Dict[str, Any]] = None
    prior_policy: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class Citation(BaseModel):
    claim: str
    source: str = "policy.pdf"
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None


class Validation(BaseModel):
    status: str = "PASS"           # PASS | FAIL
    unsupported_claims: List[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    agent: str
    action: str
    detail: str = ""
    elapsed_ms: float = 0.0


class DecisionResponse(BaseModel):
    case_id: str
    decision: str                  # ADMISSIBLE | ADMISSIBLE_WITH_LIMITS | NOT_ADMISSIBLE | NEEDS_REVIEW
    confidence: float
    key_findings: List[str] = Field(default_factory=list)
    applicable_limits: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)
    trace: List[TraceStep] = Field(default_factory=list)


class AgentState(BaseModel):
    """
    Shared, structured state object that is passed from agent to agent.
    Each agent reads what it needs and writes its own section - this is
    what keeps the pipeline a genuine multi-agent workflow instead of
    several independent prompts to the same model.
    """
    case: ClaimCase

    # filled in by RetrievalAgent
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    dimensions_checked: List[str] = Field(default_factory=list)

    # filled in by PolicyReasoningAgent
    draft_decision: Optional[str] = None
    findings: List[str] = Field(default_factory=list)
    limits: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.5

    # filled in by ValidationAgent
    validation: Validation = Field(default_factory=Validation)

    # filled in by DecisionAgent
    final_decision: Optional[str] = None

    trace: List[TraceStep] = Field(default_factory=list)

    # internal cache: the full rule-engine result, so PolicyReasoningAgent
    # doesn't have to re-run retrieval that RetrievalAgent already did.
    raw_result: Optional[Dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}

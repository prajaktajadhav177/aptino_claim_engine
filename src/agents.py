"""
Four specialized agents that share ONE AgentState object (src/schema.py).

Each agent has ONE job and reads/writes its own slice of the state:

  RetrievalAgent        -> runs hybrid retrieval + the rule engine's evidence
                            gathering, fills state.evidence / dimensions_checked
  PolicyReasoningAgent   -> turns the retrieved evidence into a draft decision:
                            findings, limits, missing_evidence, citations
  ValidationAgent        -> checks every material claim has a real citation
                            (catches "the LLM/rules made an unsupported claim")
  DecisionAgent          -> finalizes the decision (can downgrade to
                            NEEDS_REVIEW if validation failed) and returns
                            the response contract

This is what makes it a *genuine* multi-agent pipeline rather than one prompt
repeated four times: each agent only sees/produces structured data, and later
agents cannot see earlier agents' internal reasoning, only their structured
output.
"""

import time
from typing import List

from src.rule_engine import evaluate
from src.schema import AgentState, Citation, TraceStep, Validation


class RetrievalAgent:
    name = "RetrievalAgent"

    def run(self, state: AgentState, retriever) -> AgentState:
        t0 = time.time()
        result = evaluate(state.case, retriever)  # does dimension selection + all retrieval
        state.raw_result = result
        state.dimensions_checked = result["dimensions_checked"]
        state.evidence = result["evidence_pool"]
        elapsed = (time.time() - t0) * 1000
        state.trace.append(
            TraceStep(
                agent=self.name,
                action="hybrid_search",
                detail=(
                    f"dimensions={result['dimensions_checked']} -> "
                    f"{len(result['evidence_pool'])} evidence chunks retrieved"
                ),
                elapsed_ms=round(elapsed, 1),
            )
        )
        return state


class PolicyReasoningAgent:
    name = "PolicyReasoningAgent"

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        result = state.raw_result

        def to_citation(finding) -> Citation:
            ev = finding.evidence
            if ev is None:
                return None
            return Citation(
                claim=finding.text,
                source="policy.pdf",
                page=ev.chunk.page,
                section=ev.chunk.section,
                chunk_id=ev.chunk.chunk_id,
            )

        citations: List[Citation] = []
        for f in result["findings"] + result["limits"]:
            c = to_citation(f)
            if c is not None:
                citations.append(c)

        state.draft_decision = result["decision"]
        state.confidence = result["confidence"]
        state.findings = [f.text for f in result["findings"]]
        state.limits = [f.text for f in result["limits"]]
        state.missing_evidence = [f.text for f in result["missing_evidence"]]
        state.citations = citations

        elapsed = (time.time() - t0) * 1000
        state.trace.append(
            TraceStep(
                agent=self.name,
                action="reason_over_evidence",
                detail=(
                    f"draft_decision={state.draft_decision} findings={len(state.findings)} "
                    f"limits={len(state.limits)} missing_evidence={len(state.missing_evidence)}"
                ),
                elapsed_ms=round(elapsed, 1),
            )
        )
        return state


class ValidationAgent:
    """Makes sure every material (non-missing-evidence) statement carries a
    real citation back to a real chunk_id. This is the check that catches
    "the LLM attempts to make a claim that cannot be supported by the policy"
    (requirement 10)."""

    name = "ValidationAgent"

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        cited_claims = {c.claim for c in state.citations if c.chunk_id}
        unsupported = []
        for text in state.findings + state.limits:
            if text not in cited_claims:
                unsupported.append(text)

        status = "PASS" if not unsupported else "FAIL"
        state.validation = Validation(status=status, unsupported_claims=unsupported)

        elapsed = (time.time() - t0) * 1000
        state.trace.append(
            TraceStep(
                agent=self.name,
                action="validate_citations",
                detail=f"status={status} unsupported_claims={len(unsupported)}",
                elapsed_ms=round(elapsed, 1),
            )
        )
        return state


class DecisionAgent:
    """Finalizes the decision. If validation failed (a statement had no
    real citation), we do not let that statement's decision stand -- we
    downgrade to NEEDS_REVIEW rather than risk an unsupported conclusion."""

    name = "DecisionAgent"

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        if state.validation.status == "FAIL":
            state.final_decision = "NEEDS_REVIEW"
            state.confidence = min(state.confidence, 0.5)
        else:
            state.final_decision = state.draft_decision

        elapsed = (time.time() - t0) * 1000
        state.trace.append(
            TraceStep(
                agent=self.name,
                action="finalize_decision",
                detail=f"decision={state.final_decision} confidence={state.confidence}",
                elapsed_ms=round(elapsed, 1),
            )
        )
        return state

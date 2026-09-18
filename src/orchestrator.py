"""
Orchestrator: wires the pipeline together.

    ClaimCase -> RetrievalAgent -> PolicyReasoningAgent -> ValidationAgent
              -> DecisionAgent -> DecisionResponse

The policy PDF is ingested and indexed ONCE at startup (see load_retriever)
and reused for every claim, so answering a claim is fast.
"""

import time

from src.agents import DecisionAgent, PolicyReasoningAgent, RetrievalAgent, ValidationAgent
from src.ingestion import chunk_policy
from src.retrieval import HybridRetriever
from src.schema import AgentState, ClaimCase, DecisionResponse, TraceStep


def load_retriever(pdf_path: str) -> HybridRetriever:
    chunks = chunk_policy(pdf_path)
    return HybridRetriever(chunks)


class ClaimOrchestrator:
    def __init__(self, pdf_path: str):
        self.retriever = load_retriever(pdf_path)
        self.n_chunks = len(self.retriever.chunks)

        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = PolicyReasoningAgent()
        self.validation_agent = ValidationAgent()
        self.decision_agent = DecisionAgent()

    def analyze(self, case: ClaimCase) -> DecisionResponse:
        t0 = time.time()
        state = AgentState(case=case)

        state = self.retrieval_agent.run(state, self.retriever)
        state = self.reasoning_agent.run(state)
        state = self.validation_agent.run(state)
        state = self.decision_agent.run(state)

        elapsed = (time.time() - t0) * 1000
        state.trace.append(
            TraceStep(
                agent="Orchestrator",
                action="pipeline_complete",
                detail="",
                elapsed_ms=round(elapsed, 1),
            )
        )

        return DecisionResponse(
            case_id=case.case_id,
            decision=state.final_decision,
            confidence=state.confidence,
            key_findings=state.findings,
            applicable_limits=state.limits,
            missing_evidence=state.missing_evidence,
            citations=state.citations,
            validation=state.validation,
            trace=state.trace,
        )

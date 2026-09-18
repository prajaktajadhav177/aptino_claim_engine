"""
FastAPI backend.

Endpoints:
    GET  /health              -> {"status": "ok", "policy_indexed": true, "chunks": N}
    GET  /cases/public        -> list of the 12 supplied public cases
    GET  /cases/public/{id}   -> one public case by case_id
    GET  /cases/custom        -> list of the 5+ candidate-created cases
    POST /analyze-claim       -> run the full pipeline on a ClaimCase JSON body

Run with:
    uvicorn src.main:app --reload --port 8000
"""

import json
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from src.orchestrator import ClaimOrchestrator
from src.schema import ClaimCase, DecisionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claim_engine")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(BASE_DIR, "data", "policy", "policy.pdf")
PUBLIC_CASES_PATH = os.path.join(BASE_DIR, "data", "public_test_cases.json")
CUSTOM_CASES_PATH = os.path.join(BASE_DIR, "data", "custom_test_cases.json")

app = FastAPI(title="Aptino Claim Decision Engine", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None
_startup_error = None


@app.on_event("startup")
def startup():
    global _orchestrator, _startup_error
    try:
        logger.info(f"Indexing policy PDF at {POLICY_PATH} ...")
        _orchestrator = ClaimOrchestrator(POLICY_PATH)
        logger.info(f"Policy index ready: {_orchestrator.n_chunks} chunks.")
    except Exception as e:  # keep the app alive even if indexing fails
        _startup_error = str(e)
        logger.exception("Failed to index policy at startup")


def _load_json(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {
        "status": "ok" if _orchestrator else "degraded",
        "policy_indexed": _orchestrator is not None,
        "chunks": _orchestrator.n_chunks if _orchestrator else 0,
        "startup_error": _startup_error,
    }


@app.get("/cases/public")
def list_public_cases():
    return _load_json(PUBLIC_CASES_PATH)


@app.get("/cases/public/{case_id}")
def get_public_case(case_id: str):
    for c in _load_json(PUBLIC_CASES_PATH):
        if c["case_id"] == case_id:
            return c
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@app.get("/cases/custom")
def list_custom_cases():
    return _load_json(CUSTOM_CASES_PATH)


@app.post("/analyze-claim", response_model=DecisionResponse)
def analyze_claim(case: ClaimCase):
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail=f"Policy index is not ready. startup_error={_startup_error}",
        )
    try:
        return _orchestrator.analyze(case)
    except Exception as e:
        logger.exception("Error analyzing claim")
        raise HTTPException(status_code=500, detail=f"Internal error while analyzing claim: {e}")

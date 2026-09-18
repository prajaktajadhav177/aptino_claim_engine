# Aptino — Policy-Aware Claim Decision Engine

Reads the Universal Sompo CSC Individual Health Insurance policy PDF and decides
whether a claim is admissible, using retrieval + a multi-agent pipeline. No API
key needed — reasoning is a deterministic rule engine, not an LLM call.

**Live app:** https://prajaktajadhav177-aptino-claim-engine-frontendapp-by1iig.streamlit.app
**Live API:** https://aptino-claim-engine-hdk7.onrender.com (docs at `/docs`)
**Repo:** https://github.com/prajaktajadhav177/aptino_claim_engine

If the policy + case facts aren't enough to decide safely, it returns
`NEEDS_REVIEW` instead of guessing.

> Note: the backend is on Render's free tier and sleeps after 15 min idle —
> the first request after a while may take 30-60s to wake up.

## What's here
aptino_claim_engine/
├── data/
│ ├── policy/policy.pdf
│ ├── public_test_cases.json # supplied, not modified
│ └── custom_test_cases.json # cases I added
├── src/
│ ├── ingestion.py # PDF -> chunks by section, not fixed size
│ ├── retrieval.py # BM25 + TF-IDF/SVD + RRF fusion + rerank
│ ├── rule_engine.py # the actual policy logic (no LLM)
│ ├── schema.py # Pydantic models
│ ├── agents.py # Retrieval / Reasoning / Validation / Decision agents
│ ├── orchestrator.py # runs the 4 agents in order
│ └── main.py # FastAPI app
├── frontend/app.py # Streamlit UI
├── eval/evaluate.py # runs all cases, writes eval/results.json
└── requirements.txt


## Running locally

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                  # terminal 2
```

## Why a rule engine instead of an LLM

No API key required, and the numbers are auditable — I hardcoded the actual
policy figures (30-day waiting period, 48-month PED period, 25%/40%/1%/2%/75%
sub-limits, 20% domiciliary cap, 30/60-day pre/post windows) instead of
trusting a model to get them right every time. Retrieval is still real hybrid
search (BM25 + TF-IDF/SVD, fused with RRF); only the reasoning on top is
rule-based.

## Retrieval

PDF is chunked by the policy's own section headings (DEFINITIONS, WHAT WE
COVER, WHAT WE EXCLUDE, etc.), not fixed-size windows. DEFINITIONS is split
further, one chunk per defined term, since most decisions hinge on a single
definition. Every chunk keeps its section and page number for traceability.

Search combines BM25 (sparse) with TF-IDF/SVD (dense, no model download
needed), fused with Reciprocal Rank Fusion, then reranked on keyword overlap.

## Agents

Four agents share one state object (`schema.py` / `agents.py`):
- **RetrievalAgent** — figures out which policy dimensions apply (waiting
  period, domiciliary rules, exclusions, sub-limits, etc.) and retrieves
  evidence for each.
- **PolicyReasoningAgent** — applies each rule and attaches the evidence
  chunk it relied on as a citation.
- **ValidationAgent** — checks every finding has a real citation; fails if
  anything is unsupported.
- **DecisionAgent** — finalizes the decision, downgrading to `NEEDS_REVIEW`
  if validation failed.

## Decision format

```json
{
  "case_id": "PUB-008",
  "decision": "NOT_ADMISSIBLE",
  "confidence": 0.85,
  "key_findings": ["..."],
  "applicable_limits": ["..."],
  "missing_evidence": ["..."],
  "citations": [
    {"claim": "...", "source": "policy.pdf", "page": 8, "section": "WHAT WE EXCLUDE", "chunk_id": "chunk_..."}
  ],
  "validation": {"status": "PASS", "unsupported_claims": []},
  "trace": [{"agent": "RetrievalAgent", "action": "hybrid_search", "detail": "...", "elapsed_ms": 12.3}]
}
```

`decision` is one of: `ADMISSIBLE`, `ADMISSIBLE_WITH_LIMITS`,
`PARTIALLY_ADMISSIBLE`, `NOT_ADMISSIBLE`, `NEEDS_REVIEW`.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | status + whether the policy index is ready |
| GET | `/cases/public` | the 12 supplied cases |
| GET | `/cases/public/{case_id}` | one case |
| GET | `/cases/custom` | cases I added |
| POST | `/analyze` | send a claim JSON, get a decision back |

```bash
curl -X POST https://aptino-claim-engine-hdk7.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d @<(curl -s https://aptino-claim-engine-hdk7.onrender.com/cases/public/PUB-008)
```

Bad input returns a 422 with field-level errors (FastAPI/Pydantic, automatic).

## Evaluation

```bash
python -m eval.evaluate
```
Runs all public + custom cases, writes `eval/results.json`. See that file for
the full per-case breakdown (decision, confidence, validation status,
citation count).

## Failure cases found and fixed during development

1. **Permanent disease exclusions weren't checked at all.** Early version let
   Asthma/Diabetes/Hypertension claims through as admissible because only
   waiting periods were modeled, not the policy's separate "permanently
   excluded illness" list (item 20). Root cause: I'd modeled waiting periods
   thoroughly but missed that some exclusions aren't time-based at all. Fixed
   by adding `EXCLUDED_ILLNESS_KEYWORDS` as its own check, independent of
   waiting-period logic.
2. **Alternative treatment (Ayurveda/Homeopathy) wasn't excluded.** A
   Panchakarma claim came back admissible because nothing checked treatment
   *system*, only diagnosis/procedure against exclusion lists. Fixed by
   adding a dedicated alternative-treatment keyword check.
3. **Accident-related joint replacement was wrongly blocked.** The policy
   excludes "joint replacement unless due to accident" — my keyword match
   didn't parse the exception clause, so it always applied the 1-year
   waiting period even for accident cases. Fixed by checking for an
   accident/injury flag before applying that specific waiting period.
4. **Excluded-illness keyword matching can't distinguish "treatment of X"
   from "treatment of a complication of X"** (e.g. diabetic foot infection
   vs. diabetes itself) — still unresolved, surfaced as `NEEDS_REVIEW`
   rather than guessed. Would need an LLM read of the discharge summary to
   resolve properly.

## Known limitations

- No `length_of_stay_days` field in the input, so the room-rent sub-limit
  (1%/day) can never be fully pinned down — always shows as missing evidence.
- Dense retrieval is TF-IDF/SVD, not real embeddings — weaker on pure
  paraphrases with no shared vocabulary.
- `PARTIALLY_ADMISSIBLE` currently only fires for pre/post-hospitalization
  expenses claimed outside the policy's 30/60-day windows; other exclusion
  types remain binary (blocking or not).

## Deployment

Backend on Render (free tier): `uvicorn src.main:app --host 0.0.0.0 --port $PORT`.
Frontend on Streamlit Community Cloud, `frontend/app.py`, with `BACKEND_URL`
secret set to the Render URL above. No other secrets required.
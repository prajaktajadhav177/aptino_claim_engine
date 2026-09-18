# Aptino — Policy-Aware Claim Decision Engine

Reads the Universal Sompo CSC Individual Health Insurance policy PDF and
decides whether a claim is admissible, using retrieval + a small multi-agent
pipeline. No API key needed — the reasoning step is a rule engine, not an
LLM call.

If the policy + case facts aren't enough to decide safely, it returns
`NEEDS_REVIEW` instead of guessing.

## What's here
aptino_claim_engine/
├── data/
│ ├── policy/policy.pdf
│ ├── public_test_cases.json # supplied, not modified
│ └── custom_test_cases.json # cases I added
├── src/
│ ├── ingestion.py # PDF -> chunks (by section, not fixed size)
│ ├── retrieval.py # BM25 + TF-IDF/SVD + RRF fusion + rerank
│ ├── rule_engine.py # the actual policy logic
│ ├── schema.py # Pydantic models
│ ├── agents.py # Retrieval / Reasoning / Validation / Decision agents
│ ├── orchestrator.py # runs the 4 agents in order
│ └── main.py # FastAPI app
├── frontend/app.py # Streamlit UI
├── eval/evaluate.py # runs all cases, writes eval/results.json
└── requirements.txt


## Running it

```bash
pip install -r requirements.txt

uvicorn src.main:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                  # terminal 2
```

Pick a case in the sidebar, click Analyze.

Or skip the servers and just run it in Python:

```python
from src.orchestrator import ClaimOrchestrator
from src.schema import ClaimCase
import json

orch = ClaimOrchestrator("data/policy/policy.pdf")
case = ClaimCase(**json.load(open("data/public_test_cases.json"))[7])
print(orch.analyze(case).model_dump_json(indent=2))
```

## Why a rule engine instead of an LLM

Didn't want to require an API key, and honestly it's easier to trust — I
hardcoded the actual numbers from the policy (30-day waiting period,
48-month PED period, the 25%/40%/1%/2%/75% sub-limits, 20% domiciliary cap,
30/60-day pre/post windows) instead of hoping a model gets them right every
time. Retrieval is still real hybrid search (BM25 + TF-IDF/SVD, fused with
RRF) — it's just the reasoning on top that's rules, not a model call.

If I swap in an LLM later, it'd replace `PolicyReasoningAgent` only. The
sub-limit math should probably stay rule-based regardless.

## Retrieval

PDF gets chunked by the policy's own section headings (DEFINITIONS, WHAT WE
COVER, WHAT WE EXCLUDE, etc.) instead of cutting every N characters.
DEFINITIONS gets split further, one chunk per defined term, since most
decisions come down to a single definition like "what counts as a
Hospital." Every chunk keeps its section and page number.

Search is BM25 (sparse, good for exact terms like "30 days") plus TF-IDF
compressed with SVD (a cheap stand-in for dense embeddings — no model
download needed). Results get combined with Reciprocal Rank Fusion, then
reranked by keyword overlap.

## Agents

Four agents, one shared state object (see `schema.py` / `agents.py`) —
not a chat history, an actual typed object each agent reads/writes fields
on:

- `RetrievalAgent` — figures out which policy dimensions matter for this
  case (waiting period? domiciliary rules? exclusions?) and pulls evidence
  for each.
- `PolicyReasoningAgent` — applies the actual rule for each dimension and
  attaches the evidence chunk it used as a citation.
- `ValidationAgent` — checks every finding actually has a real citation.
  If something's uncited, it fails validation.
- `DecisionAgent` — finalizes the decision. If validation failed, it
  downgrades to `NEEDS_REVIEW` instead of letting an unsupported claim
  through.

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
  "trace": [
    {"agent": "RetrievalAgent", "action": "hybrid_search", "detail": "...", "elapsed_ms": 12.3}
  ]
}
```

`decision` is one of `ADMISSIBLE`, `ADMISSIBLE_WITH_LIMITS`,
`NOT_ADMISSIBLE`, `NEEDS_REVIEW`.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | status + whether the policy index is ready |
| GET | `/cases/public` | the 12 supplied cases |
| GET | `/cases/public/{case_id}` | one case |
| GET | `/cases/custom` | cases I added |
| POST | `/analyze-claim` | send a claim JSON, get a decision back |

```bash
curl -X POST http://localhost:8000/analyze-claim \
  -H "Content-Type: application/json" \
  -d @<(curl -s http://localhost:8000/cases/public/PUB-008)
```

Bad input just gets a 422 with the field errors — FastAPI/Pydantic handle
that for free.

## Evaluation

```bash
python -m eval.evaluate
```

Runs everything in `public_test_cases.json` + `custom_test_cases.json`,
writes `eval/results.json`. Last run:

<!-- PASTE YOUR ACTUAL eval/results.json TABLE HERE — don't reuse old numbers -->

## Known issues

- 30-day waiting period has no accident carve-out in this policy text, so
  an accident claim inside the first 30 days still comes back
  `NOT_ADMISSIBLE`. Some real insurers waive this for accidents, but it's
  not written in this policy, so I didn't invent it.
- Keyword matching on the excluded-illness list (item 20) can't tell
  "treatment of diabetes" from "treatment of a complication of diabetes."
  Comes back `NEEDS_REVIEW` rather than guessing.
- No `length_of_stay_days` field in the input, so the room-rent sub-limit
  (1%/day) can never be fully pinned down — always shows up as missing
  evidence.
- Dense retrieval is TF-IDF/SVD, not real embeddings. Weaker on pure
  paraphrases with no shared vocabulary.

## Deploying

Backend on Render (`uvicorn src.main:app --host 0.0.0.0 --port $PORT`),
frontend on Streamlit Community Cloud pointed at `frontend/app.py` with
`BACKEND_URL` set to the Render URL. No secrets needed either way.
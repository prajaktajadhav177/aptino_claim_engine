# Architecture & Design Note

## 1. Problem framing

A claim case is not answerable by one retrieval query — it usually depends
on several independent policy dimensions at once (is the facility a
"Hospital"? has the waiting period elapsed? does a sub-limit reduce the
payable amount? is the treatment on an exclusion list?). Getting any one
of those wrong changes the outcome, so the system investigates each
dimension separately and only combines them at the end.

## 2. Pipeline

```
ClaimCase (JSON)
      │
      ▼
┌─────────────────┐   picks which policy "dimensions" apply to this case
│ RetrievalAgent   │   (waiting period? domiciliary? cosmetic? ...), runs
│                  │   hybrid retrieval per dimension, tags every result
└────────┬─────────┘   with a chunk_id/section/page.
         │ state.evidence, state.dimensions_checked
         ▼
┌─────────────────┐   applies the concrete policy rule for each dimension
│PolicyReasoningAgt│   (a % cap, a day count, an exclusion keyword), and
│                  │   attaches the best-matching evidence chunk as the
└────────┬─────────┘   citation for that statement.
         │ state.findings / limits / missing_evidence / citations
         ▼
┌─────────────────┐   confirms every finding/limit text has a REAL
│ ValidationAgent  │   citation with a chunk_id. If any statement is
│                  │   uncited, marks validation as FAIL.
└────────┬─────────┘   state.validation
         ▼
┌─────────────────┐   if validation FAILED, downgrades to NEEDS_REVIEW
│ DecisionAgent    │   (never lets an unsupported statement stand);
│                  │   otherwise keeps the draft decision.
└────────┬─────────┘   state.final_decision
         ▼
   DecisionResponse (JSON) — the contract in section 5 of the assignment
```

All four agents read and write the **same `AgentState` Pydantic object**
(see `src/schema.py`), each touching only its own fields. This is the part
of the assignment that most directly tests "genuine multi-agent workflow":
the boundary between agents is a typed, inspectable object, not a shared
prompt string.

## 3. Retrieval design and trade-offs

- **Chunking by policy structure, not fixed size.** The PDF's own section
  headings are detected with a simple heuristic (short, upper-case-heavy
  line matching a known heading list). Inside `DEFINITIONS` we split further
  into one chunk per defined term, because a decision usually hinges on one
  definition ("what counts as a Hospital"), not the whole section.
- **Hybrid retrieval without a hosted embedding model.** BM25 (`rank_bm25`)
  gives sparse/lexical matching; TF-IDF + Truncated SVD gives a lightweight
  "dense" signal that tolerates some vocabulary mismatch. Both are fused
  with Reciprocal Rank Fusion (rank-based, so no score-scale tuning is
  needed) and then reranked by keyword overlap on the fused shortlist.
  **Trade-off:** this is weaker than a real sentence-embedding model on
  pure paraphrases, but it needs no model download, no GPU, and no
  network access — appropriate for a take-home assignment that should run
  anywhere in minutes.
- **Every citation is traceable.** `chunk_id`, `section`, and `page` travel
  with every retrieved chunk all the way to the final response.

## 4. Reasoning design and trade-offs

The reasoning step is a **deterministic rule engine**
(`src/rule_engine.py`), not an LLM call, by design:
- It hardcodes the *exact* numeric rules of this one policy (30-day and
  48-month waiting periods, 25%/40%/1%/2%/75% sub-limits, 20% domiciliary
  cap, 30/60-day pre/post windows, named exclusions), which keeps the
  system's numbers reproducible and auditable.
- It still *finds* the supporting evidence dynamically through the hybrid
  retriever rather than hardcoding page numbers, so citations stay accurate
  even if the chunking changes.
- **Trade-off:** a rule engine cannot handle a wording variant it wasn't
  written for. Anywhere the rules can't safely resolve an ambiguity (e.g.
  "is this a complication of an excluded illness, or the excluded illness
  itself?"), the engine deliberately emits a `NEEDS_REVIEW` finding plus a
  `missing_evidence` entry instead of guessing — this is the abstention
  behavior the assignment requires.

## 5. Validation as a safety net

`ValidationAgent` re-checks, independently of how a statement was produced,
that it is backed by a real `chunk_id`. This is what would catch a future
swap-in of an LLM-based reasoner going off-script and asserting something
the policy doesn't say — the statement simply won't have a citation, so
`ValidationAgent` flags it and `DecisionAgent` downgrades the case to
`NEEDS_REVIEW` rather than letting it through.

## 6. What I would do with more time

- Swap the TF-IDF/SVD dense retriever for a real sentence-embedding model
  (e.g. `sentence-transformers/all-MiniLM-L6-v2`) for better paraphrase
  recall.
- Add an optional LLM-backed reasoning path (behind the same
  `PolicyReasoningAgent` interface) for cases the rule engine can't resolve,
  with `ValidationAgent` unchanged as the safety net.
- Add a `length_of_stay_days` field to the claim schema so the room-rent
  sub-limit can be fully resolved instead of always appearing in
  `missing_evidence`.

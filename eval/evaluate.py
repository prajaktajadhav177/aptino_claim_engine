"""
Evaluation script.

Runs the claim engine against:
  - all 12 supplied public cases (data/public_test_cases.json)
  - all candidate-created custom cases (data/custom_test_cases.json)

and writes a combined results file to eval/results.json, plus prints a
one-line summary per case.

Run with:
    python -m eval.evaluate
(from the project root, so the "src" package is importable)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import ClaimOrchestrator
from src.schema import ClaimCase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(BASE_DIR, "data", "policy", "policy.pdf")
PUBLIC_CASES_PATH = os.path.join(BASE_DIR, "data", "public_test_cases.json")
CUSTOM_CASES_PATH = os.path.join(BASE_DIR, "data", "custom_test_cases.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def run_suite(orch, cases, suite_name):
    results = []
    print(f"\n=== {suite_name} ({len(cases)} cases) ===")
    for raw in cases:
        case = ClaimCase(**raw)
        response = orch.analyze(case)
        print(
            f"{case.case_id:10s} -> {response.decision:22s} "
            f"confidence={response.confidence:.2f} validation={response.validation.status} "
            f"citations={len(response.citations)} missing_evidence={len(response.missing_evidence)}"
        )
        results.append({"suite": suite_name, "case_id": case.case_id, "response": response.model_dump()})
    return results


def main():
    orch = ClaimOrchestrator(POLICY_PATH)
    print(f"Policy indexed: {orch.n_chunks} chunks from {POLICY_PATH}")

    public_cases = json.load(open(PUBLIC_CASES_PATH))
    custom_cases = json.load(open(CUSTOM_CASES_PATH))

    all_results = []
    all_results += run_suite(orch, public_cases, "public_cases")
    all_results += run_suite(orch, custom_cases, "custom_cases")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    decisions = [r["response"]["decision"] for r in all_results]
    print("\n=== Decision distribution ===")
    for d in sorted(set(decisions)):
        print(f"  {d}: {decisions.count(d)}")

    fails = [r for r in all_results if r["response"]["validation"]["status"] != "PASS"]
    print(f"\nValidation failures (unsupported claims caught): {len(fails)}")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

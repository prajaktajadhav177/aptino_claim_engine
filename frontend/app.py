"""
Streamlit frontend for the Aptino claim decision engine.

Run with (from the project root):
    streamlit run frontend/app.py

By default it talks to the FastAPI backend at http://localhost:8000.
Set BACKEND_URL as an environment variable to point elsewhere.
"""

import json
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")
st.title("🩺 Policy-Aware Claim Decision Engine")
st.caption(
    "Multi-agent, RAG-grounded claim admissibility assistant over the "
    "Universal Sompo CSC Individual Health Insurance policy."
)

DECISION_COLORS = {
    "ADMISSIBLE": "🟢",
    "ADMISSIBLE_WITH_LIMITS": "🟡",
    "NOT_ADMISSIBLE": "🔴",
    "NEEDS_REVIEW": "🟠",
}


@st.cache_data(ttl=30)
def fetch_cases(kind: str):
    try:
        resp = requests.get(f"{BACKEND_URL}/cases/{kind}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
        return []


def analyze(case_dict: dict):
    resp = requests.post(f"{BACKEND_URL}/analyze-claim", json=case_dict, timeout=60)
    return resp


# ---- Sidebar: health + case selection -------------------------------------
with st.sidebar:
    st.header("Backend")
    st.write(f"URL: `{BACKEND_URL}`")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if health.get("policy_indexed"):
            st.success(f"Policy indexed ✓ ({health.get('chunks')} chunks)")
        else:
            st.error(f"Not ready: {health.get('startup_error')}")
    except Exception as e:
        st.error(f"Backend unreachable: {e}")

    st.header("Pick a case")
    source = st.radio("Source", ["Public test cases", "Custom test cases", "Paste JSON"])

selected_case = None

if source == "Public test cases":
    cases = fetch_cases("public")
    if cases:
        ids = [c["case_id"] for c in cases]
        chosen = st.sidebar.selectbox("Case", ids)
        selected_case = next(c for c in cases if c["case_id"] == chosen)

elif source == "Custom test cases":
    cases = fetch_cases("custom")
    if cases:
        ids = [c["case_id"] for c in cases]
        chosen = st.sidebar.selectbox("Case", ids)
        selected_case = next(c for c in cases if c["case_id"] == chosen)

else:
    pasted = st.sidebar.text_area("Paste a claim case JSON", height=200)
    if pasted.strip():
        try:
            selected_case = json.loads(pasted)
        except json.JSONDecodeError as e:
            st.sidebar.error(f"Invalid JSON: {e}")

# ---- Main panel -------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Claim case")
    if selected_case:
        st.json(selected_case, expanded=False)
        run = st.button("🔍 Analyze claim", type="primary", use_container_width=True)
    else:
        st.info("Pick a case from the sidebar, or paste your own JSON.")
        run = False

with col2:
    st.subheader("Decision")
    if selected_case and run:
        with st.spinner("Running retrieval + multi-agent reasoning..."):
            try:
                resp = analyze(selected_case)
            except Exception as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp is not None:
            if resp.status_code != 200:
                st.error(f"Backend returned {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                decision = data["decision"]
                icon = DECISION_COLORS.get(decision, "⚪")
                st.markdown(f"### {icon} {decision}")
                st.metric("Confidence", f"{data['confidence']:.0%}")

                if decision == "NEEDS_REVIEW":
                    st.warning(
                        "The system is **abstaining** from a final admissible/not-admissible "
                        "call because the supplied evidence is insufficient for a safe conclusion."
                    )

                st.markdown("#### Key findings")
                for f in data["key_findings"]:
                    st.write(f"- {f}")

                st.markdown("#### Applicable limits / deductions")
                for l in data["applicable_limits"]:
                    st.write(f"- {l}")

                if data["missing_evidence"]:
                    st.markdown("#### ⚠️ Missing evidence")
                    for m in data["missing_evidence"]:
                        st.write(f"- {m}")

                st.markdown("#### Policy citations")
                for c in data["citations"]:
                    with st.expander(f"📄 p.{c['page']} — {c['section']}"):
                        st.write(c["claim"])
                        st.caption(f"chunk_id: `{c['chunk_id']}`  |  source: {c['source']}")

                st.markdown("#### Validation")
                v = data["validation"]
                if v["status"] == "PASS":
                    st.success("All material claims are backed by a policy citation.")
                else:
                    st.error("Some claims were NOT backed by a citation and were downgraded:")
                    for u in v["unsupported_claims"]:
                        st.write(f"- {u}")

                st.markdown("#### Execution trace (agents, actions, timings)")
                trace_rows = [
                    {
                        "agent": t["agent"],
                        "action": t["action"],
                        "detail": t["detail"],
                        "elapsed_ms": t["elapsed_ms"],
                    }
                    for t in data["trace"]
                ]
                st.dataframe(trace_rows, use_container_width=True)
    elif selected_case:
        st.info("Click **Analyze claim** to run the pipeline.")

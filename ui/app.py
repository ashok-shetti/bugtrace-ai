import os
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="BugTrace AI",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TOP_K = 3

st.markdown("""
<style>
.stApp {
    background: #0f172a;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
    margin-bottom: 0.4rem !important;
}

.product-title {
    color: #60a5fa;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 1.4rem;
}

.stTextArea textarea {
    background: #111827 !important;
    color: #e5e7eb !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    padding: 0.9rem !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

.stButton button {
    border-radius: 8px !important;
    height: 2.6rem;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 9px;
}

[data-testid="stExpander"] {
    border-radius: 9px;
    border: 1px solid #334155;
}

hr {
    margin: 1.2rem 0 !important;
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🛠️ BugTrace AI")

st.markdown("""
<div class="product-title">AI-Powered Error Diagnosis & Resolution</div>
<div class="subtitle">
Paste an error or stack trace to identify the root cause,
discover similar historical issues, and get a recommended fix.
</div>
""", unsafe_allow_html=True)

# INPUT
query = st.text_area(
    "Error Input",
    placeholder="Paste your stack trace, exception, error logs, or describe the bug...",
    height=190,
    label_visibility="collapsed",
)

st.caption("Supports stack traces, exceptions, error logs, and bug descriptions.")

# ACTIONS
col1, col2 = st.columns([2, 1])

with col1:
    diagnose = st.button(
        "✨ Analyze Error",
        type="primary",
        use_container_width=True,
    )

with col2:
    search = st.button(
        "Search Similar Issues",
        use_container_width=True,
    )

payload = {
    "query": query.strip(),
    "top_k": TOP_K,
    "labels": None,
}


def show_error(title, error):
    st.error(title)
    with st.expander("Technical Details"):
        st.code(str(error))


# DIAGNOSE
if diagnose:
    if not query.strip():
        st.warning("Please paste an error message or stack trace first.")
    else:
        try:
            with st.spinner("Analyzing your error..."):
                start = time.time()

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/diagnose",
                    json=payload,
                    timeout=45,
                )

                elapsed = round(time.time() - start, 2)

            if response.status_code != 200:
                show_error(
                    "Unable to analyze the error.",
                    f"Status: {response.status_code}\n\n{response.text}",
                )
            else:
                data = response.json()
                confidence = data.get("confidence_score", 0)

                st.divider()

                st.caption(
                    f"Completed in {elapsed}s · "
                    f"Confidence: {confidence * 100:.0f}%"
                )

                st.subheader("Summary")
                st.write(data.get("summary", "No summary available."))

                st.subheader("Root Cause")
                st.info(
                    data.get(
                        "root_cause_analysis",
                        "No root cause analysis available.",
                    )
                )

                st.subheader("Recommended Fix")
                fix = data.get("recommended_fix", "")

                if fix:
                    st.markdown(str(fix))
                else:
                    st.write("No specific fix was provided.")

                issues = data.get("referenced_issues", [])

                if issues:
                    st.subheader("Related Historical Issues")

                    for issue in issues:
                        number = issue.get("issue_number", "Unknown")
                        reason = issue.get("relevance_reason", "")

                        with st.expander(f"Issue #{number}"):
                            st.write(str(reason))

        except requests.exceptions.RequestException as error:
            show_error("Unable to reach the BugTrace backend.", error)

        except Exception as error:
            show_error("Error while displaying diagnosis results.", error)


# SEARCH
elif search:
    if not query.strip():
        st.warning("Please enter an error or search query first.")
    else:
        try:
            with st.spinner("Searching similar historical issues..."):
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/search",
                    json=payload,
                    timeout=15,
                )

            if response.status_code != 200:
                show_error(
                    "Unable to search historical issues.",
                    f"Status: {response.status_code}\n\n{response.text}",
                )
            else:
                data = response.json()

                # Support backend responses as either:
                # {"results": [...]} or [...]
                if isinstance(data, dict):
                    results = (
                        data.get("results")
                        or data.get("issues")
                        or data.get("data")
                        or []
                    )
                else:
                    results = data

                if not isinstance(results, list):
                    results = []

                st.divider()
                st.subheader(f"Similar Issues ({len(results)})")

                if not results:
                    st.info("No matching historical issues found.")

                for index, item in enumerate(results):
                    if not isinstance(item, dict):
                        continue

                    number = item.get("issue_number", "N/A")
                    title = item.get("title", "Untitled Issue")

                    with st.expander(
                        f"#{number} — {title}",
                        expanded=(index == 0),
                    ):
                        tags = item.get("labels", [])

                        if tags:
                            if isinstance(tags, list):
                                st.caption(
                                    f"Tags: {', '.join(map(str, tags))}"
                                )
                            else:
                                st.caption(f"Tags: {tags}")

                        st.write(
                            str(
                                item.get(
                                    "body",
                                    "No description available.",
                                )
                            )
                        )

        except requests.exceptions.RequestException as error:
            show_error("Unable to reach the BugTrace backend.", error)

        except Exception as error:
            show_error("Error while displaying search results.", error)
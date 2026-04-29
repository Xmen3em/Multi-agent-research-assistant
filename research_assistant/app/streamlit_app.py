from __future__ import annotations

import time

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("Document Processing")
    st.caption("Upload PDF Document")

    # Health indicator
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.success("Assistant: Online")
        else:
            st.error("Assistant: Offline")
    except Exception:
        st.warning("Assistant: Cannot connect to API (start with `uv run start_api`)")

    st.divider()

    uploaded_file = st.file_uploader(
        "Drag and drop file here\nLimit 200MB per file • PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner(f"Parsing and indexing {uploaded_file.name}..."):
                try:
                    r = httpx.post(
                        f"{API_BASE}/documents/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=120,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("status") == "error":
                            st.error(f"Indexing failed: {data['message']}")
                        else:
                            st.success(data["message"])
                            st.session_state["doc_processed"] = True
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as e:
                    st.error(f"Upload error: {e}")
    else:
        st.info("No document processed")

# --- Main Area ---
st.title("🔬 AI Research Assistant")
st.caption(
    "**Powered by** Tensorlake • Zep • Firecrawl • CrewAI • ChromaDB  \n"
    "Context Engineering Workflow with RAG, Web Search, Memory & Academic Research"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.info("Please process a document first using the sidebar, then ask a research question.")

# Handle new query
if query := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info(
            "Gathering context from RAG, Memory, Web, and ArXiv in parallel..."
        )

        try:
            # Submit query to API
            r = httpx.post(
                f"{API_BASE}/query",
                json={"query": query, "user_id": "default_user"},
                timeout=30,
            )
            r.raise_for_status()
            session_id = r.json()["session_id"]

            # Poll for result (max 15 minutes: 180 x 5s)
            final_response = None
            for _ in range(180):
                time.sleep(5)
                poll = httpx.get(f"{API_BASE}/query/{session_id}", timeout=10)
                data = poll.json()

                if data["status"] == "complete":
                    final_response = data["final_response"]
                    status_placeholder.empty()
                    st.markdown(final_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_response}
                    )

                    # Show context evaluation scores
                    if data.get("evaluation"):
                        with st.expander("Context Evaluation Details"):
                            eval_data = data["evaluation"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Relevant Sources")
                                for src in eval_data.get("relevant_sources", []):
                                    st.markdown(f"- {src}")
                            with col2:
                                st.subheader("Relevance Scores")
                                for src, score in eval_data.get(
                                    "relevance_scores", {}
                                ).items():
                                    st.progress(score, text=f"{src}: {score:.2f}")
                    break

                elif data["status"] == "error":
                    status_placeholder.error(
                        f"Error: {data.get('error', 'Unknown error occurred')}"
                    )
                    break
                else:
                    status_placeholder.info(f"Status: {data['status']}...")

            if final_response is None and data.get("status") not in ("complete", "error"):
                status_placeholder.error("Request timed out after 10 minutes.")

        except httpx.ConnectError:
            status_placeholder.error(
                "Cannot connect to the API server. "
                "Start it with: `uv run start_api`"
            )
        except Exception as e:
            status_placeholder.error(f"Unexpected error: {e}")

# Reset chat button
if st.session_state.messages:
    if st.button("Reset Chat", key="reset"):
        st.session_state.messages = []
        st.rerun()

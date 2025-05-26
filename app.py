"""
This is a Streamlit app that allows you to index and query documents using the RAGTool.

Usage:
streamlit run app.py
"""

import os
from typing import Any, Dict

import streamlit as st

import main

config_dict = main.load_config_from_yaml("config.yaml")
rag_config = main.RAGConfig(**config_dict)

# -----------------------------------------------------------------------------
# Page config & basic style tweaks
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chat • Ask your PDFs",
    page_icon="📝",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Dark/Light mode toggle and dynamic CSS
# -----------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True  # Default to dark mode

with st.sidebar:
    st.session_state["dark_mode"] = st.toggle(
        "🌙 Dark mode", value=st.session_state["dark_mode"]
    )

# Inject custom CSS from file
css_file = "custom_styles.css"
if os.path.exists(css_file):
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper to extract plain answer (hide metadata)
# -----------------------------------------------------------------------------


def extract_answer(raw_resp: Any) -> str:
    """Return the plain answer string from whatever the RAGTool gives back.

    *   If it is already a string, pass through.
    *   If it is a dict (e.g. {'result': ..., 'source_documents': ...}) take the
        'result' field and optionally append nicely-formatted citations.
    *   Else fallback to str(raw_resp).
    """
    if isinstance(raw_resp, str):
        return raw_resp

    if isinstance(raw_resp, Dict):
        answer = raw_resp.get("answer", "")
        srcs = raw_resp.get("sources", [])
        if srcs:
            answer += "\n\n**Sources:**\n"
            for i, doc in enumerate(srcs, 1):
                try:
                    page = doc.metadata.get("page", "?")
                    src = doc.metadata.get("source", "document")
                    answer += f"- ({i}) *{src}*, page {page}\n"
                except Exception:
                    answer += f"- ({i}) reference\n"
        return answer

    # Fallback – just string-ify
    return str(raw_resp)


# -----------------------------------------------------------------------------
# Initialisation
# -----------------------------------------------------------------------------
if "tool" not in st.session_state:
    st.session_state.tool = main.RAGTool(rag_config)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 Upload a document in the sidebar, click *Index*, and then ask me anything about it.",
        }
    ]

if "current_metadata" not in st.session_state:
    st.session_state.current_metadata = ""

# -----------------------------------------------------------------------------
# Sidebar – Document indexing controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Document Indexing")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
    )
    dir_path = st.text_input("…or enter a folder path")
    meta_name = st.text_input("Metadata name", value="default")

    if st.button("Index"):
        if uploaded_file is not None:
            tmp_path = os.path.join("/tmp", uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.tool.index(tmp_path, meta_name)
            st.success(
                f"Indexed **{uploaded_file.name}** ➜ metadata key **{meta_name}**"
            )
            st.session_state.current_metadata = meta_name
        elif dir_path:
            st.session_state.tool.index(dir_path, meta_name)
            st.success(f"Indexed folder **{dir_path}** ➜ metadata key **{meta_name}**")
            st.session_state.current_metadata = meta_name
        else:
            st.error("Please upload a document or specify a directory path.")

    st.markdown("---")
    if st.button("🗑️  Reset Chat"):
        st.session_state.messages = st.session_state.messages[:1]  # keep greeting
        st.experimental_rerun()

# -----------------------------------------------------------------------------
# Main area – Chat interface
# -----------------------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

user_prompt = st.chat_input("Ask a question about your documents…")

if user_prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Guard: ensure we have indexed docs
    if not st.session_state.current_metadata:
        assistant_reply = "Please index a document first! Use the sidebar to upload or select a folder."
    else:
        try:
            assistant_reply = st.session_state.tool.query(
                st.session_state.current_metadata,
                user_prompt,
            )["answer"]  # Get the answer from the result
        except Exception as e:
            assistant_reply = f"⚠️ An error occurred while querying: `{e}`"

    # Show assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply,
            "sources": st.session_state.tool.query(
                st.session_state.current_metadata, user_prompt
            )["sources"],
        }
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply, unsafe_allow_html=True)

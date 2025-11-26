import os
import tempfile

import streamlit as st

from backend.config import LLMConfig
from backend.services.rag_service import RAGService

# Page config
st.set_page_config(
    page_title="DocuSage AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load custom CSS
def load_css():
    with open("assets/custom_styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_service" not in st.session_state:
    config = LLMConfig()
    st.session_state.rag_service = RAGService(config)

if "current_collection" not in st.session_state:
    st.session_state.current_collection = "default_collection"

# Sidebar
with st.sidebar:
    st.title("📚 DocuSage AI")
    st.markdown("---")

    st.subheader("📁 Document Management")

    # Collection Name Input
    collection_name = st.text_input(
        "Collection Name",
        value=st.session_state.current_collection,
        help="Enter a unique name for your document collection",
    )

    if collection_name:
        st.session_state.current_collection = collection_name

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a while."):
                # Create a temporary directory to save uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Index the documents
                    try:
                        st.session_state.rag_service.index(
                            data_path=temp_dir,
                            collection_name=st.session_state.current_collection,
                        )
                        st.success(
                            f"Successfully processed {len(uploaded_files)} files!"
                        )
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        else:
            st.warning("Please upload at least one PDF file.")

    st.markdown("---")
    st.markdown("### 🛠️ Settings")
    st.info(f"Current Collection: **{st.session_state.current_collection}**")

# Main Chat Interface
st.header("💬 Chat with your Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f"**Source {i}:** {os.path.basename(source)} (Page {page})"
                    )
                    st.markdown(f"> {doc.page_content[:300]}...")
                    st.markdown("---")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_service.query(
                    collection_name=st.session_state.current_collection, question=prompt
                )

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)

                # Display sources in an expander
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(sources, 1):
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            st.markdown(
                                f"**Source {i}:** {os.path.basename(source)} (Page {page})"
                            )
                            st.markdown(f"> {doc.page_content[:300]}...")
                            st.markdown("---")

                # Add assistant message to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

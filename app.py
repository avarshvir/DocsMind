import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import DocsMindPipeline


st.set_page_config(page_title="DocsMind", page_icon="📄", layout="wide")
st.title("DocsMind")
st.caption("Conversational document chat powered by RAG")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DocsMindPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            temp_dir = tempfile.mkdtemp()
            saved_paths = []

            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(file_path)

            with st.spinner("Processing documents..."):
                st.session_state.pipeline.add_documents(saved_paths)
                st.session_state.initialized = True
                st.success("Documents indexed successfully!")

with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.divider()

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

user_question = st.chat_input("Ask something about your documents")

if user_question:
    if not st.session_state.initialized:
        st.error("Please index documents first.")
    else:
        st.session_state.chat_history.append(("user", user_question))

        history_text = "\n".join(
            f"{role.capitalize()}: {message}" for role, message in st.session_state.chat_history[:-1]
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.pipeline.generate_answer(
                    user_question,
                    chat_history=history_text,
                )
                st.markdown(answer)

                with st.expander("Sources"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        st.write(doc.page_content[:1000])
                        if doc.metadata:
                            st.json(doc.metadata)

        st.session_state.chat_history.append(("assistant", answer))
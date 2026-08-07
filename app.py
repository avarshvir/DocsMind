"""
DocsMind - Frontend UI
Provides the Streamlit interface for document uploading and conversational chat with memory.
"""

import os
import tempfile
import uuid
import streamlit as st
from pipeline import DocsMindPipeline

# 1. Page Configuration
st.set_page_config(page_title="DocsMind AI", page_icon="🧠", layout="wide")
st.title("🧠 DocsMind")
st.markdown("Upload your PDF or Word documents and have a multi-turn, context-aware conversation.")

# 2. Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DocsMindPipeline()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar Configuration
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    
    if st.button("Process Document", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing document... (Extracting, Chunking, Embedding)"):
                # Save uploaded file to a temporary file on disk
                file_ext = uploaded_file.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    num_chunks = st.session_state.pipeline.process_document(tmp_path)
                    st.success(f"Successfully processed {num_chunks} text chunks into ChromaDB!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            st.warning("Please upload a file first.")
            
    st.markdown("---")
    st.header("⚙️ Session Controls")
    if st.button("Clear Chat & Reset Memory"):
        # Reset session ID to start fresh memory in backend and clear UI chat
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("Chat history and backend memory reset!")
        st.rerun()

# 4. Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User Input and Chat Logic
if prompt := st.chat_input("Ask a question about your document..."):
    
    # Display user query in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        if st.session_state.pipeline.vector_db is None:
            st.warning("Please upload and process a document in the sidebar first.")
        else:
            # Stream generator from pipeline passing the current session_id
            stream = st.session_state.pipeline.chat_stream(
                user_query=prompt,
                session_id=st.session_state.session_id
            )
            
            # Stream live to screen and capture final output string
            full_response = st.write_stream(stream)
            
            # Save assistant response to UI state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
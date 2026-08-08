"""
DocsMind - Backend Pipeline with Conversational Memory
Handles document loading, chunking, embedding, and a memory-enabled LCEL RAG chain.
"""

import os
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# New Core Imports for LCEL and Memory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

class DocsMindPipeline:
    def __init__(self, persist_dir="./chroma_db", embed_model="nomic-embed-text", llm_model="gemma2:2b"):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=embed_model)
        
        # Increased temperature slightly (0.3) so the AI feels a bit more conversational
        self.llm = ChatOllama(model=llm_model, temperature=0.3) 
        self.vector_db = None
        
        # Dictionary to store chat histories in memory (Key: session_id, Value: History Object)
        self.store = {}
        
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)

    def process_document(self, file_path: str):
        """Loads a document, chunks it, and saves it to ChromaDB."""
        ext = os.path.splitext(file_path)[-1].lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return len(chunks)

    def _format_docs(self, docs):
        """Internal helper to format retrieved documents."""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retrieves or creates a chat history for a specific session."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def chat_stream(self, user_query: str, session_id: str = "default_session"):
        """Builds a memory-enabled LCEL chain and streams the response token-by-token."""
        if not self.vector_db:
            yield "Please upload and process a document first."
            return

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        system_prompt = (
            "You are DocsMind, a helpful AI assistant. "
            "If user asks a question, retrieve relevant context from the uploaded document and answer based on that. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If the answer is not in the context, say 'I don't know based on the provided document'. "
            "Do not hallucinate external information.\n\n"
            "If user asks simple questions like hello, how are you, knowledge question or other casual greetings, respond in a friendly manner without using the document context.\n\n"
            "Context:\n{context}"
        )

        # Updated Prompt: Added MessagesPlaceholder for the "history" variable
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # ---------------------------------------------------------
        # THE LCEL PIPELINE (With Memory Prep)
        # ---------------------------------------------------------
        rag_chain = (
            # .assign keeps 'input' and 'history', and dynamically calculates 'context'
            RunnablePassthrough.assign(
                context=itemgetter("input") | retriever | self._format_docs
            )
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Wrap the core chain in the Memory handler
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Execute the stream. Notice we must pass a dictionary with "input" now.
        for chunk in conversational_rag_chain.stream(
            {"input": user_query},
            config={"configurable": {"session_id": session_id}}
        ):
            yield chunk
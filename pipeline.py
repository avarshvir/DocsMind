from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class DocsMindPipeline:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "qwen3.5:0.8b",
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vectorstore = None
        self.llm = ChatOllama(model=self.llm_model, temperature=0.2)

        self.prompt = ChatPromptTemplate.from_template(
            """You are DocsMind, a helpful conversational assistant that answers questions only from the provided document context.

If the answer is not present in the context, say you don't know.
Be concise and conversational.

Conversation history:
{chat_history}

Context:
{context}

User question:
{question}

Answer:
"""
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            suffix = Path(file_path).suffix.lower()

            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif suffix == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif suffix == ".docx":
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            documents.extend(loader.load())
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, documents: List[Document]):
        chunks = self.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return self.vectorstore

    def load_vectorstore(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        return self.vectorstore

    def add_documents(self, file_paths: List[str]):
        documents = self.load_documents(file_paths)
        return self.create_vectorstore(documents)

    def retrieve(self, question: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search(question, k=k)

    def generate_answer(self, question: str, chat_history: str = "") -> Tuple[str, List[Document]]:
        docs = self.retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = self.prompt.format_messages(
            chat_history=chat_history,
            context=context,
            question=question,
        )
        response = self.llm.invoke(messages)
        return response.content, docs
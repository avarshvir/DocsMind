"""
DocsMind - Learning Step 3: Vector Storage with ChromaDB
--------------------------------------------------------
In this script, you will learn:
1. How to combine Loading, Chunking, and Embedding.
2. How Chroma.from_documents() processes and saves data to your hard drive.
3. How similarity_search() uses math to find relevant text.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def main():
    # ---------------------------------------------------------
    # PART 1: The Setup (Review from Steps 1 & 2)
    # ---------------------------------------------------------
    pdf_file_path = "book1.pdf"
    
    if not os.path.exists(pdf_file_path):
        print(f"Notice: Please place a '{pdf_file_path}' in the main folder to run this.")
        return

    print("1. Loading PDF...")
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()

    print("2. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    print("3. Initializing Embedding Model...")
    # This just gets the model ready; it doesn't embed anything yet.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ---------------------------------------------------------
    # PART 2: Under the Hood of Chroma
    # ---------------------------------------------------------
    persist_directory = "./chroma_db"
    
    print("\n4. Creating Vector Database (This takes time!)...")
    print("   -> INTERNALLY: Sending chunks to Ollama to get vectors.")
    print(f"   -> INTERNALLY: Saving vectors and text to folder: {persist_directory}")
    
    # MAGIC METHOD #1: from_documents
    # We pass the chunks, the embedding engine, and where to save the files.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("   -> Database created and saved successfully!")

    # ---------------------------------------------------------
    # PART 3: Searching the Database
    # ---------------------------------------------------------
    user_question = "What is the main topic of this document?"
    print(f"\n5. Performing Similarity Search for: '{user_question}'")
    print("   -> INTERNALLY: Embedding the user's question.")
    print("   -> INTERNALLY: Calculating mathematical distance between query and documents.")
    
    # MAGIC METHOD #2: similarity_search
    # 'k' is how many chunks we want to get back. By default, it usually fetches 4.
    results = vector_db.similarity_search(query=user_question, k=2)

    print("\n--- SEARCH RESULTS ---")
    for i, doc in enumerate(results, start=1):
        print(f"\nResult #{i} (From Page: {doc.metadata.get('page', 'Unknown')}):")
        # Print just the first 150 characters to keep the terminal clean
        print(f"Text: {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()
"""
DocsMind - Learning Step 4: CLI Application 1 (Search Only)
-----------------------------------------------------------
In this script, you will learn:
1. How to load an EXISTING Chroma vector database from disk.
2. Why we must provide the embedding model to read the DB.
3. How to use similarity_search_with_score to see the mathematical distance.
4. How to build a continuous loop for user querying.
"""

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def load_existing_database(persist_directory: str = "./chroma_db"):
    """Connects to the previously saved ChromaDB."""
    if not os.path.exists(persist_directory):
        print(f"Error: Database folder '{persist_directory}' not found.")
        print("Please run '3_vector_storage.py' first to generate the database.")
        return None

    print("[1/2] Initializing Embedding Model (to embed user queries)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"[2/2] Connecting to existing database at '{persist_directory}'...")
    # By initializing Chroma this way, we do NOT overwrite or recreate the data.
    # We simply open a connection to the existing files and tell it how to embed new queries.
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_db


def main():
    print("==================================================")
    print("       DocsMind - Document Search CLI             ")
    print("==================================================")
    
    # Connect to the database we built in Step 3
    db = load_existing_database()
    if db is None:
        return

    print("\nSystem ready! You can now search your documents.")
    print("Type 'exit' or 'quit' to stop.\n")

    # Start an infinite loop to keep the CLI running
    while True:
        # Get user input
        query = input("\nEnter your search query: ")
        
        # Check if user wants to exit
        if query.lower() in ['exit', 'quit']:
            print("Exiting DocsMind CLI. Goodbye!")
            break
            
        if not query.strip():
            continue

        print(f"\nSearching for: '{query}'...")
        
        # MAGIC METHOD EXPOSED: similarity_search_with_score
        # Returns a list of tuples: (Document_Object, Float_Score)
        # k=3 means we want the top 3 closest matches.
        results = db.similarity_search_with_score(query=query, k=3)

        print("\n" + "-"*40)
        print("TOP 3 MOST RELEVANT CHUNKS FOUND:")
        print("-"*40)
        
        for i, (doc, score) in enumerate(results, start=1):
            # Remember: Lower score = closer distance = better match
            print(f"\n[Match #{i}] - L2 Distance Score: {score:.4f}")
            print(f"Source Page: {doc.metadata.get('page', 'N/A')}")
            
            # We strip whitespace and limit to 300 characters to keep it readable
            content_preview = doc.page_content.strip()[:300]
            print(f"Text: {content_preview}...")


if __name__ == "__main__":
    main()
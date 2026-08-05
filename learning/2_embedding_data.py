"""
DocsMind - Learning Step 2: Vector Embeddings
---------------------------------------------
In this script, you will learn:
1. How to initialize a local embedding model using Ollama.
2. How to embed a single user query.
3. How to embed multiple document chunks.
4. What the resulting vector data actually looks like.
"""

from langchain_ollama import OllamaEmbeddings

def get_embedding_model(model_name: str = "nomic-embed-text"):
    """Initializes and returns the Ollama embedding model."""
    print(f"[1/3] Initializing OllamaEmbeddings with model: '{model_name}'...")
    
    # Initialize the embedding model pointing to your local Ollama instance
    embeddings = OllamaEmbeddings(
        model=model_name,
    )
    return embeddings


def test_query_embedding(embeddings):
    """Embeds a single string (like a user's search query)."""
    print("\n[2/3] Testing single query embedding...")
    
    sample_query = "How do I install this application?"
    print(f"Text to embed: '{sample_query}'")
    
    # embed_query transforms a single string into a vector
    query_vector = embeddings.embed_query(sample_query)
    
    print(f"--> Success! Generated a vector with {len(query_vector)} dimensions.")
    print(f"--> Preview of vector (first 5 values): {query_vector[:5]}")
    
    return query_vector


def test_document_embeddings(embeddings):
    """Embeds a list of strings (like your chunked document pages)."""
    print("\n[3/3] Testing bulk document chunk embeddings...")
    
    # Simulating the chunks we created in Step 1
    sample_chunks = [
        "DocsMind requires Python 3.10 or higher.",
        "To start the server, run the app.py file using streamlit.",
        "Vector databases store embeddings for fast semantic retrieval."
    ]
    
    print(f"Embedding {len(sample_chunks)} document chunks...")
    
    # embed_documents transforms a list of strings into a list of vectors
    doc_vectors = embeddings.embed_documents(sample_chunks)
    
    print(f"--> Success! Generated {len(doc_vectors)} vectors.")
    print(f"--> Each vector has {len(doc_vectors[0])} dimensions.")
    

def main():
    try:
        # 1. Load the model
        embeddings = get_embedding_model()
        
        # 2. Test query embedding
        test_query_embedding(embeddings)
        
        # 3. Test document embeddings
        test_document_embeddings(embeddings)
        
        print("\nAll embedding tests passed! Your local Ollama instance is working perfectly.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Make sure Ollama is running and you have pulled the model using: 'ollama pull nomic-embed-text'")


if __name__ == "__main__":
    main()
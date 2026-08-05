"""
DocsMind - Learning Step 5: CLI Application 2 (RAG Chat without Memory)
-----------------------------------------------------------------------
In this script, you will learn:
1. How to initialize a local generative LLM (ChatOllama).
2. How to create a system prompt template that forces the AI to use context.
3. How to link the database search and the LLM together using chains.
"""

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_database():
    """Connects to the existing Chroma vector database."""
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        print(f"Error: Database folder '{persist_directory}' not found. Run Step 3 first.")
        return None

    # We still need the embedding model so Chroma can convert the user's typed question into a vector
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # We turn the database into a "retriever" object. 
    # search_kwargs={"k": 3} means it will always grab the 3 most relevant chunks.
    return vector_db.as_retriever(search_kwargs={"k": 3})

def create_rag_chain(retriever):
    """Builds the pipeline connecting the retriever to the LLM."""
    print("[1/2] Initializing the Generative LLM...")
    llm = ChatOllama(
        model="qwen3.5:0.8b",
        temperature=0.2 # Low temperature = less creative, more strictly factual
    )

    print("[2/2] Building the prompt template and chains...")
    
    # The System Prompt tells the AI its job. 
    # {context} is a special LangChain variable where the document chunks will be pasted.
    system_prompt_text = (
        "You are an intelligent assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know based on the document'. "
        "Do not make up information outside of the context.\n\n"
        "Context:\n{context}"
    )

    # Convert our text into an official LangChain Prompt object
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{input}")
    ])

    # 1. The QA Chain: Tells the system how to format the prompt and call the LLM
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

    # 2. The Retrieval Chain: Connects the database (retriever) to the QA chain
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)
    
    return rag_chain


def main():
    print("==================================================")
    print("       DocsMind - AI Document Chat CLI            ")
    print("==================================================")
    
    retriever = load_database()
    if not retriever:
        return

    rag_chain = create_rag_chain(retriever)
    
    print("\nSystem ready! Ask questions about your document.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("\nYou: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting DocsMind Chat. Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("\nDocsMind is thinking...")
        
        # MAGIC METHOD EXPOSED: invoke()
        # This triggers the entire pipeline: 
        # User Query -> Vector Embed -> Chroma Search -> Stuff Prompt -> LLM Generate -> Output
        response = rag_chain.invoke({"input": user_query})
        
        # The response dictionary contains the AI's final text in the "answer" key
        print(f"\nAI: {response['answer']}")
        
        # (Optional) You can also see the exact chunks it used by looking at response['context']
        # print("\n--- [Debug: Sources Used] ---")
        # for doc in response['context']:
        #     print(f"Page {doc.metadata.get('page')}")

if __name__ == "__main__":
    main()
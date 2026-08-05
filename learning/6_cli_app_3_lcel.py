"""
DocsMind - Learning Step 6: RAG via LangChain Expression Language (LCEL)
------------------------------------------------------------------------
In this script, you will learn:
1. How to use the pipe (|) operator to build a transparent data pipeline.
2. How to use RunnablePassthrough and StrOutputParser.
3. How to manually format the retrieved documents before they hit the prompt.
"""

import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_database():
    """Connects to the existing Chroma vector database."""
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        print(f"Error: Database folder '{persist_directory}' not found. Run Step 3 first.")
        return None

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    return vector_db.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    """
    Helper function for LCEL.
    Takes a list of Document objects and joins their text into a single string
    separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_lcel_rag_chain(retriever):
    """Builds the LCEL pipeline connecting the retriever to the LLM."""
    print("[1/2] Initializing ChatOllama...")
    llm = ChatOllama(model="gemma2:2b", temperature=0.2)

    print("[2/2] Assembling the LCEL Pipe (|)...")
    
    system_prompt = (
        "You are an intelligent assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know based on the document'. "
        "Do not make up information outside of the context.\n\n"
        "Context:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # ---------------------------------------------------------
    # THE LCEL PIPELINE
    # ---------------------------------------------------------
    # This reads top-to-bottom, exactly how the data flows.
    rag_chain = (
        # 1. Define inputs: 
        # - 'context' gets the documents from the retriever and formats them.
        # - 'input' passes the user's string straight through.
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        
        # 2. Feed those dictionary keys into the Prompt Template
        | prompt_template 
        
        # 3. Feed the formatted prompt to the local LLM
        | llm 
        
        # 4. Parse the complex LLM output object back into a simple string
        | StrOutputParser()
    )
    
    return rag_chain


def main():
    print("==================================================")
    print("   DocsMind - Advanced AI Chat CLI (LCEL Engine)  ")
    print("==================================================")
    
    retriever = load_database()
    if not retriever:
        return

    rag_chain = create_lcel_rag_chain(retriever)
    
    print("\nSystem ready! LCEL Pipeline is active.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("\nYou: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting DocsMind CLI. Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("\nDocsMind is thinking...")
        
        # Because we used StrOutputParser() at the end of our LCEL chain,
        # invoke() now returns a clean string directly, instead of a dictionary!
        response = rag_chain.invoke(user_query)
        
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()
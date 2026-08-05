"""
DocsMind - Learning Step 1: Data Loading & Text Chunking
--------------------------------------------------------
In this script, you will learn:
1. How to load PDF documents using PyPDFLoader.
2. How to split raw document text into semantically chunked pieces.
3. How to inspect Document objects, page contents, and metadata.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_document(file_path: str):
    """Loads a PDF file and returns a list of page Document objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[1/3] Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"--> Successfully loaded {len(pages)} page(s).")
    return pages


def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits loaded document pages into smaller chunks."""
    print(f"[2/3] Chunking documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    # Initialize the recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"--> Split {len(documents)} page(s) into {len(chunks)} total text chunk(s).")
    return chunks


def inspect_sample_data(pages, chunks):
    """Prints diagnostic information about loaded pages and generated chunks."""
    print("\n" + "=" * 50)
    print("[3/3] DATA INSPECTION")
    print("=" * 50)

    # Raw page example
    if pages:
        print("\n--- SAMPLE PAGE RAW METADATA ---")
        print(pages[0].metadata)
        print("\n--- SAMPLE PAGE CONTENT (First 200 chars) ---")
        print(f"{pages[0].page_content[:200]}...")

    # Split chunk example
    if chunks:
        print("\n--- SAMPLE CHUNK 1 METADATA ---")
        print(chunks[0].metadata)
        print("\n--- SAMPLE CHUNK 1 TEXT ---")
        print(chunks[0].page_content)
        
        if len(chunks) > 1:
            print("\n--- SAMPLE CHUNK 2 TEXT ---")
            print(chunks[1].page_content)


def main():
    # Provide a path to any sample PDF file in your local workspace
    pdf_file_path = "book1.pdf"

    if not os.path.exists(pdf_file_path):
        print(f"Notice: Place a test PDF at '{pdf_file_path}' or update 'pdf_file_path' in the script to run this test.")
        return

    # Load
    raw_pages = load_pdf_document(pdf_file_path)
    
    # Chunk
    doc_chunks = chunk_documents(raw_pages, chunk_size=1000, chunk_overlap=200)
    
    # Inspect
    inspect_sample_data(raw_pages, doc_chunks)


if __name__ == "__main__":
    main()
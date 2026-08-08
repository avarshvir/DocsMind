# DocsMind

DocsMind is a simple document chat app. You upload documents, index them, and then ask questions about the uploaded content.

The app uses:

- Streamlit for the web interface
- LangChain for the RAG pipeline
- ChromaDB for storing document embeddings
- Ollama for local embeddings and chat responses

## Features

- Upload PDF, TXT, and DOCX files
- Split documents into smaller chunks
- Store document chunks in a local Chroma vector database
- Ask questions in a chat interface
- See the source text used to answer each question
- Keep short chat history during the session

## Project Files

- `app.py` - Streamlit user interface
- `pipeline.py` - document loading, chunking, retrieval, and answer generation
- `requirements.txt` - Python packages needed to run the app
- `chroma_db/` - local vector database created by Chroma

## Requirements

You need Python and Ollama installed on your machine.

Install and start Ollama, then pull the models used by the project:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:0.8b
```

If you want to use another Ollama chat model, change the `llm_model` value in `pipeline.py`.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then open the local Streamlit link in your browser.

## How To Use

1. Upload one or more PDF, TXT, or DOCX files.
2. Click `Index Documents`.
3. Wait until the documents are processed.
4. Ask a question in the chat box.
5. Open `Sources` if you want to see the text used for the answer.

## How It Works

DocsMind loads the uploaded files and converts them into text documents. It splits the text into chunks, creates embeddings with Ollama, and stores those chunks in ChromaDB.

When you ask a question, the app searches for the most relevant chunks. Those chunks are sent to the local Ollama chat model, which writes an answer using the retrieved document context.

## Evaluation & System Telemetry

DocsMind includes an automated **LLM-as-a-Judge telemetry suite** (`evaluate.py`) that evaluates RAG accuracy locally without exposing data to external APIs.

### Evaluation Metrics
1. **Context Relevance (1–5):** Measures whether ChromaDB successfully retrieves the precise context needed for the query.
2. **Faithfulness (1–5):** Evaluates whether the generated response strictly aligns with facts in the source document without hallucinating.
3. **Guardrail Compliance:** Verifies intent routing between document-grounded queries and general conversational interactions.

### Benchmark Results (Sample Test Run)
* **Fact Extraction (Operation Searchlight):** Relevance `5/5` | Faithfulness `5/5`
* **Comparative Retrieval (Political Stability Index):** Relevance `5/5` | Faithfulness `5/5`
* **Intent Routing / Fallback (Out-of-domain query):** Handled via prompt guardrails.

### Running the Evaluation Suite
1. Ensure your target document is processed in ChromaDB.
2. Run the evaluation script:
   ```
   python evaluate.py
   ```

## License

This project is licensed under the MIT License.

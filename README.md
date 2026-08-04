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

## Evaluation

DocsMind is a good basic RAG project. It is easy to understand, runs locally, and gives users a simple way to ask questions about their own documents. The source viewer is useful because it helps users check where an answer came from.

The project is best for small or medium document collections. It should work well for notes, reports, assignments, manuals, and simple research files. Since it uses local Ollama models, it can also be useful when the user does not want to send documents to an external API.

There are some limitations. The app currently rebuilds the vector database when new documents are indexed, and it does not show document names clearly in the chat answer. The answer quality depends heavily on the local Ollama model. If the model is small, answers may be short, incomplete, or less accurate. The app also does not yet include login, saved chat history, advanced file management, or automatic evaluation metrics.

Overall, DocsMind is a solid starting point. It shows the main idea of retrieval-augmented generation in a clean and practical way. With better document management, stronger models, improved source citations, and more testing, it could become a more complete document assistant.

## License

This project is licensed under the MIT License.

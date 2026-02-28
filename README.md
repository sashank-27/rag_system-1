# Multilingual RAG System with ServiceNow Integration

A production-grade **Multilingual Retrieval-Augmented Generation** system built with FastAPI. Supports PDF documents in any language (Hindi, Gujarati, English, and more), semantic search via Qdrant, and ServiceNow CMDB integration with automatic intent routing.

---

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌───────────────┐
│  PDF Upload  │────▶│  Text Extract   │────▶│   Chunking    │
│  (any lang)  │     │ PyMuPDF + OCR   │     │ 600 chars/100 │
└─────────────┘     └────────────────┘     └───────┬───────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌────────────────┐     ┌───────────────┐
│   Answer     │◀───│   LLM (Qwen)   │◀───│   Reranker    │
│ (same lang)  │     │  + RAG Prompt   │     │ Cross-Encoder │
└─────────────┘     └────────────────┘     └───────┬───────┘
                                                    │
                                                    ▲
┌─────────────┐     ┌────────────────┐     ┌───────────────┐
│  User Query  │────▶│   Embedding    │────▶│    Qdrant     │
│  (any lang)  │     │   BAAI/bge-m3  │     │  Vector DB    │
└─────────────┘     └────────────────┘     └───────────────┘
```

---

## Prerequisites

| Requirement     | Notes                                                   |
|-----------------|---------------------------------------------------------|
| **Python**      | 3.10+                                                    |
| **Tesseract**   | Required for scanned PDF OCR                            |
| **GPU (optional)** | NVIDIA GPU with ≥16 GB VRAM recommended for LLM     |
| **Disk space**  | ~15 GB for model downloads (embedding + LLM + reranker) |

### Install Tesseract (Windows)

```powershell
# Option 1: Chocolatey
choco install tesseract

# Option 2: Manual
# Download from https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH or set TESSERACT_CMD in .env
```

---

## Quick Start

### 1. Clone & enter the project

```bash
cd rag_system
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
# Edit .env with your settings (ServiceNow credentials, Tesseract path, etc.)
```

### 5. Run the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation is available at: **http://localhost:8000/docs**

---

## API Endpoints

| Method   | Endpoint                  | Description                        |
|----------|---------------------------|------------------------------------|
| `GET`    | `/health`                 | Health check                       |
| `POST`   | `/upload`                 | Upload and index a PDF             |
| `POST`   | `/ask`                    | Ask a question (RAG or ServiceNow) |
| `GET`    | `/documents`              | List all indexed documents         |
| `DELETE` | `/documents/{document_id}`| Delete a document                  |
| `POST`   | `/servicenow/host`        | Direct ServiceNow CMDB lookup      |

---

## Example curl Requests

### Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "document_id": "a1b2c3d4e5f6...",
  "filename": "document.pdf",
  "total_chunks": 24,
  "detected_language": "hi",
  "message": "Document uploaded and indexed successfully."
}
```

### Ask a question (English)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of this report?"}'
```

### Ask a question (Hindi)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "इस दस्तावेज़ का मुख्य विषय क्या है?"}'
```

**Response:**
```json
{
  "question": "इस दस्तावेज़ का मुख्य विषय क्या है?",
  "answer": "इस दस्तावेज़ का मुख्य विषय...",
  "detected_language": "hi",
  "source_documents": [
    {
      "filename": "report.pdf",
      "page_number": 3,
      "score": 0.8923,
      "snippet": "..."
    }
  ],
  "routed_to": "rag"
}
```

### Ask about a server (auto-routes to ServiceNow)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about host server01"}'
```

### Direct ServiceNow lookup

```bash
curl -X POST http://localhost:8000/servicenow/host \
  -H "Content-Type: application/json" \
  -d '{"host": "server01"}'
```

**Response:**
```json
{
  "name": "server01",
  "ip_address": "10.0.1.50",
  "os": "Linux Red Hat",
  "location": "US-East DC1",
  "install_status": "1"
}
```

### List documents

```bash
curl http://localhost:8000/documents
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/documents/a1b2c3d4e5f6
```

---

## Project Structure

```
rag_system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, routers
│   ├── config.py               # Pydantic settings from .env
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Request/response models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── documents.py        # /upload, /documents, /documents/{id}
│   │   ├── query.py            # /ask with intent routing
│   │   └── servicenow_routes.py# /servicenow/host
│   └── services/
│       ├── __init__.py
│       ├── embedding.py        # BAAI/bge-m3 embeddings
│       ├── language_detector.py# langdetect wrapper
│       ├── llm.py              # Qwen2.5-Instruct generation
│       ├── pdf_processor.py    # PyMuPDF + Tesseract OCR + chunking
│       ├── reranker.py         # Cross-encoder reranker
│       ├── retrieval.py        # Qdrant vector store
│       └── servicenow.py       # ServiceNow CMDB client
├── vector_db/                  # Qdrant local storage (auto-created)
├── .env.example                # Environment template
├── requirements.txt
└── README.md
```

---

## Configuration Reference

| Variable              | Default                        | Description                    |
|-----------------------|--------------------------------|--------------------------------|
| `QDRANT_PATH`         | `./vector_db`                  | Qdrant local storage path      |
| `QDRANT_COLLECTION`   | `rag_documents`                | Collection name                |
| `EMBEDDING_MODEL`     | `BAAI/bge-m3`                  | Multilingual embedding model   |
| `LLM_MODEL`           | `Qwen/Qwen2.5-7B-Instruct`    | LLM for answer generation      |
| `RERANKER_MODEL`      | `BAAI/bge-reranker-base`       | Reranker model                 |
| `CHUNK_SIZE`          | `600`                          | Chunk size in characters       |
| `CHUNK_OVERLAP`       | `100`                          | Overlap between chunks         |
| `TOP_K`               | `5`                            | Number of results to retrieve  |
| `MAX_FILE_SIZE_MB`    | `50`                           | Max upload file size           |
| `TESSERACT_CMD`       | `tesseract`                    | Path to Tesseract binary       |
| `SERVICENOW_INSTANCE` | —                              | ServiceNow instance name       |
| `SERVICENOW_USERNAME` | —                              | ServiceNow username            |
| `SERVICENOW_PASSWORD` | —                              | ServiceNow password            |

---

## Key Design Decisions

- **No Docker** — Qdrant runs in embedded/local mode via `qdrant-client`.
- **Lazy model loading** — Embedding, LLM, and reranker models load on first use, keeping startup fast.
- **Anti-hallucination** — System prompt strictly constrains the LLM to answer only from provided context.
- **Same-language response** — Language detection drives prompt construction to enforce output in the query's language.
- **Intent routing** — Keywords like `host`, `server`, `cmdb` route queries to ServiceNow instead of the RAG pipeline.

---

## License

MIT

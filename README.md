# RAG Support Bot Demo

A RAG-based customer support chatbot built with FastAPI, ChromaDB, and OpenAI. It ingests a FAQ dataset into a vector store and uses retrieval-augmented generation to answer customer questions.

## Setup

```bash
git clone https://github.com/eldencodingv3/rag-support-bot-demo.git
cd rag-support-bot-demo
pip install -r requirements.txt
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings and chat completions |
| `PORT` | No | `8000` | Port the server listens on |

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

## Running

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will automatically ingest the FAQ dataset into ChromaDB on startup.

## API Endpoints

### `GET /api/health`

Health check endpoint.

**Response:** `{"status": "ok"}`

### `POST /api/chat`

Send a customer support question to the RAG pipeline.

**Request:**
```json
{"question": "How do I reset my password?"}
```

**Response:**
```json
{
  "answer": "To reset your password, go to the login page and click...",
  "sources": [
    {"question": "How do I reset my password?", "category": "account"}
  ]
}
```

### `GET /`

Serves the chat UI from `static/index.html`.

## Updating the Dataset

Edit `dataset/faq.json` to add, modify, or remove FAQ entries. Each entry needs:

```json
{"question": "...", "answer": "...", "category": "..."}
```

Categories: `account`, `billing`, `technical`, `shipping`, `returns`, `general`

Restart the server after making changes — the dataset is re-ingested on startup.

## Deployment

The app includes a `Procfile` for platforms like Railway or Heroku. Set the `OPENAI_API_KEY` environment variable in your deployment platform and deploy from the repo.

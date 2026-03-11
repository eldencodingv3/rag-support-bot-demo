# RAG Support Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions using a FAQ dataset, OpenAI embeddings, and ChromaDB vector search.

## Tech Stack

- **Backend**: Python + FastAPI
- **Vector Store**: ChromaDB (in-memory)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Frontend**: Vanilla HTML/CSS/JS

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/eldencodingv3/rag-support-bot-demo.git
   cd rag-support-bot-demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

4. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. Open http://localhost:8000 in your browser.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | Your OpenAI API key |
| PORT | No | 8000 | Server port |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Chat UI |
| GET | /api/health | Health check |
| POST | /api/chat | Send a question, get a RAG answer |

### POST /api/chat

Request:
```json
{"question": "How do I reset my password?"}
```

Response:
```json
{
  "answer": "To reset your password, click 'Forgot Password' on the login page...",
  "sources": [{"question": "...", "category": "account", "relevance": 0.95}]
}
```

## Updating the Dataset

Edit `dataset/faq.json` to add, remove, or modify FAQ entries. Each entry needs:
```json
{"question": "...", "answer": "...", "category": "..."}
```

Restart the application after updating the dataset.

## Deployment

Deployed on Railway with auto-deploy from the main branch.

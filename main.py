import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from ingest import ingest_dataset
from rag import query_fallback, query_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ingest FAQ dataset into ChromaDB
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            ingest_dataset()
            print("Dataset ingested successfully")
        except Exception as e:
            print(f"Warning: Failed to ingest dataset: {e}")
    else:
        print("Warning: OPENAI_API_KEY not set — skipping dataset ingestion")
    yield


app = FastAPI(title="RAG Support Bot", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    question: str
    category: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if os.environ.get("OPENAI_API_KEY"):
            result = query_rag(request.question)
        else:
            result = query_fallback(request.question)
        return ChatResponse(**result)
    except Exception as e:
        return ChatResponse(
            answer=f"An error occurred while processing your question: {str(e)}",
            sources=[],
        )


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# Mount static files AFTER API routes so /api/* takes priority
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

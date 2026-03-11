import json
import os
import re

from openai import OpenAI
import chromadb

_client = None
_collection = None

SYSTEM_PROMPT = """You are a helpful customer support assistant. Answer the customer's question based ONLY on the provided context below. Be concise, friendly, and accurate.

If the provided context does not contain enough information to answer the question, say "I don't have information about that. Please contact our support team for further assistance."

Do not make up information that is not in the context."""


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def init_vector_store():
    """Create an ephemeral ChromaDB client and collection."""
    global _client, _collection
    _client = chromadb.Client()
    _collection = _client.get_or_create_collection(
        name="support_faq",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def get_collection():
    """Return the current collection, initializing if needed."""
    global _collection
    if _collection is None:
        init_vector_store()
    return _collection


def ingest_documents(documents: list[dict]):
    """Ingest FAQ entries into ChromaDB with OpenAI embeddings."""
    collection = get_collection()
    openai_client = _get_openai_client()

    ids = []
    texts = []
    metadatas = []
    for i, doc in enumerate(documents):
        ids.append(f"faq_{i}")
        texts.append(doc["answer"])
        metadatas.append({
            "question": doc["question"],
            "category": doc.get("category", "general"),
        })

    # Generate embeddings in a single batch
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    embeddings = [item.embedding for item in response.data]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Ingested {len(documents)} documents into ChromaDB")


def query_rag(question: str) -> dict:
    """Query the RAG pipeline: embed question, retrieve context, generate answer."""
    collection = get_collection()
    openai_client = _get_openai_client()

    # Embed the question
    embed_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    query_embedding = embed_response.data[0].embedding

    # Retrieve top 5 results from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    # Build context from retrieved documents
    context_parts = []
    sources = []
    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        context_parts.append(
            f"Q: {metadata['question']}\nA: {doc}"
        )
        sources.append({
            "question": metadata["question"],
            "category": metadata["category"],
        })

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer using OpenAI chat completion
    chat_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nCustomer question: {question}",
            },
        ],
    )

    answer = chat_response.choices[0].message.content
    return {"answer": answer, "sources": sources}


_faq_data = None


def _load_faq():
    global _faq_data
    if _faq_data is None:
        faq_path = os.path.join(os.path.dirname(__file__), "dataset", "faq.json")
        with open(faq_path, "r") as f:
            _faq_data = json.load(f)
    return _faq_data


def _keyword_score(query: str, text: str) -> float:
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def query_fallback(question: str) -> dict:
    faq = _load_faq()
    scored = []
    for entry in faq:
        q_score = _keyword_score(question, entry["question"])
        a_score = _keyword_score(question, entry["answer"])
        score = max(q_score, a_score * 0.8)
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    if top[0][0] < 0.1:
        return {
            "answer": "I don't have specific information about that in our FAQ. Please contact our support team for further assistance.",
            "sources": [],
            "mode": "fallback"
        }
    best = top[0][1]
    answer = best["answer"]
    sources = [{"question": e["question"], "category": e.get("category", "general"), "relevance": round(s, 3)} for s, e in top if s > 0.1]
    return {"answer": answer, "sources": sources, "mode": "fallback"}

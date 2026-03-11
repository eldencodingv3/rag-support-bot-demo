import os
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

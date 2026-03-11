import os
from openai import OpenAI
import chromadb

client_db = chromadb.Client()
collection = None


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def init_collection():
    global collection
    collection = client_db.get_or_create_collection(
        name="support_faq",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embedding(text: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def ingest_documents(documents: list[dict]):
    col = init_collection()
    if col.count() > 0:
        return

    ids = []
    docs = []
    metadatas = []
    embeddings = []

    for i, doc in enumerate(documents):
        text = f"Question: {doc['question']}\nAnswer: {doc['answer']}"
        ids.append(f"faq_{i}")
        docs.append(text)
        metadatas.append({"question": doc["question"], "category": doc.get("category", "general")})

    # Batch embed
    client = get_openai_client()
    batch_size = 20
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for item in response.data:
            embeddings.append(item.embedding)

    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)


def query_rag(question: str) -> dict:
    global collection
    if collection is None:
        collection = init_collection()

    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=5, include=["documents", "metadatas", "distances"])

    context_parts = []
    sources = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            context_parts.append(doc)
            if results["metadatas"] and results["metadatas"][0]:
                sources.append({
                    "question": results["metadatas"][0][i].get("question", ""),
                    "category": results["metadatas"][0][i].get("category", ""),
                    "relevance": round(1 - (results["distances"][0][i] if results["distances"] else 0), 3)
                })

    context = "\n\n---\n\n".join(context_parts)

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. Answer the user's question based ONLY on the "
                    "provided context below. If the context does not contain enough information to answer the "
                    "question, say 'I don't have specific information about that in our FAQ. Please contact our "
                    "support team for further assistance.' Be concise and friendly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources[:3]}

import json
import os


def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return json.load(f)


def ingest_dataset():
    filepath = os.path.join(os.path.dirname(__file__), "dataset", "faq.json")
    if not os.path.exists(filepath):
        print(f"Warning: Dataset not found at {filepath}")
        return
    documents = load_dataset(filepath)
    print(f"Loaded {len(documents)} FAQ entries")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Skipping document ingestion.")
        return

    from rag import ingest_documents
    ingest_documents(documents)
    print("Documents ingested into vector store")

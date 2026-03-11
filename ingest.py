import json
import os
from rag import ingest_documents


def load_dataset(filepath: str) -> list[dict]:
    """Load FAQ entries from a JSON file."""
    abs_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(abs_path, "r") as f:
        data = json.load(f)
    return data


def ingest_dataset():
    """Load the FAQ dataset and ingest it into the vector store."""
    documents = load_dataset("dataset/faq.json")
    print(f"Loaded {len(documents)} FAQ entries")
    ingest_documents(documents)

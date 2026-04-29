"""
Vector store tool using ChromaDB (cross-platform).
The architecture mirrors Milvus: local persistent store, COSINE similarity,
sentence-transformers embeddings. Named "milvus_rag" for API compatibility.
"""
from __future__ import annotations

import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

COLLECTION_NAME = "context_engineering"
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


class MilvusRAGInput(BaseModel):
    query: str = Field(..., description="Query to search the document vector store")
    top_k: int = Field(5, description="Number of top results to retrieve")


class MilvusRAGTool(BaseTool):
    name: str = "milvus_rag_search"
    description: str = (
        "Search the local vector database for relevant document chunks "
        "from uploaded PDFs and research papers. Use for retrieving content "
        "from documents that have been indexed into the system."
    )
    args_schema: Type[BaseModel] = MilvusRAGInput

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            return f"Required library not installed: {e}. Run: uv add chromadb sentence-transformers"

        client = chromadb.PersistentClient(path=DB_PATH)

        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            return "No documents indexed yet. Upload PDFs via POST /api/documents/upload first."

        if collection.count() == 0:
            return "No documents indexed yet. Upload PDFs via POST /api/documents/upload first."

        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vector = model.encode([query])[0].tolist()

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return f"Vector search failed: {e}"

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            source = meta.get("source", "unknown") if meta else "unknown"
            score = 1.0 - dist  # ChromaDB returns L2/cosine distance; convert to similarity
            chunks.append(f"[Score: {score:.3f}] Source: {source}\n{doc}")

        return "\n---\n".join(chunks) if chunks else "No relevant documents found."


def init_vector_collection():
    """Ensure the ChromaDB collection exists. Returns client."""
    try:
        import chromadb
    except ImportError:
        return None

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client


# Keep backward-compatible alias used by tensorlake_tool
init_milvus_collection = init_vector_collection

from __future__ import annotations

import os
import textwrap
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_assistant.tools.milvus_rag_tool import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    init_vector_collection,
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def _parse_pdf_pypdf(file_path: str) -> list[str]:
    """Extract text from PDF using pypdf (cross-platform fallback)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is not installed. Run: uv add pypdf")

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return pages


def _parse_pdf_tensorlake(file_path: str) -> list[str]:
    """Extract text from PDF using Tensorlake DocumentAI."""
    api_key = os.environ.get("TENSORLAKE_API_KEY")
    if not api_key:
        raise ValueError("TENSORLAKE_API_KEY environment variable is not set.")

    from tensorlake.documentai import DocumentAI, ParsingOptions, ChunkingStrategy

    dai = DocumentAI(api_key=api_key)
    opts = ParsingOptions(chunking_strategy=ChunkingStrategy.SENTENCE)
    file_id = dai.upload(path=file_path)
    parse_id = dai.parse(file=file_id, parsing_options=opts)
    result = dai.wait_for_completion(parse_id)
    return [
        chunk.content
        for chunk in result.chunks
        if chunk.content and chunk.content.strip()
    ]


def _chunk_text(pages: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Split page texts into overlapping word-level chunks."""
    chunks = []
    for page_text in pages:
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
            if start >= len(words):
                break
    return chunks if chunks else pages


class TensorlakeInput(BaseModel):
    file_path: str = Field(..., description="Absolute path to PDF file to parse and index")
    source_name: str = Field(..., description="Human-readable name for this document")


class TensorlakeParserTool(BaseTool):
    name: str = "tensorlake_parse_and_index"
    description: str = (
        "Parse a PDF, extract text chunks, embed them using "
        "sentence-transformers, and store in the local vector database "
        "for later retrieval. Uses Tensorlake when available, otherwise "
        "falls back to pypdf."
    )
    args_schema: Type[BaseModel] = TensorlakeInput

    def _run(self, file_path: str, source_name: str) -> str:
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"

        chunks: list[str] = []
        used_parser = ""

        try:
            chunks = _parse_pdf_tensorlake(file_path)
            used_parser = "Tensorlake"
        except (ImportError, ValueError) as e:
            print(f"[PDF] Tensorlake unavailable ({e}), falling back to pypdf")
        except Exception as e:
            print(f"[PDF] Tensorlake parsing failed ({e}), falling back to pypdf")

        if not chunks:
            try:
                pages = _parse_pdf_pypdf(file_path)
                if not pages:
                    return f"No text extracted from '{source_name}'."
                chunks = _chunk_text(pages, CHUNK_SIZE, CHUNK_OVERLAP)
                used_parser = "pypdf"
            except ImportError as e:
                return f"{e}. Run: uv add pypdf"
            except Exception as e:
                return f"PDF parsing failed: {e}"

        if not chunks:
            return f"No text extracted from '{source_name}'."

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return "sentence-transformers is not installed. Run: uv add sentence-transformers"

        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            vectors = model.encode(chunks).tolist()
        except Exception as e:
            return f"Embedding failed: {e}"

        try:
            client = init_vector_collection()
            if client is None:
                return "chromadb is not installed. Run: uv add chromadb"

            collection = client.get_collection(COLLECTION_NAME)

            existing_count = collection.count()
            ids = [f"{source_name}_{existing_count + i}" for i in range(len(chunks))]
            metadatas = [
                {"source": source_name, "chunk_id": existing_count + i}
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=chunks,
                metadatas=metadatas,
            )
        except Exception as e:
            return f"Vector DB insert failed: {e}"

        return (
            f"Indexed {len(chunks)} chunks from '{source_name}' into vector database "
            f"(parser: {used_parser})."
        )

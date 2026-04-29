from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ContextEvaluationOutput(BaseModel):
    relevant_sources: list[str] = Field(
        description="List of source identifiers that passed the relevance threshold"
    )
    filtered_context: str = Field(
        description="Concatenated, deduplicated context from relevant sources only"
    )
    relevance_scores: dict[str, float] = Field(
        description="Map of source_id to relevance score between 0.0 and 1.0"
    )


class FlowState(BaseModel):
    session_id: str = ""
    user_id: str = "default_user"
    query: str = ""
    rag_context: str = ""
    memory_context: str = ""
    web_context: str = ""
    arxiv_context: str = ""
    evaluation: Optional[ContextEvaluationOutput] = None
    final_response: str = ""
    status: str = "pending"  # pending | running | complete | error
    error: Optional[str] = None

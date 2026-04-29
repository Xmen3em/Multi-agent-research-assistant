from __future__ import annotations

from typing import Optional
from pydantic import BaseModel
from research_assistant.schemas import ContextEvaluationOutput


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    status: str  # pending | running | complete | error
    final_response: Optional[str] = None
    evaluation: Optional[ContextEvaluationOutput] = None
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"

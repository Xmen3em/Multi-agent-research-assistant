from __future__ import annotations

import os
import tempfile
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from research_assistant.api.schemas import (
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from research_assistant.flow import ContextEngineeringFlow
from research_assistant.schemas import FlowState

router = APIRouter()

_sessions: Dict[str, FlowState] = {}

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "research_uploads")


def _run_flow(session_id: str, query: str, user_id: str) -> None:
    """Blocking function executed inside FastAPI BackgroundTasks thread."""
    state = _sessions.get(session_id)
    if state is None:
        return

    state.status = "running"
    try:
        flow = ContextEngineeringFlow()
        flow.kickoff(
            inputs={
                "session_id": session_id,
                "query": query,
                "user_id": user_id,
            }
        )
        _sessions[session_id] = flow.state
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        _sessions[session_id] = state


@router.post("/query", response_model=QueryResponse)
async def submit_query(payload: QueryRequest, background_tasks: BackgroundTasks):
    session_id = payload.session_id or str(uuid.uuid4())
    _sessions[session_id] = FlowState(
        session_id=session_id,
        user_id=payload.user_id,
        query=payload.query,
        status="pending",
    )
    background_tasks.add_task(
        _run_flow,
        session_id=session_id,
        query=payload.query,
        user_id=payload.user_id,
    )
    return QueryResponse(session_id=session_id, status="pending")


@router.get("/query/{session_id}", response_model=QueryResponse)
async def get_result(session_id: str):
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return QueryResponse(
        session_id=state.session_id,
        status=state.status,
        final_response=state.final_response or None,
        evaluation=state.evaluation,
        error=state.error,
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    from research_assistant.tools.tensorlake_tool import TensorlakeParserTool
    tool = TensorlakeParserTool()
    message = tool._run(file_path=file_path, source_name=file.filename)

    if "failed" in message.lower() or "not installed" in message.lower() or "not set" in message.lower():
        return DocumentUploadResponse(
            filename=file.filename,
            status="error",
            message=message,
        )

    return DocumentUploadResponse(
        filename=file.filename,
        status="indexed",
        message=message,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()

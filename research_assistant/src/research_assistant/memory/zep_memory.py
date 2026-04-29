"""
Zep Cloud v3 memory layer.
Uses client.thread for storing messages and client.graph.search for retrieval.
zep_crewai is incompatible with crewai 1.14+ — custom BaseTool wrappers are used instead.
"""
from __future__ import annotations

import os
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_zep_client = None


def get_zep_client():
    global _zep_client
    if _zep_client is None:
        try:
            from zep_cloud.client import Zep
        except ImportError:
            raise ImportError("zep-cloud is not installed. Run: uv add zep-cloud")
        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise ValueError("ZEP_API_KEY environment variable is not set.")
        _zep_client = Zep(api_key=api_key)
    return _zep_client


def _ensure_user_and_thread(user_id: str, session_id: str) -> None:
    """Create user and thread if they don't exist yet (idempotent)."""
    client = get_zep_client()
    try:
        client.user.add(user_id=user_id)
    except Exception:
        pass  # user already exists
    try:
        client.thread.create(thread_id=session_id, user_id=user_id)
    except Exception:
        pass  # thread already exists


def save_to_zep(user_id: str, session_id: str, query: str, response: str) -> None:
    """Save a user query and assistant response to Zep thread memory.

    Zep thread.add_messages caps content at 4096 chars; long responses are
    truncated for the thread and pushed in full to the user's graph.
    """
    try:
        from zep_cloud.types import Message

        client = get_zep_client()
        _ensure_user_and_thread(user_id, session_id)

        MAX = 4000
        truncated_query = query if len(query) <= MAX else query[:MAX] + "...[truncated]"
        truncated_response = (
            response if len(response) <= MAX else response[:MAX] + "...[truncated]"
        )

        client.thread.add_messages(
            thread_id=session_id,
            messages=[
                Message(role="user", content=truncated_query),
                Message(role="assistant", content=truncated_response),
            ],
        )

        if len(response) > MAX:
            try:
                client.graph.add(user_id=user_id, type="text", data=response)
            except Exception as e:
                print(f"[Zep] Warning: failed to save full response to graph: {e}")
    except Exception as e:
        print(f"[Zep] Warning: failed to save memory: {e}")


# ── Custom CrewAI tools wrapping Zep v3 ──────────────────────────────────────

class ZepSearchInput(BaseModel):
    query: str = Field(..., description="Search query to retrieve relevant memories")


class ZepSearchTool(BaseTool):
    """Search the user's Zep knowledge graph for relevant prior context."""

    name: str = "zep_memory_search"
    description: str = (
        "Search the user's conversation history and prior research sessions "
        "stored in Zep memory. Use for retrieving previously gathered context "
        "that may be relevant to the current query."
    )
    args_schema: Type[BaseModel] = ZepSearchInput
    user_id: str = "default_user"

    def _run(self, query: str) -> str:
        try:
            client = get_zep_client()
            results = client.graph.search(
                query=query,
                user_id=self.user_id,
                limit=5,
                scope="episodes",
            )
            episodes = results.episodes or []
            if not episodes:
                return "No relevant memory found for this user."

            output = []
            for ep in episodes:
                score_str = f"{ep.score:.2f}" if ep.score is not None else "N/A"
                output.append(f"[Relevance: {score_str}] {ep.content}")
            return "\n---\n".join(output)
        except Exception as e:
            return f"Memory search failed: {e}"


class ZepAddInput(BaseModel):
    content: str = Field(..., description="Content to save to memory")


class ZepAddTool(BaseTool):
    """Save important findings to the user's Zep memory."""

    name: str = "zep_memory_add"
    description: str = (
        "Save important information to the user's Zep memory for future retrieval. "
        "Use to persist key research findings, summaries, or facts."
    )
    args_schema: Type[BaseModel] = ZepAddInput
    user_id: str = "default_user"
    session_id: str = "default_session"

    def _run(self, content: str) -> str:
        try:
            from zep_cloud.types import Message

            client = get_zep_client()
            _ensure_user_and_thread(self.user_id, self.session_id)
            client.thread.add_messages(
                thread_id=self.session_id,
                messages=[Message(role="assistant", content=content)],
            )
            return f"Saved to memory successfully."
        except Exception as e:
            return f"Failed to save to memory: {e}"


def get_zep_search_tool(user_id: str) -> ZepSearchTool:
    return ZepSearchTool(user_id=user_id)


def get_zep_add_tool(user_id: str, session_id: str = "default_session") -> ZepAddTool:
    return ZepAddTool(user_id=user_id, session_id=session_id)

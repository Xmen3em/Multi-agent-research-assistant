from __future__ import annotations

import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FirecrawlInput(BaseModel):
    query: str = Field(..., description="Web search query")
    limit: int = Field(3, description="Maximum number of web results to return")


class FirecrawlSearchTool(BaseTool):
    name: str = "firecrawl_web_search"
    description: str = (
        "Search the web using Firecrawl to retrieve clean, structured content. "
        "Use for current events, recent news, blog posts, documentation, "
        "and any web-based information sources."
    )
    args_schema: Type[BaseModel] = FirecrawlInput

    def _run(self, query: str, limit: int = 3) -> str:
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "firecrawl-py is not installed. Run: uv add firecrawl-py"

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return "FIRECRAWL_API_KEY environment variable is not set."

        try:
            app = FirecrawlApp(api_key=api_key)
            response = app.search(query, limit=limit)
        except Exception as e:
            return f"Firecrawl search failed: {e}"

        # firecrawl-py v4 returns SearchData(web=[...], news=[...], images=[...])
        # older versions returned {"data": [...]} or an object with .data
        if isinstance(response, dict):
            data = response.get("data") or response.get("web") or []
        elif hasattr(response, "web"):
            data = list(response.web or []) + list(getattr(response, "news", None) or [])
        elif hasattr(response, "data"):
            data = response.data or []
        else:
            data = []

        if not data:
            return "No web results found for this query."

        def _get(obj, key, default=""):
            if isinstance(obj, dict):
                return obj.get(key, default) or default
            return getattr(obj, key, default) or default

        formatted = []
        for item in data:
            url = _get(item, "url")
            title = _get(item, "title")
            content = (
                _get(item, "markdown")
                or _get(item, "content")
                or _get(item, "description")
                or _get(item, "snippet")
            )
            formatted.append(f"Source: {title}\nURL: {url}\nContent:\n{str(content)[:800]}")

        return "\n---\n".join(formatted)

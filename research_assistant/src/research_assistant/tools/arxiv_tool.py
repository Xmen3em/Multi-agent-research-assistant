from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ArxivInput(BaseModel):
    query: str = Field(..., description="Search query for arXiv papers")
    category: Optional[str] = Field(None, description="arXiv category filter, e.g. cs.AI, cs.LG")
    author: Optional[str] = Field(None, description="Filter by author name")
    max_results: int = Field(5, description="Maximum number of papers to return")


class ArxivAPITool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for academic papers and preprints. "
        "Use for finding peer-reviewed research, scientific literature, "
        "and cutting-edge technical papers on any topic."
    )
    args_schema: Type[BaseModel] = ArxivInput

    def _run(
        self,
        query: str,
        category: Optional[str] = None,
        author: Optional[str] = None,
        max_results: int = 5,
    ) -> str:
        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category} AND {search_query}"
        if author:
            search_query = f"au:{author} AND {search_query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            return f"ArXiv API request failed: {e}"

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            return f"Failed to parse ArXiv response: {e}"

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link_el = entry.find("atom:id", ns)
            link = link_el.text.strip() if link_el is not None else ""
            authors = [
                a.findtext("atom:name", default="", namespaces=ns)
                for a in entry.findall("atom:author", ns)
            ]
            results.append(
                f"Title: {title}\n"
                f"Authors: {', '.join(authors)}\n"
                f"URL: {link}\n"
                f"Abstract: {summary[:500]}...\n"
            )

        if not results:
            return "No papers found for this query."
        return "\n---\n".join(results)

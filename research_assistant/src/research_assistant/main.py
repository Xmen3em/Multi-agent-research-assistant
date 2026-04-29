#!/usr/bin/env python
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Alias for kickoff — kept for backwards compatibility."""
    kickoff()


def kickoff():
    """Primary entry point. Called by `crewai run` (type = flow in pyproject.toml)."""
    from research_assistant.flow import kickoff as flow_kickoff
    flow_kickoff()


def start_api():
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    from fastapi import FastAPI
    from research_assistant.api.router import router

    app = FastAPI(
        title="Multi-Agent Research Assistant API",
        description="Context engineering pipeline: RAG + Memory + Web + ArXiv",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.include_router(router, prefix="/api")

    host = "0.0.0.0"
    port = 8000
    print(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)


def run_with_trigger():
    """Handle trigger payload from CLI — maps to flow kickoff with JSON input."""
    import json
    from research_assistant.flow import ContextEngineeringFlow

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Pass JSON as argument.")

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload.")

    flow = ContextEngineeringFlow()
    flow.kickoff(inputs=payload)
    return flow.state.final_response


if __name__ == "__main__":
    kickoff()

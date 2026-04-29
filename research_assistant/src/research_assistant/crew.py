from __future__ import annotations

import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from research_assistant.schemas import ContextEvaluationOutput
from research_assistant.tools.arxiv_tool import ArxivAPITool
from research_assistant.tools.firecrawl_tool import FirecrawlSearchTool
from research_assistant.tools.milvus_rag_tool import MilvusRAGTool


def _build_llm() -> LLM:
    model = os.environ.get("MODEL", "openai/moonshotai/Kimi-K2-Instruct-0905")
    return LLM(
        model=model,
        base_url=os.environ.get("OPENAI_API_BASE", "https://router.huggingface.co/v1"),
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN"),
        max_tokens=4096,
        temperature=0.3,
        max_retries=3,
        context_window_size=131072,
        timeout=60,
    )


_SHARED_LLM = _build_llm()

AGENT_TIMEOUT = 180


@CrewBase
class ContextCrew:
    """Runs 4 context-gathering agents independently for fault isolation."""

    tasks_config = "config/context_tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],  # type: ignore[index]
            tools=[MilvusRAGTool()],
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=6,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @agent
    def memory_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["memory_agent"],  # type: ignore[index]
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=6,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @agent
    def web_search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["web_search_agent"],  # type: ignore[index]
            tools=[FirecrawlSearchTool()],
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=6,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @agent
    def arxiv_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["arxiv_agent"],  # type: ignore[index]
            tools=[ArxivAPITool()],
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=6,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @task
    def rag_task(self) -> Task:
        return Task(config=self.tasks_config["rag_task"])  # type: ignore[index]

    @task
    def memory_task(self) -> Task:
        return Task(config=self.tasks_config["memory_task"])  # type: ignore[index]

    @task
    def web_search_task(self) -> Task:
        return Task(config=self.tasks_config["web_search_task"])  # type: ignore[index]

    @task
    def arxiv_task(self) -> Task:
        return Task(config=self.tasks_config["arxiv_task"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


@CrewBase
class EvaluationCrew:
    """Single-agent crew that evaluates and filters gathered context."""

    tasks_config = "config/evaluation_tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def context_evaluator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["context_evaluator_agent"],  # type: ignore[index]
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=3,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluation_task"],  # type: ignore[index]
            output_pydantic=ContextEvaluationOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


@CrewBase
class SynthesisCrew:
    """Single-agent crew that synthesizes the final research response."""

    tasks_config = "config/synthesis_tasks.yaml"

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["synthesizer_agent"],  # type: ignore[index]
            verbose=True,
            respect_context_window=True,
            llm=_SHARED_LLM,
            max_iter=3,
            max_retry_limit=2,
            max_execution_time=AGENT_TIMEOUT,
        )

    @task
    def synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

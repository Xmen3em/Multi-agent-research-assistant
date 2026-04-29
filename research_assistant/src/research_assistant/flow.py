from __future__ import annotations

import uuid

from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, start

from research_assistant.crew import ContextCrew, EvaluationCrew, SynthesisCrew
from research_assistant.memory.zep_memory import (
    get_zep_add_tool,
    get_zep_search_tool,
    save_to_zep,
)
from research_assistant.schemas import ContextEvaluationOutput, FlowState

MAX_CONTEXT_CHARS = 3000


class ContextEngineeringFlow(Flow[FlowState]):
    """
    4-step context engineering pipeline:
      1. process_query           — save query to Zep, assign session_id
      2. gather_context          — run 4 agents independently for fault isolation
      3. evaluate_context        — filter & score context via EvaluationCrew
      4. synthesize_final_response — generate answer, save to Zep
    """

    @start()
    def process_query(self):
        if not self.state.session_id:
            self.state.session_id = str(uuid.uuid4())
        self.state.status = "running"
        save_to_zep(
            user_id=self.state.user_id,
            session_id=self.state.session_id,
            query=self.state.query,
            response="[pending]",
        )

    @listen(process_query)
    def gather_context(self):
        """Run each context agent as a separate crew so one failure does not
        prevent the other agents from producing results."""
        query = self.state.query

        agent_task_pairs = [
            ("rag_agent", "rag_task", "rag_context"),
            ("memory_agent", "memory_task", "memory_context"),
            ("web_search_agent", "web_search_task", "web_context"),
            ("arxiv_agent", "arxiv_task", "arxiv_context"),
        ]

        for agent_method_name, task_method_name, state_attr in agent_task_pairs:
            try:
                ctx = ContextCrew()
                agent = getattr(ctx, agent_method_name)()
                task = getattr(ctx, task_method_name)()
                task.agent = agent

                if "memory" in agent_method_name:
                    try:
                        search_tool = get_zep_search_tool(self.state.user_id)
                        add_tool = get_zep_add_tool(
                            self.state.user_id, self.state.session_id
                        )
                        agent.tools = [search_tool, add_tool]
                    except Exception as e:
                        print(f"[Flow] Warning: Zep tools unavailable: {e}")

                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                result = crew.kickoff(inputs={"query": query})
                if result.tasks_output:
                    setattr(self.state, state_attr, result.tasks_output[0].raw)
            except Exception as e:
                print(f"[Flow] Warning: {agent_method_name} failed: {e}")
                setattr(self.state, state_attr, "")

    @listen(gather_context)
    def evaluate_context_relevance(self):
        """Filter and score the gathered context with the evaluation crew."""
        def _truncate(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
            if len(text) <= limit:
                return text
            return text[:limit] + "\n...[truncated]"

        try:
            result = EvaluationCrew().crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "rag_context": _truncate(self.state.rag_context),
                    "memory_context": _truncate(self.state.memory_context),
                    "web_context": _truncate(self.state.web_context),
                    "arxiv_context": _truncate(self.state.arxiv_context),
                }
            )

            if result.pydantic:
                self.state.evaluation = result.pydantic
            else:
                all_context = "\n\n".join(
                    filter(
                        None,
                        [
                            self.state.rag_context,
                            self.state.memory_context,
                            self.state.web_context,
                            self.state.arxiv_context,
                        ],
                    )
                )
                self.state.evaluation = ContextEvaluationOutput(
                    relevant_sources=["rag", "memory", "web", "arxiv"],
                    filtered_context=all_context,
                    relevance_scores={
                        "rag": 0.7,
                        "memory": 0.7,
                        "web": 0.7,
                        "arxiv": 0.7,
                    },
                )
        except Exception as e:
            print(f"[Flow] Warning: Evaluation failed, using fallback: {e}")
            all_context = "\n\n".join(
                filter(
                    None,
                    [
                        self.state.rag_context,
                        self.state.memory_context,
                        self.state.web_context,
                        self.state.arxiv_context,
                    ],
                )
            )
            self.state.evaluation = ContextEvaluationOutput(
                relevant_sources=["rag", "memory", "web", "arxiv"],
                filtered_context=all_context or "No context available.",
                relevance_scores={
                    "rag": 0.5,
                    "memory": 0.5,
                    "web": 0.5,
                    "arxiv": 0.5,
                },
            )

    @listen(evaluate_context_relevance)
    def synthesize_final_response(self):
        """Synthesize the final answer and save it to Zep memory."""
        evaluation = self.state.evaluation

        try:
            result = SynthesisCrew().crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "filtered_context": evaluation.filtered_context,
                    "relevant_sources": ", ".join(evaluation.relevant_sources),
                }
            )
            self.state.final_response = result.raw
        except Exception as e:
            print(f"[Flow] Warning: Synthesis failed, using raw context: {e}")
            self.state.final_response = (
                f"I was unable to generate a fully synthesized response due to a"
                f" processing error: {e}\n\n"
                f"Here is the raw context gathered from available sources:\n\n"
                f"--- Document Context ---\n{self.state.rag_context}\n\n"
                f"--- Memory Context ---\n{self.state.memory_context}\n\n"
                f"--- Web Context ---\n{self.state.web_context}\n\n"
                f"--- ArXiv Context ---\n{self.state.arxiv_context}"
            )

        self.state.status = "complete"

        save_to_zep(
            user_id=self.state.user_id,
            session_id=self.state.session_id,
            query=self.state.query,
            response=self.state.final_response,
        )


def kickoff():
    """Entry point called by `crewai run` (type = flow in pyproject.toml)."""
    flow = ContextEngineeringFlow()
    flow.kickoff(
        inputs={
            "query": "What are the latest advances in context engineering for LLMs?",
            "user_id": "default_user",
        }
    )
    print("\n" + "=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)
    print(flow.state.final_response)

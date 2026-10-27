from typing import Annotated, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from harness.experiments.initialised_orchs import LLM_MODEL, llm_orchestrator
from harness.experiments.initialised_subagents import (
    llm_browser_subagent,
    llm_code_interpreter_subagent,
    llm_doc_reader_subagent,
    llm_media_reader_subagent,
    llm_web_search_subagent,
)
from harness.orchestrators.base import BaseOrchestrator

llm_finalizer = ChatOllama(model=LLM_MODEL)

GAIA_FINALIZER_PROMPT = """You are the finalizer agent in a GAIA benchmark evaluation.
Given the task and the intermediate steps/results from the subagents, synthesize the findings and output ONLY the final answer to the user's question.
- Format the response exactly as required by GAIA: direct, minimal, and without any explanations or conversational filler.
- If the question asks for a number, output only the number.
- If the question asks for a specific format (e.g., comma-separated list, unit, exact wording), follow it strictly."""


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    state_step: int = 0
    next_subagent: Optional[str] = None
    subagent_prompt: Optional[str] = None


class MultiAgentSystem():
    def __init__(self, orchestrator: BaseOrchestrator, log_path: str):
        self.orchestrator = orchestrator
        self.subagents = orchestrator.get_subagents()
        self.builder = StateGraph(State)
        self.log_path = log_path

    def build_graph(self):
        self._create_orchestrator_node()
        self._create_all_subagent_nodes()
        self._create_finalizer_node()
        self._add_edges()
        self._set_log_path()
        return self.builder.compile()
    
    def _create_orchestrator_node(self):
        def orchestrator_node(state: State) -> dict:
            decision = self.orchestrator(state)
            return {
                "next_subagent": str(decision).strip(),
                "state_step": state.state_step + 1,
            }

        self.builder.add_node("orchestrator", orchestrator_node)
        return orchestrator_node

    def _create_subagent_node(self, name: str, subagent):
        def subagent_node(state: State) -> dict:
            res = subagent(list(state.messages))
            return {"messages": [AIMessage(content=res, name=name)]}
        return subagent_node

    def _create_all_subagent_nodes(self):
        for subagent in self.subagents:
            name = subagent.subagent_name
            self.builder.add_node(name, self._create_subagent_node(name, subagent))

    def _create_finalizer_node(self):
        def finalizer_node(state: State) -> dict:
            past_messages = []
            for m in state.messages:
                role = m.name
                past_messages.append(f"[{role}]: {m.content}")
            past_conversation = "\n".join(past_messages)

            res = llm_finalizer.invoke([
                SystemMessage(content=GAIA_FINALIZER_PROMPT),
                HumanMessage(
                    content=f"Here is the task and all intermediate subagent findings:\n{past_conversation}\n\nProvide the final answer:"
                ),
            ])
            return {"messages": [AIMessage(content=str(res.content).strip(), name="finalizer")]}

        self.builder.add_node("finalizer", finalizer_node)
        return finalizer_node

    def _add_edges(self):
        self.builder.add_edge(START, "orchestrator")
        for subagent in self.subagents:
            name = subagent.subagent_name
            self.builder.add_edge(name, "orchestrator")
        self.builder.add_edge("finalizer", END)
        self.builder.add_conditional_edges(
            "orchestrator",
            self._routing_condition,
            {
                "web_search": "web_search",
                "browser": "browser",
                "doc_reader": "doc_reader",
                "media_reader": "media_reader",
                "code_interpreter": "code_interpreter",
                "finalizer": "finalizer",
            },
        )



    @staticmethod
    def _routing_condition(state: State) -> str:
        if state.next_subagent == "web_search":
            return "web_search"
        if state.next_subagent == "browser":
            return "browser"
        if state.next_subagent == "doc_reader":
            return "doc_reader"
        if state.next_subagent == "media_reader":
            return "media_reader"
        if state.next_subagent == "code_interpreter":
            return "code_interpreter"

        return "finalizer"
    
    def _set_log_path(self):
        for subagent in self.subagents:
            subagent.set_log_path(self.log_path)
        self.orchestrator.set_log_path(self.log_path)

multi_agent_system = MultiAgentSystem(llm_orchestrator, "testing.jsonl")
agent = multi_agent_system.build_graph()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    query = "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"
    print(f"Query: {query}")
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
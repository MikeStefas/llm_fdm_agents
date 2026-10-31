import json
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
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
from harness.utils import FinalizerNode


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    state_step: int = 0
    next_subagent: Optional[str] = None
    subagent_prompt: Optional[str] = None


class MultiAgentSystem():
    def __init__(
        self,
        orchestrator: BaseOrchestrator,
        log_path: str,
        finalizer_node: Optional[FinalizerNode] = None,
    ):
        self.orchestrator = orchestrator
        self.subagents = orchestrator.get_subagents()
        self.finalizer_node = finalizer_node if finalizer_node is not None else FinalizerNode(log_path=log_path)
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
            if isinstance(decision, dict):
                return {
                    "next_subagent": str(decision["next_subagent"]).strip(),
                    "subagent_prompt": str(decision["subagent_prompt"]),
                    "state_step": state.state_step + 1,
                }
            return {
                "next_subagent": str(decision).strip(),
                "state_step": state.state_step + 1,
            }

        self.builder.add_node("orchestrator", orchestrator_node)
        return orchestrator_node

    def _create_subagent_node(self, name: str, subagent):
        def subagent_node(state: State) -> dict:
            messages = list(state.messages)
            if state.subagent_prompt:
                messages.append(HumanMessage(content=state.subagent_prompt))
            res = subagent(messages, state_step=state.state_step)
            return {"messages": [AIMessage(content=res, name=name)], "subagent_prompt": None}
        return subagent_node

    def _create_all_subagent_nodes(self):
        for subagent in self.subagents:
            name = subagent.subagent_name
            self.builder.add_node(name, self._create_subagent_node(name, subagent))

    def _create_finalizer_node(self):
        def finalizer_node(state: State) -> dict:
            output_text = self.finalizer_node(state.messages, state_step=state.state_step)
            return {"messages": [AIMessage(content=output_text, name="finalizer")]}

        self.builder.add_node("finalizer", finalizer_node)
        return finalizer_node

    def log_finalizer_step(
        self,
        state_step: int,
        tokens_in: int,
        tokens_out: int,
        duration: float,
        final_response: str | None = None,
        error: str | None = None,
    ) -> None:
        self.finalizer_node.log_step(
            state_step=state_step,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration=duration,
            response=final_response,
            error=error,
        )

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
        self.finalizer_node.set_log_path(self.log_path)

from typing import Annotated, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from harness.mas_builder.finalizer import FinalizerNode
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    state_step: int = 1
    next_subagent: Optional[str] = None
    subagent_prompt: Optional[str] = None
    task_id: Optional[str] = None


class MultiAgentSystem:
    def __init__(
        self,
        orchestrator: BaseOrchestrator,
        log_path: str,
        subagents: Optional[list[BaseReactAgent]] = None,
        max_steps: int = 8,
    ):
        self.orchestrator = orchestrator
        self.subagents: list[BaseReactAgent] = []
        if subagents is not None:
            self.set_subagents(subagents)
        elif orchestrator.has_subagents():
            self.subagents = orchestrator.get_subagents()

        final_llm = orchestrator.llm_client
        if final_llm is None and self.subagents:
            final_llm = self.subagents[0].llm_client
        if final_llm is None:
            from config import llm_client
            final_llm = llm_client

        self.finalizer = FinalizerNode(log_path=log_path, llm_client=final_llm)
        self.max_steps = max_steps

        self.builder = StateGraph(State)
        self.log_path = log_path
        self.set_log_path()

    def set_subagents(self, subagents: list[BaseReactAgent]) -> None:
        """Sets the subagents on the MAS and passes them to the orchestrator before building the graph."""
        self.subagents = subagents
        self.orchestrator.set_subagents(subagents)

    def build_graph(self):
        if not self.subagents:
            if self.orchestrator.has_subagents():
                self.subagents = self.orchestrator.get_subagents()
            else:
                raise ValueError(
                    "MultiAgentSystem has no subagents configured. "
                    "Provide subagents in __init__ or call mas.set_subagents(subagents) before build_graph()."
                )
        self.orchestrator.ensure_has_subagents()

        self._create_orchestrator_node()
        self._create_all_subagent_nodes()
        self._create_finalizer_node()
        self._add_edges()
        self.set_log_path()
        return self.builder.compile()

    def _create_orchestrator_node(self):
        def orchestrator_node(state: State) -> dict:
            decision = self.orchestrator(state)
            if isinstance(decision, dict):
                return {
                    "next_subagent": str(decision["next_subagent"]).strip(),
                    "subagent_prompt": str(decision["subagent_prompt"]),
                    "state_step": state.state_step + 1,
                    "task_id": state.task_id,
                }
            return {
                "next_subagent": str(decision).strip(),
                "state_step": state.state_step + 1,
                "task_id": state.task_id,
            }

        self.builder.add_node("orchestrator", orchestrator_node)
        return orchestrator_node

    def _create_subagent_node(self, name: str, subagent):
        def subagent_node(state: State) -> dict:
            messages = list(state.messages)
            if state.subagent_prompt:
                messages.append(HumanMessage(content=state.subagent_prompt))
            res = subagent(messages, state_step=state.state_step, task_id=state.task_id)
            return {
                "messages": [AIMessage(content=res, name=name)],
                "subagent_prompt": None,
                "task_id": state.task_id,
            }

        return subagent_node

    def _create_all_subagent_nodes(self):
        for subagent in self.subagents:
            name = subagent.subagent_name
            self.builder.add_node(name, self._create_subagent_node(name, subagent))

    def _create_finalizer_node(self):
        def finalizer_node(state: State) -> dict:
            output_text = self.finalizer(
                state.messages,
                state_step=state.state_step,
                task_id=state.task_id,
            )
            return {
                "messages": [AIMessage(content=output_text, name="finalizer")],
                "task_id": state.task_id,
            }

        self.builder.add_node("finalizer", finalizer_node)
        return finalizer_node

    def _add_edges(self):
        self.builder.add_edge(START, "orchestrator")
        for subagent in self.subagents:
            name = subagent.subagent_name
            self.builder.add_edge(name, "orchestrator")
        self.builder.add_edge("finalizer", END)

        routing_destinations = {}
        for subagent in self.subagents:
            name = subagent.subagent_name
            routing_destinations[name] = name
        routing_destinations["finalizer"] = "finalizer"

        self.builder.add_conditional_edges(
            "orchestrator",
            self._routing_condition,
            routing_destinations,
        )

    def _routing_condition(self, state: State) -> str:
        if state.state_step >= self.max_steps:
            return "finalizer"

        for subagent in self.subagents:
            if state.next_subagent == subagent.subagent_name:
                return subagent.subagent_name
        return "finalizer"

    def set_log_path(self, new_log_path: Optional[str] = None):
        if new_log_path:
            self.log_path = new_log_path
        for subagent in self.subagents:
            subagent.set_log_path(self.log_path)
        self.orchestrator.set_log_path(self.log_path)
        self.finalizer.set_log_path(self.log_path)

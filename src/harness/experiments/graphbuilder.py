import json
import time
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
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
from harness.llm import get_chat_model
from harness.logger import append_to_csv
from harness.orchestrators.base import BaseOrchestrator

llm_finalizer = get_chat_model()


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
            t_start = time.perf_counter()
            past_messages = []
            for m in state.messages:
                role = m.name if m.name else m.type
                past_messages.append(f"[{role}]: {m.content}")
            past_conversation = "\n".join(past_messages)

            try:
                res = llm_finalizer.invoke([
                    SystemMessage(content=GAIA_FINALIZER_PROMPT),
                    HumanMessage(
                        content=f"Here is the task and all intermediate subagent findings:\n{past_conversation}\n\nProvide the final answer:"
                    ),
                ])
                output_text = str(res.content).strip()
                duration = time.perf_counter() - t_start
                usage = res.usage_metadata
                tokens_in = usage["input_tokens"]
                tokens_out = usage["output_tokens"]
                self.log_finalizer_step(state.state_step, tokens_in, tokens_out, duration, output_text, None)
                return {"messages": [AIMessage(content=output_text, name="finalizer")]}
            except Exception as e:
                duration = time.perf_counter() - t_start
                error = str(e)
                self.log_finalizer_step(state.state_step, 0, 0, duration, None, error)
                raise

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
        log = {
            "node": "finalizer",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": False,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response": final_response,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

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

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

#setup log paths for said experiment
llm_orchestrator.set_log_path("orchestrator.jsonl")
llm_web_search_subagent.set_log_path("subagents.jsonl")
llm_browser_subagent.set_log_path("subagents.jsonl")
llm_doc_reader_subagent.set_log_path("subagents.jsonl")
llm_media_reader_subagent.set_log_path("subagents.jsonl")
llm_code_interpreter_subagent.set_log_path("subagents.jsonl")

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



def orchestrator_node(state: State) -> dict:
    decision = llm_orchestrator(state)
    return {
        "next_subagent": str(decision).strip(),
        "state_step": state.state_step + 1,
    }


def web_search_node(state: State) -> dict:
    res = llm_web_search_subagent(list(state.messages))
    return {"messages": [AIMessage(content=res, name="web_search")]}


def browser_node(state: State) -> dict:
    res = llm_browser_subagent(list(state.messages))
    return {"messages": [AIMessage(content=res, name="browser")]}


def doc_reader_node(state: State) -> dict:
    res = llm_doc_reader_subagent(list(state.messages))
    return {"messages": [AIMessage(content=res, name="doc_reader")]}


def media_reader_node(state: State) -> dict:
    res = llm_media_reader_subagent(list(state.messages))
    return {"messages": [AIMessage(content=res, name="media_reader")]}


def code_interpreter_node(state: State) -> dict:
    res = llm_code_interpreter_subagent(list(state.messages))
    return {"messages": [AIMessage(content=res, name="code_interpreter")]}


def finalizer_node(state: State) -> dict:
    history_lines = []
    for m in state.messages:
        role = getattr(m, "name", None) or m.type
        history_lines.append(f"[{role}]: {m.content}")
    formatted_history = "\n".join(history_lines)

    res = llm_finalizer.invoke([
        SystemMessage(content=GAIA_FINALIZER_PROMPT),
        HumanMessage(content=f"Here is the task and all intermediate subagent findings:\n{formatted_history}\n\nProvide the final answer:"),
    ])
    return {"messages": [AIMessage(content=str(res.content).strip(), name="finalizer")]}


builder = StateGraph(State)

builder.add_node("orchestrator", orchestrator_node)
builder.add_node("web_search", web_search_node)
builder.add_node("browser", browser_node)
builder.add_node("doc_reader", doc_reader_node)
builder.add_node("media_reader", media_reader_node)
builder.add_node("code_interpreter", code_interpreter_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "orchestrator")
builder.add_edge("web_search", "orchestrator")
builder.add_edge("browser", "orchestrator")
builder.add_edge("doc_reader", "orchestrator")
builder.add_edge("media_reader", "orchestrator")
builder.add_edge("code_interpreter", "orchestrator")
builder.add_edge("finalizer", END)  

def condition(state: State) -> str:
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


builder.add_conditional_edges(
    "orchestrator",
    condition,
    {
        "web_search": "web_search",
        "browser": "browser",
        "doc_reader": "doc_reader",
        "media_reader": "media_reader",
        "code_interpreter": "code_interpreter",
        "finalizer": "finalizer",
    },
)


agent = builder.compile()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    query = "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"
    print(f"Query: {query}")
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from harness.mas_builder import MultiAgentSystem
from harness.orchestrators import LLMOrchestrator
from harness.react_agents import LLMReactAgent
from harness.tools import (
    CODE_TOOLS,
    DOC_READER_TOOLS,
    MEDIA_READER_TOOLS,
    WEB_SEARCH_TOOLS,
)
from harness.utils import (
    CODE_INTERPRETER_DESCRIPTION,
    DOC_READER_DESCRIPTION,
    MEDIA_READER_DESCRIPTION,
    WEB_SEARCH_DESCRIPTION,
)

from config import llm_client

web_search_subagent = LLMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    llm_client=llm_client,
)

doc_reader_subagent = LLMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    llm_client=llm_client,
)

media_reader_subagent = LLMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    llm_client=llm_client,
)

code_interpreter_subagent = LLMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    llm_client=llm_client,
)

subagents = [
    web_search_subagent,
    doc_reader_subagent,
    media_reader_subagent,
    code_interpreter_subagent,
]

llm_orchestrator = LLMOrchestrator(
    subagents=subagents,
    llm_client=llm_client,
)

mas = MultiAgentSystem(
    orchestrator=llm_orchestrator,
    log_path="results/llm_1/mas_log.csv",
)
graph = mas.build_graph()


import json

with open("tasks_l2_20.json", "r") as f:
    tasks = json.load(f)

if __name__ == "__main__":
    for i in tasks:
        task_id = i["task_id"]
        mas.set_log_path(f"results/llm_1/mas_log_{task_id}.csv")
        question = i["question"]
        expected = i["expected_answer"]

        print(f"Running Task: {task_id}")
        print(f"Expected:     {expected}\n")

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "task_id": task_id,
        }

        try:
            res = graph.invoke(initial_state)
            final_msg = res["messages"][-1].content
            print(f"Final Result: {final_msg}")
        except Exception as e:
            print(f"Error task {task_id}: {e}")

import json
from langchain_core.messages import HumanMessage

from harness.mas_builder.mas_instances import (
    mas_1_fdm_orch_llm_sub_no_think,
    mas_2_llm_orch_no_think_llm_sub_no_think,
    mas_3_llm_orch_think_llm_sub_no_think,
    mas_4_llm_orch_no_think_fdm_sub_think,
    mas_5_llm_orch_no_think_fdm_sub_no_think,
    mas_6_llm_orch_no_think_llm_sub_think,
)

# 1) FDM orch (no think) + LLM subagent (no think)
conf_1 = mas_1_fdm_orch_llm_sub_no_think

# 2) LLM orch (no think) + LLM subagent (no think)
conf_2 = mas_2_llm_orch_no_think_llm_sub_no_think

# 3) LLM orch (think) + LLM subagent (no think)
conf_3 = mas_3_llm_orch_think_llm_sub_no_think

# 4) LLM orch (no think) + FDM subagent (think on LLM for args)
conf_4 = mas_4_llm_orch_no_think_fdm_sub_think

# 5) LLM orch (no think) + FDM subagent (no think)
conf_5 = mas_5_llm_orch_no_think_fdm_sub_no_think

# 6) LLM orch (no think) + LLM subagent (think)
conf_6 = mas_6_llm_orch_no_think_llm_sub_think

configs = {
    1: conf_1,
    2: conf_2,
    3: conf_3,
    4: conf_4,
    5: conf_5,
    6: conf_6,
}

tasks_path = "tasks/tasks_benchmark_60_no_solutions.json"

for conf_id, mas in configs.items():
    print("#"* 80)
    print("Configuration: ", conf_id)
    print("#"* 80)

    graph = mas.build_graph()

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    for i, t in enumerate(tasks):
        task_id = t["task_id"]
        question = t["question"]

        mas.set_log_path(f"results/mas_{conf_id}/mas_log_{task_id}.csv")

        print(f"[{i}/{len(tasks)}]")
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "task_id": task_id,
        }

        try:
            res = graph.invoke(initial_state)
            final_msg = res["messages"][-1].content
            print(f"Final Result: {final_msg}\n" + "-" * 50)
        except Exception as e:
            print(f"Error on task {task_id}: {e}\n" + "-" * 50)

import csv
import json
import os
from langchain_core.messages import HumanMessage

from harness.mas_builder.mas_instances import (
    mas_1_fdm_orch_llm_sub_no_think,
    mas_2_llm_orch_no_think_llm_sub_no_think,
    mas_3_llm_orch_think_llm_sub_no_think,
    mas_4_llm_orch_no_think_fdm_sub_think,
    mas_5_llm_orch_no_think_fdm_sub_no_think,
    mas_6_llm_orch_no_think_llm_sub_think,
    mas_7_fdm_orch_fdm_sub_no_think,
    mas_8_llm_orch_think_llm_sub_think,
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

# 7) Full FDM: FDM orch (no think) + FDM subagent (no think)
conf_7 = mas_7_fdm_orch_fdm_sub_no_think

# 8) Full LLM: LLM orch (think) + LLM subagent (think)
conf_8 = mas_8_llm_orch_think_llm_sub_think

configs = {
    1: conf_1,
    2: conf_2,
    3: conf_3,
    4: conf_4,
    5: conf_5,
    6: conf_6,
    7: conf_7,
    8: conf_8,
}

tasks_path = "stratified_dataset/60_task_subset_test.json"

for conf_id, mas in configs.items():
    if conf_id in [1, 2, 3, 4, 5, 6]:
        continue # configs 1-6 completed in prior runs

    print("#" * 80)
    print("Configuration: ", conf_id)
    print("#" * 80)

    graph = mas.build_graph()

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    for i, t in enumerate(tasks):
        task_id = str(t.get("task_id") or "")
        question = str(t.get("Question") or t.get("question") or "")
        file_path = str(t.get("file_path") or t.get("file_name") or t.get("file") or "").strip()

        if file_path:
            question += f"\n\nProvided file is located at: {file_path}"

        mas.set_log_path(f"results/mas_{conf_id}/mas_log_{task_id}.csv")

        print(f"[{i + 1}/{len(tasks)}] Task ID: {task_id}")
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "task_id": task_id,
        }

        try:
            res = graph.invoke(initial_state)
            final_msg = res["messages"][-1].content

            # Save clean row to results.csv
            results_file = "results.csv"
            file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
            with open(results_file, "a", newline="", encoding="utf-8") as f_res:
                writer = csv.writer(f_res)
                if not file_exists:
                    writer.writerow(["task_id", "question", "config_id", "final_answer"])
                writer.writerow([task_id, question, conf_id, final_msg])

            print(f"Final Result: {final_msg}\n" + "-" * 50)
        except Exception as e:
            print(f"Error on task {task_id}: {e}\n" + "-" * 50)

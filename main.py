import csv
import json
import os
from langchain_core.messages import HumanMessage

from harness.mas_builder.mas_instances import (
    mas_1_llm_orch_think_llm_sub_think,
    mas_2_fdm_orch_llm_sub_think,
    mas_3_llm_orch_no_think_llm_sub_think,
    mas_4_fdm_orch_fdm_sub,
    mas_5_llm_orch_think_fdm_sub,
)
from harness.utils import reset_shared_directory

# 1) Full LLM: LLM Orch (Think ON) + LLM Sub (Think ON)
conf_1 = mas_1_llm_orch_think_llm_sub_think

# 2) FDM Orch + LLM Sub (Think ON)
conf_2 = mas_2_fdm_orch_llm_sub_think

# 3) LLM Orch (Think OFF) + LLM Sub (Think ON)
conf_3 = mas_3_llm_orch_no_think_llm_sub_think

# 4) Full FDM: FDM Orch + FDM Sub
conf_4 = mas_4_fdm_orch_fdm_sub

# 5) LLM Orch (Think ON) + FDM Sub (Head-to-head vs Conf 1 to compare Subagents)
conf_5 = mas_5_llm_orch_think_fdm_sub

configs = {
    # 1: conf_1,  # LLM Orch (Think) + LLM Sub
    # 2: conf_2,  # FDM Orch + LLM Sub
    # 3: conf_3,  # LLM Orch (No-Think) + LLM Sub
    # 4: conf_4,  # Full FDM: FDM Orch + FDM Sub
    5: conf_5,    # LLM Orch (Think) + FDM Sub
}

# Use 5 reasoning tasks for quick comparison, or switch to 60_task_subset_val.json for full benchmark
tasks_path = "stratified_dataset/5_reasoning_tasks.json"
# tasks_path = "stratified_dataset/60_task_subset_val.json"

if __name__ == "__main__":
    for conf_id, mas in configs.items():
        print("#" * 80)
        print("Configuration: ", conf_id)
        print("#" * 80)

        graph = mas.build_graph()

        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        completed_task_ids = set()
        results_file = "results.csv"
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f_res:
                reader = csv.reader(f_res)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3 and str(row[2]) == str(conf_id):
                        completed_task_ids.add(row[0])

        for i, t in enumerate(tasks):
            task_id = str(t.get("task_id") or "")
            if task_id in completed_task_ids:
                print(f"Skipping task [{i + 1}/{len(tasks)}] {task_id} (already completed for conf {conf_id})")
                continue
            question = str(t.get("Question") or t.get("question") or "")
            file_path = str(t.get("file_path") or t.get("file_name") or t.get("file") or "").strip()

            if file_path:
                if not os.path.exists(file_path) and os.path.exists(os.path.join("data/gaia", file_path)):
                    file_path = os.path.join("data/gaia", file_path)
                question += f"\n\nProvided file is located at: {file_path}"

            os.makedirs(f"results/mas_{conf_id}", exist_ok=True)
            mas.set_log_path(f"results/mas_{conf_id}/mas_log_{task_id}.csv")

            print(f"[{i + 1}/{len(tasks)}] Task ID: {task_id}")
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "task_id": task_id,
            }

            try:
                res = graph.invoke(initial_state)
                final_msg = res["messages"][-1].content

                file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
                with open(results_file, "a", newline="", encoding="utf-8") as f_res:
                    writer = csv.writer(f_res)
                    if not file_exists:
                        writer.writerow(["task_id", "question", "config_id", "final_answer"])
                    writer.writerow([task_id, question, conf_id, final_msg])

                print(f"Final Result: {final_msg}\n" + "-" * 50)
            except Exception as e:
                print(f"Error on task {task_id}: {e}\n" + "-" * 50)
            finally:
                reset_shared_directory()
                os.system("rm -rf data/gaia/*/")

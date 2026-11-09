import json
import os
import pandas as pd

# Load dataset
splits = {"validation": "2023/validation/metadata.parquet"}
df = pd.read_parquet("hf://datasets/gaia-benchmark/GAIA/" + splits["validation"])

# Filter Level 1 and Level 2
df_1 = df[df.Level == "1"]
df_2 = df[df.Level == "2"]

# Stratified sample: 30 from Level 1, 30 from Level 2
subset_1 = df_1.sample(n=30, random_state=42)
subset_2 = df_2.sample(n=30, random_state=42)
subset_60 = pd.concat([subset_1, subset_2], ignore_index=True)


def format_gaia_records(df_subset, include_solution=True):
    records = []
    for _, row in df_subset.iterrows():
        raw_file = str(row.get("file_name") or "").strip()
        file_path = f"data/gaia/{raw_file}" if raw_file else None

        question = str(row["Question"])
        if file_path:
            question += f" (The attached file is located at: {file_path})"

        item = {
            "task_id": str(row["task_id"]),
            "level": int(row["Level"]),
            "file": file_path,
            "question": question,
        }
        if include_solution:
            item["expected_answer"] = str(row["Final answer"])

        records.append(item)
    return records


# Ensure output directories exist
os.makedirs("tasks", exist_ok=True)
os.makedirs("stratified_dataset", exist_ok=True)

# 1. 60-Task benchmark files (for main.py)
with open("tasks/tasks_benchmark_60.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_60, include_solution=True), f, indent=2)

with open("tasks/tasks_benchmark_60_no_solutions.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_60, include_solution=False), f, indent=2)

# 2. Individual level files
with open("stratified_dataset/level_1_subset_w_solutions.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_1, include_solution=True), f, indent=2)

with open("stratified_dataset/level_2_subset_w_solutions.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_2, include_solution=True), f, indent=2)

with open("stratified_dataset/level_1_subset_no_solutions.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_1, include_solution=False), f, indent=2)

with open("stratified_dataset/level_2_subset_no_solutions.json", "w", encoding="utf-8") as f:
    json.dump(format_gaia_records(subset_2, include_solution=False), f, indent=2)

print("Successfully generated all formatted JSON files with indent=2!")
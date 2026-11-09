import json
import os
import pandas as pd

# Load dataset
splits = {"validation": "2023/validation/metadata.parquet"}
df = pd.read_parquet("hf://datasets/gaia-benchmark/GAIA/" + splits["validation"]).drop(columns=['file_path'])
# Filter Level 1 and Level 2
df_1 = df[df.Level == "1"]
df_2 = df[df.Level == "2"]

subset_1 = df_1.sample(n=30, random_state=42)
subset_2 = df_2.sample(n=30, random_state=42)
subset_60_val = pd.concat([subset_1, subset_2])

subset_60_val.to_json("60_task_subset_val.json", orient="records", indent=4)

subset_60_test = subset_60_val.copy()
subset_60_test['file_path'] = subset_60_test.apply(lambda x: "data/gaia/" + x["file_name"] if x["file_name"] else "", axis=1)

subset_60_test = subset_60_test.drop(columns=['Final answer', "Annotator Metadata", "file_name"])
subset_60_test.to_json("60_task_subset_test.json", orient="records", indent=4)
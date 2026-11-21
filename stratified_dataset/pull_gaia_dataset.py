import os
import pandas as pd

splits = {"validation": "2023/validation/metadata.parquet"}
df = pd.read_parquet("hf://datasets/gaia-benchmark/GAIA/" + splits["validation"]).drop(columns=["file_path"])

script_dir = os.path.dirname(os.path.abspath(__file__))

full_val = df.copy()
full_val.to_json(os.path.join(script_dir, "full_gaia_val.json"), orient="records", indent=4)

full_test = full_val.copy()
full_test["file_path"] = full_test.apply(lambda x: "data/gaia/" + x["file_name"] if x["file_name"] else "", axis=1)
full_test = full_test.drop(columns=["Final answer", "Annotator Metadata", "file_name"])
full_test.to_json(os.path.join(script_dir, "full_gaia_test.json"), orient="records", indent=4)

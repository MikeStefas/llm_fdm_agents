import csv
import json
import os
from typing import Any

LOG_FIELDNAMES = [
    "node",
    "subagent_steps",
    "state_step",
    "reasoning",
    "compute_time_seconds",
    "tokens_input",
    "tokens_output",
    "response",
    "error",
]


def append_to_csv(log_path: str, log_data: dict[str, Any]) -> None:
    row = dict(log_data)
    if isinstance(row.get("response"), (dict, list)):
        row["response"] = json.dumps(row["response"])

    dirname = os.path.dirname(log_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

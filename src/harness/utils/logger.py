import csv
import json
import os
from typing import Any

LOG_FIELDNAMES = [
    "task_id",
    "node",
    "subagent_steps",
    "state_step",
    "tool_name",
    "tool_args",
    "confidence",
    "probabilities",
    "reasoning",
    "compute_time_seconds",
    "llm_tkns_in",
    "llm_tkns_out",
    "fdm_tkns_in",
    "fdm_tkns_out",
    "response",
    "error",
]


def append_to_csv(log_path: str, log_data: dict[str, Any]) -> None:
    row = dict(log_data)
    if "response" in row and isinstance(row["response"], (dict, list)):
        row["response"] = json.dumps(row["response"])
    if "tool_args" in row and isinstance(row["tool_args"], (dict, list)):
        row["tool_args"] = json.dumps(row["tool_args"])

    dirname = os.path.dirname(log_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_state_to_console_and_md(
    title: str = "",
    input_text: str = "",
    output_text: str = "",
    log_path: str = "",
) -> None:
    """Prints an I/O interaction to the console in markdown format and appends to a .md trace file."""
    entry = (
        f"\n### {title}\n\n"
        f"**Input:**\n```\n{input_text.strip()}\n```\n\n"
        f"**Output:**\n```\n{output_text.strip()}\n```\n\n"
        f"---\n"
    )
#    print(entry)

    if log_path:
        if log_path.endswith(".csv"):
            md_path = log_path[:-4] + ".md"
        elif log_path.endswith(".md"):
            md_path = log_path
        else:
            md_path = log_path + ".md"

        dirname = os.path.dirname(md_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(md_path, mode="a", encoding="utf-8") as f:
            f.write(entry)



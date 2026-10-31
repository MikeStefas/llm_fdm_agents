"""Top-level harness package."""

from harness.utils import (
    FDM_BASE_URL,
    FDM_MODEL,
    LLM_MODEL,
    LLM_SEED,
    LLM_TOP_P,
    LOG_FIELDNAMES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ORCHESTRATOR_LOG_PATH,
    SUBAGENTS_LOG_PATH,
    FinalizerNode,
    append_to_csv,
    get_chat_model,
    prompts,
)
from harness import experiments, orchestrators, react_agents, tools, utils


def main() -> None:
    print("Hello from harness (OpenRouter)!")


__all__ = [
    "main",
    "utils",
    "orchestrators",
    "react_agents",
    "tools",
    "experiments",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_MODEL",
    "FDM_MODEL",
    "FDM_BASE_URL",
    "SUBAGENTS_LOG_PATH",
    "ORCHESTRATOR_LOG_PATH",
    "LLM_SEED",
    "LLM_TOP_P",
    "get_chat_model",
    "LOG_FIELDNAMES",
    "append_to_csv",
    "FinalizerNode",
    "prompts",
]

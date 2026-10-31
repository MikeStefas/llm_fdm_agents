from harness.config import (
    FDM_BASE_URL,
    FDM_MODEL,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ORCHESTRATOR_FDM_MODEL,
    ORCHESTRATOR_LLM_MODEL,
    ORCHESTRATOR_LOG_PATH,
    SUBAGENT_MODEL,
    SUBAGENTS_LOG_PATH,
)
from harness.llm import get_chat_model


def main() -> None:
    print("Hello from harness (OpenRouter)!")

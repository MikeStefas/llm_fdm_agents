from harness.utils.descriptions import (
    FINALIZER_DESCRIPTION,
    FINALIZER_OPTION_DESCRIPTION,
)

NEXT_SUBAGENT_FIELD_DESCRIPTION = "The name of the next subagent to act, or 'finalizer'"
SUBAGENT_PROMPT_FIELD_DESCRIPTION = "The instructions or prompt for the chosen subagent to execute"

LLM_ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a team of subagents.
Given the current task state and previous agent observations, select the single best option to act next.

Available options:
{available_options}

CRITICAL RULES:
- Reply ONLY with the exact name of the chosen option (e.g. 'doc_reader' or 'finalizer').
- Do NOT output explanations, thoughts, punctuation, markdown backticks, or any additional words."""

FDM_ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator of a team of subagents. "
    "Given the task state and previous agent observations, decide which subagent should act next, or select 'finalizer' if the task is complete."
)

__all__ = [
    "FINALIZER_DESCRIPTION",
    "FINALIZER_OPTION_DESCRIPTION",
    "NEXT_SUBAGENT_FIELD_DESCRIPTION",
    "SUBAGENT_PROMPT_FIELD_DESCRIPTION",
    "LLM_ORCHESTRATOR_SYSTEM_PROMPT",
    "FDM_ORCHESTRATOR_INSTRUCTIONS",
]

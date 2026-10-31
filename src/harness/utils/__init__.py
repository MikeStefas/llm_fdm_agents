"""Utility functions, configuration, prompts, and nodes for harness."""

from harness.utils.config import (
    FDM_BASE_URL,
    FDM_MODEL,
    LLM_MODEL,
    LLM_SEED,
    LLM_TOP_P,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ORCHESTRATOR_LOG_PATH,
    SUBAGENTS_LOG_PATH,
)
from harness.utils.finalizer import FinalizerNode
from harness.utils.llm import get_chat_model
from harness.utils.logger import LOG_FIELDNAMES, append_to_csv
from harness.utils.prompts import (
    FDM_ORCHESTRATOR_INSTRUCTIONS,
    FDM_REACT_FINISH_DESCRIPTION,
    FDM_REACT_INSTRUCTIONS_TEMPLATE,
    FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE,
    FDM_ROUTED_LLM_PROMPTED_ROUTING_INSTRUCTIONS,
    FDM_ROUTED_LLM_PROMPTED_SUBAGENT_PROMPT_TEMPLATE,
    FINALIZER_OPTION_DESCRIPTION,
    FINALIZER_TASK_PROMPT_TEMPLATE,
    GAIA_FINALIZER_PROMPT,
    LLM_ORCHESTRATOR_SYSTEM_PROMPT,
    LLM_ROUTED_LLM_PROMPTED_SYSTEM_PROMPT,
    NEXT_SUBAGENT_FIELD_DESCRIPTION,
    SUBAGENT_PROMPT_FIELD_DESCRIPTION,
)
from harness.utils import config, finalizer, llm, logger, prompts

__all__ = [
    # Submodules
    "config",
    "finalizer",
    "llm",
    "logger",
    "prompts",
    # Config
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_MODEL",
    "FDM_MODEL",
    "FDM_BASE_URL",
    "SUBAGENTS_LOG_PATH",
    "ORCHESTRATOR_LOG_PATH",
    "LLM_SEED",
    "LLM_TOP_P",
    # LLM
    "get_chat_model",
    # Logger
    "LOG_FIELDNAMES",
    "append_to_csv",
    # Finalizer Node
    "FinalizerNode",
    # Prompts
    "GAIA_FINALIZER_PROMPT",
    "FINALIZER_TASK_PROMPT_TEMPLATE",
    "FINALIZER_OPTION_DESCRIPTION",
    "NEXT_SUBAGENT_FIELD_DESCRIPTION",
    "SUBAGENT_PROMPT_FIELD_DESCRIPTION",
    "LLM_ORCHESTRATOR_SYSTEM_PROMPT",
    "FDM_ORCHESTRATOR_INSTRUCTIONS",
    "LLM_ROUTED_LLM_PROMPTED_SYSTEM_PROMPT",
    "FDM_ROUTED_LLM_PROMPTED_ROUTING_INSTRUCTIONS",
    "FDM_ROUTED_LLM_PROMPTED_SUBAGENT_PROMPT_TEMPLATE",
    "FDM_REACT_INSTRUCTIONS_TEMPLATE",
    "FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE",
    "FDM_REACT_FINISH_DESCRIPTION",
]
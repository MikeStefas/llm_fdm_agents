from harness.config import (
    FDM_BASE_URL,
    FDM_MODEL,
    LLM_MODEL,
    ORCHESTRATOR_LOG_PATH,
)
from harness.experiments.initialised_subagents import LLM_SUBAGENTS
from harness.orchestrators import (
    FDMOrchestrator,
    FDMRoutedLLMPromptedOrchestrator,
    LLMOrchestrator,
    LLMRoutedLLMPromptedOrchestrator,
)

fdm_orchestrator_with_llm_subagents = FDMOrchestrator(
    model_name=FDM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    base_url=FDM_BASE_URL,
    reasoning=False,
)

llm_orchestrator = LLMOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    reasoning=False,
)

llm_orchestrator_thinking = LLMOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    reasoning=True,
)

fdm_routed_llm_prompted_orchestrator = FDMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    fdm_model=FDM_MODEL,
    fdm_base_url=FDM_BASE_URL,
    reasoning=False,
)

fdm_routed_llm_prompted_orchestrator_thinking = FDMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    fdm_model=FDM_MODEL,
    fdm_base_url=FDM_BASE_URL,
    reasoning=True,
)

llm_routed_llm_prompted_orchestrator = LLMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    reasoning=False,
)

llm_routed_llm_prompted_orchestrator_thinking = LLMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    log_path=ORCHESTRATOR_LOG_PATH,
    reasoning=True,
)

import os

from dotenv import load_dotenv

load_dotenv()

from harness.experiments.initialised_subagents import LLM_SUBAGENTS
from harness.orchestrators import (
    FDMOrchestrator,
    FDMRoutedLLMPromptedOrchestrator,
    LLMOrchestrator,
    LLMRoutedLLMPromptedOrchestrator,
)

LLM_MODEL = os.getenv("ORCHESTRATOR_LLM_MODEL") 
FDM_MODEL = os.getenv("ORCHESTRATOR_FDM_MODEL") 
FDM_BASE_URL = os.getenv("FDM_BASE_URL") 

fdm_orchestrator = FDMOrchestrator(
    model_name=FDM_MODEL,
    subagents=LLM_SUBAGENTS,
    base_url=FDM_BASE_URL,
    reasoning=False,
)

llm_orchestrator = LLMOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    reasoning=False,
)

llm_orchestrator_thinking = LLMOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    reasoning=True,
)

fdm_routed_llm_prompted_orchestrator = FDMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    fdm_model=FDM_MODEL,
    fdm_base_url=FDM_BASE_URL,
    reasoning=False,
)

fdm_routed_llm_prompted_orchestrator_thinking = FDMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    fdm_model=FDM_MODEL,
    fdm_base_url=FDM_BASE_URL,
    reasoning=True,
)

llm_routed_llm_prompted_orchestrator = LLMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    reasoning=False,
)

llm_routed_llm_prompted_orchestrator_thinking = LLMRoutedLLMPromptedOrchestrator(
    model_name=LLM_MODEL,
    subagents=LLM_SUBAGENTS,
    reasoning=True,
)

from config import (
    fdm_client,
    llm_client_thinking_off,
    llm_client_thinking_on,
)
from harness.orchestrators.fdm_orchestrator import FDMOrchestrator
from harness.orchestrators.llm_orchestrator import LLMOrchestrator
from harness.react_agents.subagent_instances import (
    fdm_subagents_thinking_off,
    fdm_subagents_thinking_on,
    llm_subagents_thinking_off,
    llm_subagents_thinking_on,
)

llm_orchestrator_thinking_off = LLMOrchestrator(
    llm_client=llm_client_thinking_off,
    subagents=llm_subagents_thinking_off,
)

llm_orchestrator_thinking_on = LLMOrchestrator(
    llm_client=llm_client_thinking_on,
    subagents=llm_subagents_thinking_on,
)

fdm_orchestrator_thinking_off = FDMOrchestrator(
    fdm_client=fdm_client,
    subagents=fdm_subagents_thinking_off,
)

fdm_orchestrator_thinking_on = FDMOrchestrator(
    fdm_client=fdm_client,
    subagents=fdm_subagents_thinking_on,
)

all_orchestrators = [
    llm_orchestrator_thinking_off,
    llm_orchestrator_thinking_on,
    fdm_orchestrator_thinking_off,
    fdm_orchestrator_thinking_on,
]

ORCHESTRATORS_BY_CONFIG = {
    "llm_thinking_off": llm_orchestrator_thinking_off,
    "llm_thinking_on": llm_orchestrator_thinking_on,
    "fdm_thinking_off": fdm_orchestrator_thinking_off,
    "fdm_thinking_on": fdm_orchestrator_thinking_on,
}


from config import finalizer_llm_client
from harness.mas_builder.mas_builder import MultiAgentSystem
from harness.orchestrators.orch_instances import (
    fdm_orchestrator_thinking_off,
    llm_orchestrator_thinking_off,
    llm_orchestrator_thinking_on,
)
from harness.react_agents.subagent_instances import (
    fdm_subagents_thinking_on,
    llm_subagents_thinking_on,
)


# Config 1: Full LLM with thinking on orchestrator and subagents
mas_1_llm_orch_think_llm_sub_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_on,
    subagents=llm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_1/llm_orch_think_llm_sub_think.csv",
)
mas_8_llm_orch_think_llm_sub_think = mas_1_llm_orch_think_llm_sub_think
mas_llm_orch_think_llm_sub_think = mas_1_llm_orch_think_llm_sub_think

# Config 2: FDM orchestrator with LLM subagents (thinking on)
mas_2_fdm_orch_llm_sub_think = MultiAgentSystem(
    orchestrator=fdm_orchestrator_thinking_off,
    subagents=llm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_2/fdm_orch_llm_sub_think.csv",
)
mas_9_fdm_orch_llm_sub_think = mas_2_fdm_orch_llm_sub_think
mas_fdm_orch_llm_sub_think = mas_2_fdm_orch_llm_sub_think

# Config 3: LLM orchestrator (thinking off) with LLM subagents (thinking on)
mas_3_llm_orch_no_think_llm_sub_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_off,
    subagents=llm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_3/llm_orch_no_think_llm_sub_think.csv",
)
mas_6_llm_orch_no_think_llm_sub_think = mas_3_llm_orch_no_think_llm_sub_think
mas_llm_orch_no_think_llm_sub_think = mas_3_llm_orch_no_think_llm_sub_think

# Config 4: Full FDM (FDM orchestrator + FDM subagents)
mas_4_fdm_orch_fdm_sub = MultiAgentSystem(
    orchestrator=fdm_orchestrator_thinking_off,
    subagents=fdm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_4/fdm_orch_fdm_sub.csv",
)
mas_7_fdm_orch_fdm_sub_no_think = mas_4_fdm_orch_fdm_sub
mas_full_fdm = mas_4_fdm_orch_fdm_sub

# Config 5: LLM orchestrator (thinking on) with FDM subagents
mas_5_llm_orch_think_fdm_sub = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_on,
    subagents=fdm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_5/llm_orch_think_fdm_sub.csv",
)
mas_llm_orch_think_fdm_sub = mas_5_llm_orch_think_fdm_sub

EXPERIMENT_MAS_SYSTEMS = [
    mas_1_llm_orch_think_llm_sub_think,
    mas_2_fdm_orch_llm_sub_think,
    mas_3_llm_orch_no_think_llm_sub_think,
    mas_4_fdm_orch_fdm_sub,
    mas_5_llm_orch_think_fdm_sub,
]

MAS_BY_CONFIG = {
    1: mas_1_llm_orch_think_llm_sub_think,
    2: mas_2_fdm_orch_llm_sub_think,
    3: mas_3_llm_orch_no_think_llm_sub_think,
    4: mas_4_fdm_orch_fdm_sub,
    5: mas_5_llm_orch_think_fdm_sub,
    "mas_1_llm_orch_think_llm_sub_think": mas_1_llm_orch_think_llm_sub_think,
    "mas_2_fdm_orch_llm_sub_think": mas_2_fdm_orch_llm_sub_think,
    "mas_3_llm_orch_no_think_llm_sub_think": mas_3_llm_orch_no_think_llm_sub_think,
    "mas_4_fdm_orch_fdm_sub": mas_4_fdm_orch_fdm_sub,
    "mas_5_llm_orch_think_fdm_sub": mas_5_llm_orch_think_fdm_sub,
    "mas_8_llm_orch_think_llm_sub_think": mas_1_llm_orch_think_llm_sub_think,
    "mas_9_fdm_orch_llm_sub_think": mas_2_fdm_orch_llm_sub_think,
    "mas_6_llm_orch_no_think_llm_sub_think": mas_3_llm_orch_no_think_llm_sub_think,
    "mas_7_fdm_orch_fdm_sub_no_think": mas_4_fdm_orch_fdm_sub,
}

__all__ = [
    "mas_1_llm_orch_think_llm_sub_think",
    "mas_2_fdm_orch_llm_sub_think",
    "mas_3_llm_orch_no_think_llm_sub_think",
    "mas_4_fdm_orch_fdm_sub",
    "mas_5_llm_orch_think_fdm_sub",
    "mas_8_llm_orch_think_llm_sub_think",
    "mas_9_fdm_orch_llm_sub_think",
    "mas_6_llm_orch_no_think_llm_sub_think",
    "mas_7_fdm_orch_fdm_sub_no_think",
    "mas_llm_orch_think_llm_sub_think",
    "mas_fdm_orch_llm_sub_think",
    "mas_llm_orch_no_think_llm_sub_think",
    "mas_full_fdm",
    "mas_llm_orch_think_fdm_sub",
    "EXPERIMENT_MAS_SYSTEMS",
    "MAS_BY_CONFIG",
]

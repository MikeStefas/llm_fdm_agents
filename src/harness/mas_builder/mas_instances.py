from config import finalizer_llm_client
from harness.mas_builder.mas_builder import MultiAgentSystem
from harness.orchestrators.orch_instances import (
    fdm_orchestrator_thinking_off,
    llm_orchestrator_thinking_off,
    llm_orchestrator_thinking_on,
)
from harness.react_agents.subagent_instances import (
    fdm_subagents_thinking_off,
    fdm_subagents_thinking_on,
    llm_subagents_thinking_off,
    llm_subagents_thinking_on,
)


# 1) FDM orch with LLM subagent (no think)
mas_1_fdm_orch_llm_sub_no_think = MultiAgentSystem(
    orchestrator=fdm_orchestrator_thinking_off,
    subagents=llm_subagents_thinking_off,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_1/fdm_orch_llm_sub_no_think.csv",
)

# 2) LLM orch (no think) with LLM subagent (no think)
mas_2_llm_orch_no_think_llm_sub_no_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_off,
    subagents=llm_subagents_thinking_off,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_2/llm_orch_no_think_llm_sub_no_think.csv",
)

# 3) LLM orch (think) with LLM subagent (no think)
mas_3_llm_orch_think_llm_sub_no_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_on,
    subagents=llm_subagents_thinking_off,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_3/llm_orch_think_llm_sub_no_think.csv",
)


# 4) LLM orch (no think) with FDM subagent (thinks on the LLM for args)
mas_4_llm_orch_no_think_fdm_sub_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_off,
    subagents=fdm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_4/llm_orch_no_think_fdm_sub_think.csv",
)

# 5) LLM orch (no think) with FDM subagent (no think)
mas_5_llm_orch_no_think_fdm_sub_no_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_off,
    subagents=fdm_subagents_thinking_off,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_5/llm_orch_no_think_fdm_sub_no_think.csv",
)

# 6) LLM orch (no think) with LLM subagent (think)
mas_6_llm_orch_no_think_llm_sub_think = MultiAgentSystem(
    orchestrator=llm_orchestrator_thinking_off,
    subagents=llm_subagents_thinking_on,
    finalizer_llm_client=finalizer_llm_client,
    log_path="results/mas_6/llm_orch_no_think_llm_sub_think.csv",
)

EXPERIMENT_MAS_SYSTEMS = [
    mas_1_fdm_orch_llm_sub_no_think,
    mas_2_llm_orch_no_think_llm_sub_no_think,
    mas_3_llm_orch_think_llm_sub_no_think,
    mas_4_llm_orch_no_think_fdm_sub_think,
    mas_5_llm_orch_no_think_fdm_sub_no_think,
    mas_6_llm_orch_no_think_llm_sub_think,
]

MAS_BY_CONFIG = {
    "mas_1_fdm_orch_llm_sub_no_think": mas_1_fdm_orch_llm_sub_no_think,
    "mas_2_llm_orch_no_think_llm_sub_no_think": mas_2_llm_orch_no_think_llm_sub_no_think,
    "mas_3_llm_orch_think_llm_sub_no_think": mas_3_llm_orch_think_llm_sub_no_think,
    "mas_4_llm_orch_no_think_fdm_sub_think": mas_4_llm_orch_no_think_fdm_sub_think,
    "mas_5_llm_orch_no_think_fdm_sub_no_think": mas_5_llm_orch_no_think_fdm_sub_no_think,
    "mas_6_llm_orch_no_think_llm_sub_think": mas_6_llm_orch_no_think_llm_sub_think,
}

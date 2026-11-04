FINALIZER_OPTION_DESCRIPTION = (
    "Conclude and synthesize the final answer when the task is finished or sufficient information is gathered."
)

NEXT_SUBAGENT_FIELD_DESCRIPTION = "The name of the next subagent to act, or 'finalizer'"
SUBAGENT_PROMPT_FIELD_DESCRIPTION = "The instructions or prompt for the chosen subagent to execute"

LLM_ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a team of subagents.
Given the task state and previous agent observations, decide which subagent should act next.

Available options:
{available_options}

Reply ONLY with a single word which is the name of the option to act next (e.g. one of the subagents, or 'finalizer' if the task is complete)."""

FDM_ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator of a team of subagents. "
    "Given the task state and previous agent observations, decide which subagent should act next. "
)

LLM_ROUTED_LLM_PROMPTED_SYSTEM_PROMPT = """You are the orchestrator of a team of subagents.
Given the task state, decide which subagent should act next (or choose 'finalizer' if the task is complete) and write a clear, concise instruction/prompt for it.

Available options:
{available_options}"""

FDM_ROUTED_LLM_PROMPTED_ROUTING_INSTRUCTIONS = (
    "You are the orchestrator of a team of subagents. "
    "Given the task state, decide which subagent should act next, or choose 'finalizer' if the task is complete."
)

FDM_ROUTED_LLM_PROMPTED_SUBAGENT_PROMPT_TEMPLATE = """You are the orchestrator of a team of subagents.
The option '{chosen_agent}' ({agent_desc}) has been selected to act next.
Write a clear, concise instruction/prompt for '{chosen_agent}' based on the task state."""


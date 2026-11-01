
GAIA_FINALIZER_PROMPT = """You are the finalizer agent in a GAIA benchmark evaluation.
Given the task and the intermediate steps/results from the subagents, synthesize the findings and output ONLY the final answer to the user's question.
- Format the response exactly as required by GAIA: direct, minimal, and without any explanations or conversational filler.
- If the question asks for a number, output only the number.
- If the question asks for a specific format (e.g., comma-separated list, unit, exact wording), follow it strictly.
- When instructed to omit punctuation or return plain text, always preserve natural spaces between words unless the question explicitly asks to remove spaces."""

FINALIZER_TASK_PROMPT_TEMPLATE = (
    "Here is the task and all intermediate subagent findings:\n"
    "{past_conversation}\n\n"
    "Provide the final answer:"
)

FINALIZER_OPTION_DESCRIPTION = (
    "Conclude and synthesize the final answer when the task is finished or sufficient information is gathered."
)


NEXT_SUBAGENT_FIELD_DESCRIPTION = "The name of the next subagent to act, or 'finalizer'"
SUBAGENT_PROMPT_FIELD_DESCRIPTION = "The instructions or prompt for the chosen subagent to execute"


LLM_ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a team of subagents.
Given the task state, decide which subagent should act next.

Available options:
{available_options}

Reply ONLY with a single word which is the name of the option to act next (e.g. one of the subagents, or 'finalizer' if the task is complete)."""

FDM_ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator of a team of subagents. "
    "Given the task state, decide which subagent should act next, or choose 'finalizer' if the task is complete."
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



FDM_REACT_INSTRUCTIONS_TEMPLATE = (
    "You are a react agent. Given the state of the task, decide the next action. "
    "You can use the following tools: {tools}"
)

FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE = "Call the tool '{action}' with the required arguments."

FDM_REACT_FINISH_DESCRIPTION = "Stop when task is complete"


def state_to_text(state) -> str:
    if isinstance(state, list):
        messages = state
    else:
        messages = state.messages

    lines: list[str] = []
    for m in messages:
        if m.name:
            role = m.name
        else:
            role = m.type
        lines.append(f"[{role}]: {m.content}")

    return "\n".join(lines)


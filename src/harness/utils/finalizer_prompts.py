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


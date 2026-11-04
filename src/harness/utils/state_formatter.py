from langchain_core.messages import AnyMessage


def state_to_text(state: list[AnyMessage] | object) -> str:
    """Converts a list of messages or a LangGraph State object into formatted text."""
    if isinstance(state, list):
        messages = state
    else:
        messages = state.messages

    lines: list[str] = []
    for m in messages:
        role = m.name if getattr(m, "name", None) else m.type
        lines.append(f"[{role}]: {m.content}")

    return "\n".join(lines)


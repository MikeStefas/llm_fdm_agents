from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, Tool

class BaseReactAgent(ABC):

    def __init__(self, subagent: str, tools: list[Tool], description: str, max_iter: int = 10) -> None:
        self.subagent = subagent
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.description = description
        self.max_iter = max_iter

    def get_tools(self) -> list[Tool]:
        return self.tools

    def _state_to_text(self, state: list[BaseMessage]) -> str:
        lines = []
        for msg in state:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                lines.append(f"Tool Output: {msg.content}")
            else:
                lines.append(f"{getattr(msg, 'type', 'Message')}: {msg.content}")
        return "\n".join(lines)

    @abstractmethod
    def react(self, state: list[BaseMessage]) -> str:
        pass

    @abstractmethod
    def log_step(self, step_count: int, tokens_in: int, tokens_out: int, duration: float, error: str | None = None) -> None:
        pass

    def __call__(self, state: list[BaseMessage]) -> str:
        return self.react(state)

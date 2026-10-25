from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool

class BaseReactAgent(ABC):
    subagent: str
    tools: list[Tool]
    tools_by_name: dict[str, Tool]
    description: str
    max_iter: int

    def __init__(self, subagent_name: str, tools: list[Tool], description: str, max_iter: int = 10) -> None:
        self.subagent_name = subagent_name
        self.subagent = subagent_name
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.description = description
        self.max_iter = max_iter

    def get_tools(self) -> list[Tool]:
        return self.tools


    @abstractmethod
    def react(self, state: list[BaseMessage]) -> str:
        pass

    @abstractmethod
    def log_step(self, step_count: int, tokens_in: int, tokens_out: int, duration: float, final_response: str | None = None, error: str | None = None) -> None:
        pass

    def __call__(self, state: list[BaseMessage]) -> str:
        return self.react(state)

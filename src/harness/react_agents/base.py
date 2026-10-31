from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool

class BaseReactAgent(ABC):
    subagent_name: str
    subagent_model: str
    log_path: str
    tools: list[Tool]
    tools_by_name: dict[str, Tool]
    description: str
    max_iter: int

    def __init__(self, subagent_name: str, subagent_model: str, tools: list[Tool], description: str, log_path: str = "subagents.csv", max_iter: int = 10) -> None:
        self.subagent_name = subagent_name
        self.subagent_model = subagent_model
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.description = description
        self.max_iter = max_iter
        self.log_path = log_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def get_tools(self) -> list[Tool]:
        return self.tools


    @abstractmethod
    def react(self, state: list[BaseMessage], state_step: int = 0) -> str:
        pass

    @abstractmethod
    def log_step(self, substep_count: int, tokens_in: int, tokens_out: int, duration: float, final_response: str | None = None, error: str | None = None, state_step: int = 0) -> None:
        pass

    def __call__(self, state: list[BaseMessage], state_step: int = 0) -> str:
        return self.react(state, state_step=state_step)

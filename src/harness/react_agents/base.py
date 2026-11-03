from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool
from harness.utils.prompts import get_subagent_system_prompt, state_to_text
from harness.utils import append_to_csv


class BaseReactAgent(ABC):
    subagent_name: str
    reasoning: bool = False
    log_path: Optional[str]
    tools: list[Tool]
    tools_by_name: dict[str, Tool]
    description: str
    system_prompt: str
    max_iter: int

    def __init__(
        self,
        subagent_name: str,
        tools: list[Tool],
        description: str,
        log_path: Optional[str] = None,
        max_iter: int = 10,
    ) -> None:
        self.subagent_name = subagent_name
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.description = description
        self.system_prompt = get_subagent_system_prompt(subagent_name, description)
        self.max_iter = max_iter
        self.log_path = log_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def get_tools(self) -> list[Tool]:
        return self.tools

    @staticmethod
    def state_to_text(state: list[BaseMessage]) -> str:
        return state_to_text(state)

    @abstractmethod
    def react(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        pass

    def log_step(
        self,
        substep_count: int,
        llm_tkns_in: int,
        llm_tkns_out: int,
        fdm_tkns_in: int,
        fdm_tkns_out: int,
        duration: float,
        final_response: str | None = None,
        error: str | None = None,
        state_step: int = 0,
        task_id: str | None = None,
    ) -> None:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before logging.")

        log = {
            "task_id": task_id,
            "node": self.subagent_name,
            "subagent_steps": substep_count,
            "state_step": state_step,
            "compute_time_seconds": duration,
            "llm_tkns_in": llm_tkns_in,
            "llm_tkns_out": llm_tkns_out,
            "fdm_tkns_in": fdm_tkns_in,
            "fdm_tkns_out": fdm_tkns_out,
            "response": final_response,
            "error": error,
        }
        append_to_csv(self.log_path, log)

    def __call__(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before running react agent.")
        return self.react(state, state_step=state_step, task_id=task_id)

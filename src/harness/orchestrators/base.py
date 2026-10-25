from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from harness.react_agents.base import BaseReactAgent


class NextSubagent(BaseModel):
    next_subagent: str = Field(description="The name of the next subagent to act, or 'Finish'")


class OrchestratorDecision(BaseModel):
    next_subagent: str = Field(description="The name of the next subagent to act, or 'Finish'")
    subagent_prompt: str = Field(description="The instructions or prompt for the chosen subagent to execute")


class BaseOrchestrator(ABC):
    def __init__(self, model_name: str, subagents: list[BaseReactAgent]) -> None:
        self.model_name = model_name
        self.subagents = subagents
        self.agent_descriptions = {agent.subagent: agent.description for agent in self.subagents}
        self.actions = list(self.agent_descriptions.keys())

    def get_subagents(self) -> list[BaseReactAgent]:
        return self.subagents

    @abstractmethod
    def orchestrate(self, state):
        pass

    @abstractmethod
    def log_step(self, *args, **kwargs):
        pass

    def __call__(self, state) -> Any:
        return self.orchestrate(state)

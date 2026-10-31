from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from harness.react_agents.base import BaseReactAgent


class NextSubagentDecision(BaseModel):
    next_subagent: str = Field(description="The name of the next subagent to act, or 'finalizer'")


class DualOrchestratorDecision(BaseModel):
    next_subagent: str = Field(description="The name of the next subagent to act, or 'finalizer'")
    subagent_prompt: str = Field(description="The instructions or prompt for the chosen subagent to execute")


class BaseOrchestrator(ABC):
    model_name: str
    log_path: str
    subagents: list[BaseReactAgent]
    agent_descriptions: dict[str, str]
    subagents_names: list[str]

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        log_path: str = "orchestrator.csv",
    ) -> None:
        self.model_name = model_name
        self.subagents = subagents
        self.log_path = log_path
        self.agent_descriptions = {agent.subagent_name: agent.description for agent in self.subagents}
        self.subagents_names = [agent.subagent_name for agent in self.subagents]

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def get_subagents(self) -> list[BaseReactAgent]:
        return self.subagents

    @abstractmethod
    def orchestrate(self, state: object) -> object:
        pass


    def __call__(self, state: object) -> object:
        return self.orchestrate(state)

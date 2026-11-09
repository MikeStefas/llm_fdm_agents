from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typesafe_sdk import SystemOneResponse, TypeSafeClient
from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv
from harness.utils.orchestrator_prompts import (
    NEXT_SUBAGENT_FIELD_DESCRIPTION,
    SUBAGENT_PROMPT_FIELD_DESCRIPTION,
)
from harness.utils.state_formatter import state_to_text


class NextSubagentDecision(BaseModel):
    next_subagent: str = Field(description=NEXT_SUBAGENT_FIELD_DESCRIPTION)


class DualOrchestratorDecision(BaseModel):
    next_subagent: str = Field(description=NEXT_SUBAGENT_FIELD_DESCRIPTION)
    subagent_prompt: str = Field(description=SUBAGENT_PROMPT_FIELD_DESCRIPTION)


class BaseOrchestrator(ABC):
    log_path: Optional[str]
    subagents: list[BaseReactAgent]
    agent_descriptions: dict[str, str]
    subagents_names: list[str]
    llm_client: Optional[ChatOpenAI]
    fdm_client: Optional[TypeSafeClient]

    def __init__(
        self,
        subagents: Optional[list[BaseReactAgent]] = None,
        log_path: Optional[str] = None,
        llm_client: Optional[ChatOpenAI] = None,
        fdm_client: Optional[TypeSafeClient] = None,
    ) -> None:
        self.log_path = log_path
        self.llm_client = llm_client
        self.fdm_client = fdm_client
        self.subagents = []
        self.agent_descriptions = {}
        self.subagents_names = []
        if subagents is not None:
            self.set_subagents(subagents)

    def set_subagents(self, subagents: list[BaseReactAgent]) -> None:
        self.subagents = subagents
        self.agent_descriptions = {agent.subagent_name: agent.description for agent in self.subagents}
        self.subagents_names = [agent.subagent_name for agent in self.subagents]

    def has_subagents(self) -> bool:
        return bool(self.subagents)

    def ensure_has_subagents(self) -> None:
        if not self.has_subagents():
            raise ValueError(
                f"{self.__class__.__name__} has no subagents configured. "
                "Call set_subagents() with a list of subagents before orchestrating."
            )

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def get_subagents(self) -> list[BaseReactAgent]:
        return self.subagents

    @staticmethod
    def state_to_text(state: object) -> str:
        return state_to_text(state)

    @abstractmethod
    def orchestrate(self, state: object) -> object:
        pass

    def log_step(
        self,
        state_step: int,
        decision: str,
        res: AIMessage | SystemOneResponse | None = None,
        duration: float = 0.0,
        error: str | None = None,
        task_id: str | None = None,
        llm_tkns_in: int = 0,
        llm_tkns_out: int = 0,
        fdm_tkns_in: int = 0,
        fdm_tkns_out: int = 0,
        confidence: object = None,
        probabilities: object = None,
    ) -> None:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before logging.")

        if isinstance(res, SystemOneResponse):
            if res.answers:
                ans = res.answers["next_subagent"]
                confidence = ans.confidence
                probabilities = ans.probabilities
            if res.usage is not None:
                fdm_tkns_in = res.usage.input_tokens
                fdm_tkns_out = res.usage.output_tokens
        elif isinstance(res, AIMessage):
            if res.usage_metadata:
                llm_tkns_in = res.usage_metadata["input_tokens"]
                llm_tkns_out = res.usage_metadata["output_tokens"]

        log = {
            "task_id": task_id,
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "confidence": confidence,
            "probabilities": probabilities,
            "compute_time_seconds": duration,
            "llm_tkns_in": llm_tkns_in,
            "llm_tkns_out": llm_tkns_out,
            "fdm_tkns_in": fdm_tkns_in,
            "fdm_tkns_out": fdm_tkns_out,
            "response": decision,
            "error": error,
        }
        append_to_csv(self.log_path, log)

    def __call__(self, state: object) -> object:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before orchestrating.")
        return self.orchestrate(state)

import json
import time
from typing_extensions import override
from typesafe_sdk import TypeSafeClient, Choice, ChoiceAnswer, SystemOneResponse
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent


class FDMOrchestrator(BaseOrchestrator):
    base_url: str
    reasoning: bool
    fdm_client: TypeSafeClient

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents)
        self.base_url = base_url
        self.reasoning = reasoning
        self.fdm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )

    @override
    def orchestrate(self, state: object) -> str:
        t_start = time.perf_counter()

        res = self.fdm_client.system_one(
            state=str(state),
            questions={
                "next_subagent": Choice(
                    instructions="You are the orchestrator of a team of subagents. Given the task state, decide which subagent should act next.",
                    criteria=self.agent_descriptions,
                )
            },
        )
        t_end = time.perf_counter()

        decision = str(res.answers["next_subagent"].choice)

        if decision in self.subagents_names:
            error = None
        else:
            error = f"Invalid action: {decision}. Allowed: {self.subagents_names}"

        self.log_step(state, decision, res, t_end - t_start, error)
        return decision

    def log_step(
        self,
        state_step: object,
        decision: str,
        state_res: SystemOneResponse,
        duration: float,
        error: str | None,
    ) -> None:
        ans = state_res.answers["next_subagent"]
        confidence = ans.confidence
        probabilities = ans.probabilities

        tokens_in = state_res.usage.input_tokens
        tokens_out = state_res.usage.output_tokens

        log = {
            "orchestrator_type": "FDMOrchestrator",
            "model_name": self.model_name,
            "reasoning": self.reasoning,
            "state_step": state_step,
            "next_subagent": decision,
            "confidence": confidence,
            "probabilities": probabilities,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            _ = f.write(json.dumps(log) + "\n")

import json
import time
from typing_extensions import override
from typesafe_sdk import TypeSafeClient, Choice, ChoiceAnswer, SystemOneResponse
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv
from harness.utils.prompts import FDM_ORCHESTRATOR_INSTRUCTIONS, FINALIZER_OPTION_DESCRIPTION


class FDMOrchestrator(BaseOrchestrator):
    base_url: str
    reasoning: bool
    fdm_client: TypeSafeClient

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        log_path: str = "orchestrator.csv",
        base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents, log_path)
        self.base_url = base_url
        self.reasoning = reasoning
        self.fdm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )

    @override
    def orchestrate(self, state: object) -> str:
        t_start = time.perf_counter()
        task_id = state.get("task_id") if isinstance(state, dict) else state.task_id

        available_options = dict(self.agent_descriptions)
        available_options["finalizer"] = FINALIZER_OPTION_DESCRIPTION

        res = self.fdm_client.system_one(
            state=self.state_to_text(state),
            questions={
                "next_subagent": Choice(
                    instructions=FDM_ORCHESTRATOR_INSTRUCTIONS,
                    criteria=available_options,
                )
            },
        )
        t_end = time.perf_counter()

        decision = str(res.answers["next_subagent"].choice).strip().strip("'\"").lower()
        if decision == "finish":
            decision = "finalizer"

        allowed = self.subagents_names + ["finalizer"]
        if decision in allowed:
            error = None
        else:
            error = f"Invalid action: {decision}. Allowed: {allowed}"

        step_val = state.get("state_step", 0) if isinstance(state, dict) else state.state_step
        self.log_step(step_val, decision, res, t_end - t_start, error, task_id=task_id)
        return decision

    def log_step(
        self,
        state_step: int,
        decision: str,
        state_res: SystemOneResponse,
        duration: float,
        error: str | None,
        task_id: str | None = None,
    ) -> None:
        ans = state_res.answers["next_subagent"]
        confidence = ans.confidence
        probabilities = ans.probabilities

        fdm_tkns_in = state_res.usage.input_tokens
        fdm_tkns_out = state_res.usage.output_tokens

        log = {
            "task_id": task_id,
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "confidence": confidence,
            "probabilities": probabilities,
            "compute_time_seconds": duration,
            "llm_tkns_in": 0,
            "llm_tkns_out": 0,
            "fdm_tkns_in": fdm_tkns_in,
            "fdm_tkns_out": fdm_tkns_out,
            "response": decision,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

import json
import time
from typesafe_sdk import TypeSafeClient, Choice
from harness.orchestrators.base import BaseOrchestrator


class FDMOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        model_name: str,
        subagents: list,
        base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents)
        self.base_url = base_url
        self.reasoning = reasoning
        self.dfm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )

    def orchestrate(self, state):
        t_start = time.perf_counter()

        res = self.dfm_client.system_one(
            state=state,
            questions={"next_subagent": Choice(
                instructions="You are the orchestrator of a team of subagents. Given the task state, decide which subagent should act next.",
                criteria=self.agent_descriptions,
            )}
        )
        t_end = time.perf_counter()
        decision = res.answers["next_subagent"].choice
        error = None if decision in self.actions else f"Invalid action: {decision}. Allowed: {self.actions}"

        self.log_step(state, res, t_end - t_start, error)
        return decision

    def log_step(self, state_step, state_res, duration: float, error: str | None = None):
        ans = state_res.answers.get("next_subagent")
        confidence = getattr(ans, "confidence", None) if ans else None
        probabilities = getattr(ans, "probabilities", None) if ans else None
        tokens_in = state_res.usage.input_tokens if hasattr(state_res, "usage") else 0
        tokens_out = state_res.usage.output_tokens if hasattr(state_res, "usage") else 0

        log = {
            "timestamp": time.time(),
            "orchestrator_type": "FDMOrchestrator",
            "model_name": self.model_name,
            "dfm_model": self.model_name,
            "reasoning": self.reasoning,
            "step": state_step,
            "next_subagent": ans.choice if ans else None,
            "confidence": confidence,
            "probabilities": probabilities,
            "subagent_prompt": None,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")


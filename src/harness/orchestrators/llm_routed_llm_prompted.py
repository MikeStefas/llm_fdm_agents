import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from harness.orchestrators.base import BaseOrchestrator, OrchestratorDecision


class LLMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    def __init__(self, model_name: str, subagents: list, reasoning: bool = False) -> None:
        super().__init__(model_name, subagents)
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )
        self.structured_llm = self.llm_client.with_structured_output(OrchestratorDecision, include_raw=True)

    def orchestrate(self, state):
        t_start = time.perf_counter()
        result = self.structured_llm.invoke([
            SystemMessage(content=f"""You are the orchestrator of a team of subagents. Given the task state, decide which subagent should act next and write a clear, concise instruction/prompt for it.

Available subagents:
{json.dumps(self.agent_descriptions)}"""),
            HumanMessage(content=f"""{state}"""),
        ])
        t_end = time.perf_counter()
        decision: OrchestratorDecision = result["parsed"]
        raw_res = result["raw"]

        if decision.next_subagent in self.actions:
            error = None
        else:
            error = f"Invalid action: {decision.next_subagent}. Allowed: {self.actions}"

        self.log_step(state, decision, raw_res, t_end - t_start, error)

        if error:
            raise ValueError(error)

        return {
            "next_subagent": decision.next_subagent,
            "subagent_prompt": decision.subagent_prompt,
        }

    def log_step(self, state_step, decision: OrchestratorDecision, raw_res, duration: float, error: str | None = None):
        tokens_in = raw_res.usage_metadata.get("input_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else 0
        tokens_out = raw_res.usage_metadata.get("output_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else 0
        tokens_tot = raw_res.usage_metadata.get("total_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else (tokens_in + tokens_out)

        log = {
            "timestamp": time.time(),
            "orchestrator_type": "LLMRoutedLLMPromptedOrchestrator",
            "model_name": self.model_name,
            "dfm_model": None,
            "reasoning": self.reasoning,
            "step": state_step,
            "next_subagent": decision.next_subagent,
            "confidence": None,
            "probabilities": None,
            "subagent_prompt": decision.subagent_prompt,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_total": tokens_tot,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

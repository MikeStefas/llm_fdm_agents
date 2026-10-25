import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from harness.orchestrators.base import BaseOrchestrator, NextSubagent


class LLMOrchestrator(BaseOrchestrator):
    def __init__(self, model_name: str, subagents: list, reasoning: bool = False) -> None:
        super().__init__(model_name, subagents)
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )

    def orchestrate(self, state):
        t_start = time.perf_counter()
        result = self.llm_client.with_structured_output(NextSubagent, include_raw=True).invoke([
            SystemMessage(content=f"""You are the orchestrator of a team of subagents.
Given the task state, decide which subagent should act next.

Available subagents:
{json.dumps(self.agent_descriptions)}"""),
            HumanMessage(content=f"""{state}""")
        ])
        t_end = time.perf_counter()
        parsed: NextSubagent = result["parsed"]
        raw_res = result["raw"]
        decision = parsed.next_subagent if parsed else str(raw_res.content).strip()
        
        if decision in self.actions:
            error = None
        else:
            matched = [a for a in self.actions if a in decision]
            if len(matched) == 1:
                decision = matched[0]
                error = None
            else:
                error = f"Invalid action: {decision}. Allowed: {self.actions}"

        self.log_step(state, decision, raw_res, t_end - t_start, error)
        return decision
    
    def log_step(self, state_step, decision: str, raw_res, duration: float, error: str | None = None):
        tokens_in = raw_res.usage_metadata.get("input_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else 0
        tokens_out = raw_res.usage_metadata.get("output_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else 0
        tokens_tot = raw_res.usage_metadata.get("total_tokens") if hasattr(raw_res, "usage_metadata") and raw_res.usage_metadata else (tokens_in + tokens_out)

        log = {
            "timestamp": time.time(),
            "orchestrator_type": "LLMOrchestrator",
            "model_name": self.model_name,
            "dfm_model": None,
            "reasoning": self.reasoning,
            "step": state_step,
            "next_subagent": decision,
            "confidence": None,
            "probabilities": None,
            "subagent_prompt": None,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_total": tokens_tot,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

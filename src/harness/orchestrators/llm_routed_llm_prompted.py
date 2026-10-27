import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from harness.orchestrators.base import BaseOrchestrator, DualOrchestratorDecision
from harness.react_agents.base import BaseReactAgent


class LLMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    reasoning: bool
    llm_client: ChatOllama
    structured_llm: Runnable[object, object]

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        log_path: str = "orchestrator.jsonl",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents, log_path)
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )
        self.structured_llm = cast(
            Runnable[object, object],
            self.llm_client.with_structured_output(DualOrchestratorDecision, include_raw=True),
        )

    @override
    def orchestrate(self, state: object) -> dict[str, str]:
        t_start = time.perf_counter()

        raw_result = self.structured_llm.invoke([
            SystemMessage(content=f"""
            You are the orchestrator of a team of subagents.
            Given the task state, decide which subagent should act next and write a clear, concise instruction/prompt for it.

            Available subagents:
            {json.dumps(self.agent_descriptions)}
            """),
            HumanMessage(content=f"""{state}"""),
        ])
        t_end = time.perf_counter()

        decision = raw_result["parsed"]
        raw_res = raw_result["raw"]

        if decision:
            next_subagent = decision.next_subagent
            subagent_prompt = decision.subagent_prompt
        elif raw_res:
            next_subagent = str(raw_res.content).strip()
            subagent_prompt = ""
        else:
            next_subagent = ""
            subagent_prompt = ""

        if next_subagent in self.subagents_names:
            error = None
        else:
            error = f"Invalid action: {next_subagent}. Allowed: {self.subagents_names}"

        step_val = getattr(state, "state_step", getattr(state, "step", 0)) if not isinstance(state, (int, str)) else 0
        self.log_step(step_val, next_subagent, subagent_prompt, raw_res, t_end - t_start, error)
        return {
            "next_subagent": next_subagent,
            "subagent_prompt": subagent_prompt,
        }

    def log_step(
        self,
        state_step: int,
        next_subagent: str,
        subagent_prompt: str,
        raw_res: AIMessage | None,
        duration: float,
        error: str | None,
    ) -> None:
        tokens_in = raw_res.usage_metadata["input_tokens"]
        tokens_out = raw_res.usage_metadata["output_tokens"]

        log = {
            "orchestrator_type": "LLMRoutedLLMPromptedOrchestrator",
            "model_name": self.model_name,
            "reasoning": self.reasoning,
            "state_step": state_step,
            "next_subagent": next_subagent,
            "subagent_prompt": subagent_prompt,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "error": error,
        }
        print(log)
        with open(self.log_path, "a") as f:
            _ = f.write(json.dumps(log, default=str) + "\n")

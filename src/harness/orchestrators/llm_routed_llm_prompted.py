import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from harness.orchestrators.base import BaseOrchestrator, DualOrchestratorDecision
from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv, get_chat_model
from harness.utils.prompts import FINALIZER_OPTION_DESCRIPTION, LLM_ROUTED_LLM_PROMPTED_SYSTEM_PROMPT


class LLMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    reasoning: bool
    llm_client: ChatOpenAI
    structured_llm: Runnable[object, object]

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        log_path: str = "orchestrator.csv",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents, log_path)
        self.reasoning = reasoning
        self.llm_client = get_chat_model(
            reasoning=self.reasoning,
        )
        self.structured_llm = cast(
            Runnable[object, object],
            self.llm_client.with_structured_output(DualOrchestratorDecision, include_raw=True),
        )

    @override
    def orchestrate(self, state: object) -> dict[str, str]:
        t_start = time.perf_counter()

        available_options = dict(self.agent_descriptions)
        available_options["finalizer"] = FINALIZER_OPTION_DESCRIPTION

        state_text = self.state_to_text(state)

        raw_result = self.structured_llm.invoke([
            SystemMessage(content=LLM_ROUTED_LLM_PROMPTED_SYSTEM_PROMPT.format(
                available_options=json.dumps(available_options)
            )),
            HumanMessage(content=state_text),
        ])
        t_end = time.perf_counter()

        decision = raw_result["parsed"]
        raw_res = raw_result["raw"]

        if decision:
            next_subagent = str(decision.next_subagent).strip().strip("'\"").lower()
            subagent_prompt = decision.subagent_prompt
        elif raw_res:
            next_subagent = str(raw_res.content).strip().strip("'\"").lower()
            subagent_prompt = ""
        else:
            next_subagent = ""
            subagent_prompt = ""

        if next_subagent == "finish":
            next_subagent = "finalizer"

        allowed = self.subagents_names + ["finalizer"]
        if next_subagent in allowed:
            error = None
        else:
            error = f"Invalid action: {next_subagent}. Allowed: {allowed}"

        step_val = state.state_step
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
        usage = raw_res.usage_metadata
        tokens_in = usage["input_tokens"]
        tokens_out = usage["output_tokens"]

        response = {
            "next_subagent": next_subagent,
            "subagent_prompt": subagent_prompt,
        }

        log = {
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response": response,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

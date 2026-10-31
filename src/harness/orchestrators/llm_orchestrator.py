import json
import time
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv, get_chat_model
from harness.utils.prompts import FINALIZER_OPTION_DESCRIPTION, LLM_ORCHESTRATOR_SYSTEM_PROMPT


class LLMOrchestrator(BaseOrchestrator):
    reasoning: bool
    llm_client: ChatOpenAI

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

    @override
    def orchestrate(self, state: object) -> str:
        t_start = time.perf_counter()
        task_id = state.get("task_id") if isinstance(state, dict) else state.task_id
        
        state_text = self.state_to_text(state)

        available_options = dict(self.agent_descriptions)
        available_options["finalizer"] = FINALIZER_OPTION_DESCRIPTION

        res = self.llm_client.invoke([
            SystemMessage(content=LLM_ORCHESTRATOR_SYSTEM_PROMPT.format(
                available_options=json.dumps(available_options)
            )),
            HumanMessage(content=state_text)
        ])
        t_end = time.perf_counter()
        
        decision = str(res.content).strip().strip("'\"").lower()
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
        res: AIMessage,
        duration: float,
        error: str | None,
        task_id: str | None = None,
    ) -> None:
        usage = res.usage_metadata
        llm_tkns_in = usage["input_tokens"]
        llm_tkns_out = usage["output_tokens"]

        log = {
            "task_id": task_id,
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "compute_time_seconds": duration,
            "llm_tkns_in": llm_tkns_in,
            "llm_tkns_out": llm_tkns_out,
            "fdm_tkns_in": 0,
            "fdm_tkns_out": 0,
            "response": decision,
            "error": error,
        }
        append_to_csv(self.log_path, log)

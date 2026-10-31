import json
import time
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from harness.llm import get_chat_model
from harness.logger import append_to_csv
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent


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
        
        history_lines = []
        for m in state.messages:
            role = m.name if m.name else m.type
            history_lines.append(f"[{role}]: {m.content}")
        state_text = "\n".join(history_lines)

        available_options = dict(self.agent_descriptions)
        available_options["finalizer"] = "Conclude and synthesize the final answer when the task is finished or sufficient information is gathered."

        res = self.llm_client.invoke([
            SystemMessage(content=f"""
            You are the orchestrator of a team of subagents.
            Given the task state, decide which subagent should act next.

            Available options:
            {json.dumps(available_options)}
            
            Reply ONLY with a single word which is the name of the option to act next (e.g. one of the subagents, or 'finalizer' if the task is complete).
            """),
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

        if res:
            step_val = state.state_step
            self.log_step(step_val, decision, res, t_end - t_start, error)
        return decision
    
    def log_step(
        self,
        state_step: int,
        decision: str,
        res: AIMessage,
        duration: float,
        error: str | None,
    ) -> None:

        usage = res.usage_metadata
        tokens_in = usage["input_tokens"]
        tokens_out = usage["output_tokens"]

        log = {
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response": decision,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

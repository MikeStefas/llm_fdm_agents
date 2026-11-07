import json
import time
from typing import Optional
from typing_extensions import override
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils import print_state_to_console_and_md
from harness.utils.prompts import FINALIZER_OPTION_DESCRIPTION, LLM_ORCHESTRATOR_SYSTEM_PROMPT


class LLMOrchestrator(BaseOrchestrator):
    llm_client: ChatOpenAI

    def __init__(
        self,
        llm_client: ChatOpenAI,
        subagents: Optional[list[BaseReactAgent]] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            subagents=subagents,
            log_path=log_path,
            llm_client=llm_client
        )

    @override
    def orchestrate(self, state: object) -> str:
        self.ensure_has_subagents()
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before orchestrating.")
        t_start = time.perf_counter()
        task_id = state.task_id 
        
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

        allowed = self.subagents_names + ["finalizer"]
        if decision in allowed:
            error = None
        else:
            error = f"Invalid action: {decision}. Allowed: {allowed}"

        step_val = state.state_step
        self.log_step(step_val, decision, res, t_end - t_start, error, task_id=task_id)

        prompt_text = (
            f"System Prompt:\n{LLM_ORCHESTRATOR_SYSTEM_PROMPT.format(available_options=json.dumps(available_options))}\n\n"
            f"Human Message / State:\n{state_text}"
        )
        print_state_to_console_and_md(
            log_path=self.log_path,
            title="llm orch",
            input_text=prompt_text,
            output_text=decision,
        )
        return decision

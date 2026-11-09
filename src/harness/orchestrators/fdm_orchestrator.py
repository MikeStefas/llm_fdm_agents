import time
from typing import Optional
from typing_extensions import override
from typesafe_sdk import Choice, TypeSafeClient
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils import print_state_to_console_and_md
from harness.utils.orchestrator_prompts import (
    FDM_ORCHESTRATOR_INSTRUCTIONS,
    FINALIZER_OPTION_DESCRIPTION,
)


class FDMOrchestrator(BaseOrchestrator):
    fdm_client: TypeSafeClient

    def __init__(
        self,
        fdm_client: TypeSafeClient,
        subagents: Optional[list[BaseReactAgent]] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            subagents=subagents,
            log_path=log_path,
            llm_client=None,
            fdm_client=fdm_client,
        )

    @override
    def orchestrate(self, state: object) -> str:
        self.ensure_has_subagents()
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before orchestrating.")
        t_start = time.perf_counter()
        task_id = state.task_id

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

        allowed = self.subagents_names + ["finalizer"]
        if decision in allowed:
            error = None
        else:
            error = f"Invalid action: {decision}. Allowed: {allowed}"

        step_val = state.state_step
        self.log_step(step_val, decision, res, t_end - t_start, error, task_id=task_id)

        input_text = (
            f"State:\n{self.state_to_text(state)}\n\n"
            f"Instructions:\n{FDM_ORCHESTRATOR_INSTRUCTIONS}\n\n"
            f"Available options:\n{available_options}"
        )
        print_state_to_console_and_md(
            log_path=self.log_path,
            title="fdm orch",
            input_text=input_text,
            output_text=decision,
        )
        return decision

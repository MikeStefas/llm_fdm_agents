import time
from typing import Optional
from typing_extensions import override
from langchain_openai import ChatOpenAI
from typesafe_sdk import Choice, TypeSafeClient
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils.prompts import FDM_ORCHESTRATOR_INSTRUCTIONS, FINALIZER_OPTION_DESCRIPTION


class FDMOrchestrator(BaseOrchestrator):
    fdm_client: TypeSafeClient

    def __init__(
        self,
        subagents: list[BaseReactAgent],
        fdm_client: TypeSafeClient,
        llm_client: ChatOpenAI,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            subagents,
            log_path=log_path,
            llm_client=llm_client,
            fdm_client=fdm_client,
        )

    @override
    def orchestrate(self, state: object) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before orchestrating.")
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

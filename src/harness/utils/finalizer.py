import time
from typing import Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from harness.utils.llm import get_chat_model
from harness.utils.logger import append_to_csv
from harness.utils.prompts import FINALIZER_TASK_PROMPT_TEMPLATE, GAIA_FINALIZER_PROMPT


class FinalizerNode:
    model_name: Optional[str]
    system_prompt: str
    log_path: str

    def __init__(
        self,
        model_name: Optional[str] = None,
        log_path: str ,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = GAIA_FINALIZER_PROMPT
        self.log_path = log_path
        self.llm = get_chat_model()

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    @staticmethod
    def state_to_text(messages: list[AnyMessage]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m.name if m.name else m.type
            lines.append(f"[{role}]: {m.content}")
        return "\n".join(lines)

    def finalize(
        self,
        messages: list[AnyMessage],
        state_step: int = 0,
        task_id: Optional[str] = None,
    ) -> str:
        t_start = time.perf_counter()
        past_conversation = self.state_to_text(messages)

        try:
            res = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=FINALIZER_TASK_PROMPT_TEMPLATE.format(past_conversation=past_conversation)
                ),
            ])
            output_text = str(res.content).strip()
            duration = time.perf_counter() - t_start
            usage = res.usage_metadata
            llm_tkns_in = usage["input_tokens"]
            llm_tkns_out = usage["output_tokens"]
            self.log_step(
                state_step=state_step,
                llm_tkns_in=llm_tkns_in,
                llm_tkns_out=llm_tkns_out,
                fdm_tkns_in=0,
                fdm_tkns_out=0,
                duration=duration,
                response=output_text,
                error=None,
                task_id=task_id,
            )
            return output_text
        except Exception as e:
            duration = time.perf_counter() - t_start
            error = str(e)
            self.log_step(
                state_step=state_step,
                llm_tkns_in=0,
                llm_tkns_out=0,
                fdm_tkns_in=0,
                fdm_tkns_out=0,
                duration=duration,
                response=None,
                error=error,
                task_id=task_id,
            )
            raise

    def log_step(
        self,
        state_step: int,
        llm_tkns_in: int,
        llm_tkns_out: int,
        fdm_tkns_in: int,
        fdm_tkns_out: int,
        duration: float,
        response: Optional[str] = None,
        error: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        log = {
            "task_id": task_id,
            "node": "finalizer",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": False,
            "compute_time_seconds": duration,
            "llm_tkns_in": llm_tkns_in,
            "llm_tkns_out": llm_tkns_out,
            "fdm_tkns_in": fdm_tkns_in,
            "fdm_tkns_out": fdm_tkns_out,
            "response": response,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

    def __call__(
        self,
        messages: list[AnyMessage],
        state_step: int = 0,
        task_id: Optional[str] = None,
    ) -> str:
        return self.finalize(messages, state_step=state_step, task_id=task_id)

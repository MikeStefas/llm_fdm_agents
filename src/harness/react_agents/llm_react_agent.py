import json
import time
from typing import Optional
from typing_extensions import override
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from harness.react_agents.base import BaseReactAgent


class LLMReactAgent(BaseReactAgent):
    bound_llm: Runnable[object, object]

    def __init__(
        self,
        subagent_name: str,
        tools: list[Tool],
        description: str,
        llm_client: ChatOpenAI,
        log_path: Optional[str] = None,
        max_iter: int = 10,
    ) -> None:
        super().__init__(
            subagent_name=subagent_name,
            tools=tools,
            description=description,
            llm_client=llm_client,
            log_path=log_path,
            max_iter=max_iter,
        )
        self.bound_llm = self.llm_client.bind_tools(self.tools)

    @override
    def react(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before running react agent.")
        substep_count = 0
        final_res = None
        messages = self.prepare_messages(state)
        t_substep_start = time.perf_counter()
        last_tool_input = None

        try:
            for i in range(self.max_iter):
                substep_count = i + 1
                t_substep_start = time.perf_counter()
                step_tkns_in = 0
                step_tkns_out = 0

                # llm decides next action/tool and args
                t_api_start = time.perf_counter()
                res = self.bound_llm.invoke(messages)
                duration = time.perf_counter() - t_api_start
                if res.usage_metadata:
                    step_tkns_in = res.usage_metadata["input_tokens"]
                    step_tkns_out = res.usage_metadata["output_tokens"]

                # If no tool calls, this is the final natural response directly
                if not res.tool_calls:
                    final_res = res
                    self.log_step(
                        substep_count=substep_count,
                        llm_tkns_in=step_tkns_in,
                        llm_tkns_out=step_tkns_out,
                        fdm_tkns_in=0,
                        fdm_tkns_out=0,
                        duration=duration,
                        final_response=str(res.content),
                        error=None,
                        state_step=state_step,
                        task_id=task_id,
                    )
                    break

                # Enforce single tool call!!!!
                active_tool_call = res.tool_calls[0]
                res.tool_calls = [active_tool_call]
                messages.append(res)  # log required tool

                tool_msg, substep_error, last_tool_input, is_repeated = self.execute_tool_call(
                    active_tool_call, last_input=last_tool_input
                )
                messages.append(tool_msg)  # log tool res

                step_response = res.content if res.content else str(tool_msg.content)
                self.log_step(
                    substep_count=substep_count,
                    llm_tkns_in=step_tkns_in,
                    llm_tkns_out=step_tkns_out,
                    fdm_tkns_in=0,
                    fdm_tkns_out=0,
                    duration=duration,
                    final_response=step_response,
                    tool_name=str(active_tool_call["name"]),
                    tool_args=active_tool_call["args"],
                    error=substep_error,
                    state_step=state_step,
                    task_id=task_id,
                )

            # max_iter num surpassed so conclude
            if final_res is None:
                return self.synthesize_fallback(
                    messages=messages,
                    substep_count=substep_count,
                    state_step=state_step,
                    task_id=task_id,
                )

            return str(final_res.content)
        except Exception as e:
            return self.handle_unhandled_exception(
                e=e,
                substep_count=substep_count,
                t_substep_start=t_substep_start,
                state_step=state_step,
                task_id=task_id,
            )

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
    llm_client: ChatOpenAI
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
        super().__init__(subagent_name, tools, description, log_path=log_path, max_iter=max_iter)

        self.llm_client = llm_client
        self.bound_llm = self.llm_client.bind_tools(self.tools)


    @override
    def react(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before running react agent.")
        substep_count = 0
        final_res = None
        messages = list(state)
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        t_substep_start = time.perf_counter()

        try:
            for i in range(self.max_iter):
                substep_count = i + 1
                t_substep_start = time.perf_counter()
                substep_error = None

                # llm decides next action and args
                res = self.bound_llm.invoke(messages)
                if res.usage_metadata:
                    step_tkns_in = res.usage_metadata["input_tokens"]
                    step_tkns_out = res.usage_metadata["output_tokens"]

                # If no tool calls, this is the final natural response directly
                if not res.tool_calls:
                    final_res = res
                    duration = time.perf_counter() - t_substep_start
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

                messages.append(res)

                # tool/function call happens here
                for tool_call in res.tool_calls:
                    name = str(tool_call["name"])
                    tool_call_id = str(tool_call["id"])

                    if name not in self.tools_by_name:
                        substep_error = f"Tool Error: Tool '{name}' not found."
                        messages.append(ToolMessage(content=substep_error, tool_call_id=tool_call_id))
                        continue

                    tool = self.tools_by_name[name]

                    try:
                        observation = tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        substep_error = f"Tool Error ({name}): {str(e)}"
                        messages.append(ToolMessage(content=substep_error, tool_call_id=tool_call_id))

                duration = time.perf_counter() - t_substep_start
                step_response = res.content if res.content else json.dumps(res.tool_calls)
                self.log_step(
                    substep_count=substep_count,
                    llm_tkns_in=step_tkns_in,
                    llm_tkns_out=step_tkns_out,
                    fdm_tkns_in=0,
                    fdm_tkns_out=0,
                    duration=duration,
                    final_response=step_response,
                    error=substep_error,
                    state_step=state_step,
                    task_id=task_id,
                )

            # If max_iter was reached without a natural language conclusion:
            if final_res is None:
                substep_count += 1
                t_substep_start = time.perf_counter()
                messages.append(
                    HumanMessage(
                        content="You have reached the maximum number of tool iterations. "
                        "Do NOT call any more tools. Please synthesize the findings above and "
                        "provide your best answer in plain text based on the observations."
                    )
                )
                final_res = self.llm_client.invoke(messages)
                if final_res.usage_metadata:
                    step_tkns_in = final_res.usage_metadata["input_tokens"]
                    step_tkns_out = final_res.usage_metadata["output_tokens"]
                duration = time.perf_counter() - t_substep_start
                output_text = str(final_res.content)
                self.log_step(
                    substep_count=substep_count,
                    llm_tkns_in=step_tkns_in,
                    llm_tkns_out=step_tkns_out,
                    fdm_tkns_in=0,
                    fdm_tkns_out=0,
                    duration=duration,
                    final_response=output_text,
                    error=None,
                    state_step=state_step,
                    task_id=task_id,
                )
                return output_text

            return str(final_res.content)
        except Exception as e:
            self.log_step(
                substep_count=substep_count,
                llm_tkns_in=0,
                llm_tkns_out=0,
                fdm_tkns_in=0,
                fdm_tkns_out=0,
                duration=time.perf_counter() - t_substep_start,
                error=str(e),
                state_step=state_step,
                task_id=task_id,
            )
            raise

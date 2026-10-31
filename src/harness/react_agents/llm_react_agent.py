import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from harness.llm import get_chat_model
from harness.logger import append_to_csv
from harness.react_agents.base import BaseReactAgent


class LLMReactAgent(BaseReactAgent):
    reasoning: bool
    llm_client: ChatOpenAI
    bound_llm: Runnable[object, object]

    def __init__(
        self,
        subagent_name: str,
        subagent_model: str,
        tools: list[Tool],
        description: str,
        log_path: str = "subagents.csv",
        reasoning: bool = False,
        max_iter: int = 10,
    ) -> None:
        super().__init__(subagent_name, subagent_model, tools, description, log_path, max_iter)

        self.reasoning = reasoning
        self.llm_client = get_chat_model(
            reasoning=self.reasoning,
        )
        self.bound_llm = cast(Runnable[object, object], self.llm_client.bind_tools(self.tools))

    @override
    def react(self, state: list[BaseMessage], state_step: int = 0) -> str:
        t_start = time.perf_counter()
        tot_tokens_in = 0
        tot_tokens_out = 0
        last_error = None
        substep_count = 0
        final_res = None
        messages = list(state)

        try:
            for i in range(self.max_iter):
                substep_count = i + 1
                # llm decides next action and args
                res = self.bound_llm.invoke(messages)
                usage = res.usage_metadata
                tot_tokens_in += usage["input_tokens"]
                tot_tokens_out += usage["output_tokens"]

                # If no tool calls, this is the final natural response directly
                if not res.tool_calls:
                    final_res = res
                    break

                messages.append(res)

                # tool/function call happens here
                for tool_call in res.tool_calls:
                    name = str(tool_call["name"])
                    tool_call_id = str(tool_call["id"])

                    if name not in self.tools_by_name:
                        last_error = f"Tool Error: Tool '{name}' not found."
                        messages.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))
                        continue

                    tool = self.tools_by_name[name]

                    try:
                        observation = tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        last_error = f"Tool Error ({name}): {str(e)}"
                        messages.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))

            # If max_iter was reached without a natural language conclusion:
            if final_res is None:
                messages.append(
                    HumanMessage(
                        content="You have reached the maximum number of tool iterations. "
                        "Do NOT call any more tools. Please synthesize the findings above and "
                        "provide your best answer in plain text based on the observations."
                    )
                )
                final_res = self.llm_client.invoke(messages)
                usage = final_res.usage_metadata
                tot_tokens_in += usage["input_tokens"]
                tot_tokens_out += usage["output_tokens"]

            output_text = str(final_res.content)
            duration = time.perf_counter() - t_start
            self.log_step(substep_count, tot_tokens_in, tot_tokens_out, duration, output_text, last_error, state_step=state_step)
            return output_text
        except Exception as e:
            self.log_step(substep_count, tot_tokens_in, tot_tokens_out, time.perf_counter() - t_start, error=str(e), state_step=state_step)
            raise

    @override
    def log_step(
        self,
        substep_count: int,
        tokens_in: int,
        tokens_out: int,
        duration: float,
        final_response: str | None = None,
        error: str | None = None,
        state_step: int = 0,
    ) -> None:
        log = {
            "node": self.subagent_name,
            "subagent_steps": substep_count,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response": final_response,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.runnables import Runnable

from harness.react_agents.base import BaseReactAgent

class LLMReactAgent(BaseReactAgent):
    reasoning: bool
    llm_client: ChatOllama
    bound_llm: Runnable[object, object]

    def __init__(self, subagent: str, tools: list[Tool], description: str, reasoning: bool = False, max_iter: int = 10) -> None:
        super().__init__(subagent, tools, description, max_iter)

        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.subagent,
            reasoning=self.reasoning,
        )
        self.bound_llm = cast(Runnable[object, object], self.llm_client.bind_tools(self.tools))

    @override
    def react(self, state: list[BaseMessage]) -> str:
        t_start = time.perf_counter()
        tot_tokens_in = 0
        tot_tokens_out = 0
        last_error = None
        step_count = 0
        final_res = None

        try:
            for i in range(self.max_iter):
                step_count = i + 1
                #llm decides next action and args
                res = self.bound_llm.invoke(state)
                tot_tokens_in += res.usage_metadata["input_tokens"]
                tot_tokens_out += res.usage_metadata["output_tokens"]

                if not res.tool_calls:
                    break

                state.append(res)

                #tool/function call happens here
                for tool_call in res.tool_calls:
                    name = str(tool_call["name"])
                    tool = self.tools_by_name.get(name)
                    tool_call_id = str(tool_call.get("id", name))

                    if not tool:
                        last_error = f"Tool Error: Tool '{name}' not found."
                        state.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))
                        continue

                    try:
                        observation = tool.invoke(tool_call["args"])
                        state.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        last_error = f"Tool Error ({name}): {str(e)}"
                        state.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))

            #final response
            final_res = self.llm_client.invoke(state)
            tot_tokens_in += final_res.usage_metadata["input_tokens"]
            tot_tokens_out += final_res.usage_metadata["output_tokens"]

            duration = time.perf_counter() - t_start
            self.log_step(step_count, tot_tokens_in, tot_tokens_out, duration, str(final_res.content), last_error)
            return str(final_res.content)
        except Exception as e:
            self.log_step(step_count, tot_tokens_in, tot_tokens_out, time.perf_counter() - t_start, error=str(e))
            raise

    @override
    def log_step(self, step_count: int, tokens_in: int, tokens_out: int, duration: float, final_response: str | None = None, error: str | None = None) -> None:
        log = {
            "step_count": step_count,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "time": duration,
            "final_response": final_response,
            "error": error,
        }
        print(log)
        with open("subagents.jsonl", "a") as f:
            _ = f.write(json.dumps(log) + "\n")

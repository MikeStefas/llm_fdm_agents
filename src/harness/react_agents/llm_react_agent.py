import json
import time
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool, Tool

from harness.react_agents.base import BaseReactAgent

class LLMReactAgent(BaseReactAgent):

    def __init__(self, subagent: str, tools: list[Tool], description: str, reasoning: bool = False, max_iter: int = 10) -> None:
        super().__init__(subagent, tools, description, max_iter)

        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.subagent,
            reasoning=self.reasoning,
        )
        self.bound_llm = self.llm_client.bind_tools(self.tools)

    def react(self, state: list[BaseMessage]) -> str:
        t_start = time.perf_counter()
        tot_tokens_in = 0
        tot_tokens_out = 0
        last_error = None
        step_count = 0

        try:
            for i in range(self.max_iter):
                step_count = i + 1
                res = self.bound_llm.invoke(state)
                tot_tokens_in += getattr(res, "usage_metadata", {}).get("input_tokens", 0) if hasattr(res, "usage_metadata") and res.usage_metadata else 0
                tot_tokens_out += getattr(res, "usage_metadata", {}).get("output_tokens", 0) if hasattr(res, "usage_metadata") and res.usage_metadata else 0

                if not res.tool_calls:
                    duration = time.perf_counter() - t_start
                    self.log_step(step_count, tot_tokens_in, tot_tokens_out, duration, last_error)
                    return str(res.content)

                state.append(res)

                for tool_call in res.tool_calls:
                    tool = self.tools_by_name.get(tool_call["name"])
                    tool_call_id = tool_call.get("id", tool_call["name"])

                    if not tool:
                        last_error = f"Tool Error: Tool '{tool_call['name']}' not found."
                        state.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))
                        continue

                    try:
                        observation = tool.invoke(tool_call["args"])
                        state.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        last_error = f"Tool Error ({tool.name}): {str(e)}"
                        state.append(ToolMessage(content=last_error, tool_call_id=tool_call_id))

            final_res = self.llm_client.invoke(state)
            tot_tokens_in += getattr(final_res, "usage_metadata", {}).get("input_tokens", 0) if hasattr(final_res, "usage_metadata") and final_res.usage_metadata else 0
            tot_tokens_out += getattr(final_res, "usage_metadata", {}).get("output_tokens", 0) if hasattr(final_res, "usage_metadata") and final_res.usage_metadata else 0
            duration = time.perf_counter() - t_start
            self.log_step(step_count, tot_tokens_in, tot_tokens_out, duration, last_error)
            return str(final_res.content)
        except Exception as e:
            duration = time.perf_counter() - t_start
            last_error = str(e)
            self.log_step(step_count, tot_tokens_in, tot_tokens_out, duration, last_error)
            raise

    def log_step(self, step_count: int, tokens_in: int, tokens_out: int, duration: float, error: str | None = None) -> None:
        log = {
            "step_count": step_count,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "time": duration,
            "error": error,
        }
        print(log)
        with open("subagents.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
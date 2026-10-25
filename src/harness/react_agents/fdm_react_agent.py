import json
import time
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool, Tool
from typesafe_sdk import Choice, TypeSafeClient

from harness.react_agents.base import BaseReactAgent

class FDMReactAgent(BaseReactAgent):

    def __init__(self, base_url: str, model_name: str, tools: list[Tool], description: str = "", reasoning: bool = False, max_iter: int = 10) -> None:
        super().__init__(subagent=model_name, tools=tools, description=description, max_iter=max_iter)
        self.base_url = base_url
        self.model_name = model_name
        self.reasoning = reasoning
        self.fdm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )
        self.criteria = self._get_criteria()

    def _get_criteria(self) -> dict[str, str]:
        criteria = {t.name: (t.description or t.__doc__ or t.name) for t in self.tools}
        if "Finish" not in criteria:
            criteria["Finish"] = "Finish the task and return the final result."
        return criteria

    def react(self, state: list[BaseMessage]) -> str:
        t_start = time.perf_counter()
        tot_tokens_in = 0
        tot_tokens_out = 0
        last_error = None
        step_count = 0
        criteria = self._get_criteria()

        try:
            for i in range(self.max_iter):
                step_count = i + 1
                state_text = self._state_to_text(state)
                res = self.fdm_client.system_one(
                    state=state_text,
                    questions={"next_action": Choice(
                        instructions=f"You are a react agent. Given the state of the task, decide the next action. You can use the following tools: {list(criteria.keys())}",
                        criteria=criteria,
                    )}
                )
                tot_tokens_in += getattr(res.usage, "input_tokens", 0) if hasattr(res, "usage") else 0
                tot_tokens_out += getattr(res.usage, "output_tokens", 0) if hasattr(res, "usage") else 0

                action = res.answers["next_action"].choice

                if action == "Finish":
                    final_res = self.llm_client.invoke(state)
                    tot_tokens_in += getattr(final_res, "usage_metadata", {}).get("input_tokens", 0) if hasattr(final_res, "usage_metadata") and final_res.usage_metadata else 0
                    tot_tokens_out += getattr(final_res, "usage_metadata", {}).get("output_tokens", 0) if hasattr(final_res, "usage_metadata") and final_res.usage_metadata else 0
                    duration = time.perf_counter() - t_start
                    self.log_step(step_count, tot_tokens_in, tot_tokens_out, duration, last_error)
                    return str(final_res.content)

                tool = self.tools_by_name.get(action)
                if not tool:
                    last_error = f"Tool Error: Selected action '{action}' not found."
                    state.append(ToolMessage(content=last_error, tool_call_id=action))
                    continue

                bound_llm = self.llm_client.bind_tools([tool])
                prompt_messages = list(state) + [
                    HumanMessage(content=f"Call the tool '{action}' with the required arguments.")
                ]
                arg_res = bound_llm.invoke(prompt_messages)
                tot_tokens_in += getattr(arg_res, "usage_metadata", {}).get("input_tokens", 0) if hasattr(arg_res, "usage_metadata") and arg_res.usage_metadata else 0
                tot_tokens_out += getattr(arg_res, "usage_metadata", {}).get("output_tokens", 0) if hasattr(arg_res, "usage_metadata") and arg_res.usage_metadata else 0

                state.append(arg_res)

                for tool_call in arg_res.tool_calls:
                    tool_call_id = tool_call.get("id", action)
                    try:
                        observation = tool.invoke(tool_call.get("args", {}))
                        state.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        last_error = f"Tool Error ({action}): {str(e)}"
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
import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

from harness.react_agents.base import BaseReactAgent

class FDMReactAgent(BaseReactAgent):
    base_url: str
    model_name: str
    reasoning: bool
    fdm_client: TypeSafeClient
    llm_client: ChatOllama
    criteria: dict[str, str]

    def __init__(
        self,
        subagent_name: str,
        subagent_model: str,
        tools: list[Tool],
        description: str,
        log_path: str = "subagents.jsonl",
        base_url: str = "http://localhost:9931",
        reasoning: bool = False,
        max_iter: int = 10,
    ) -> None:
        super().__init__(subagent_name, subagent_model, tools, description, log_path, max_iter)
        self.base_url = base_url
        self.reasoning = reasoning
        self.criteria = self._get_criteria()
        self.fdm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )
        self.llm_client = ChatOllama(
            model=self.subagent_model,
            reasoning=self.reasoning,
        )

    @override
    def react(self, state: list[BaseMessage]) -> str:
        t_start = time.perf_counter()
        tot_tokens_in = 0
        tot_tokens_out = 0
        last_error = None
        step_count = 0

        try:
            for i in range(self.max_iter):
                step_count = i + 1

                #fdm decides next action and calc usage
                action, res = self._route(state)

                tot_tokens_in += res.usage.input_tokens
                tot_tokens_out += res.usage.output_tokens

                if action == "Finish":
                    break

                tool = self.tools_by_name.get(action)

                #llm generates arguments for tool call and calc usage
                arg_res = self._generate_args(state, action)
                tot_tokens_in += arg_res.usage_metadata["input_tokens"]
                tot_tokens_out += arg_res.usage_metadata["output_tokens"]
                state.append(arg_res)
                
                #tool/function call is happening
                for tool_call in arg_res.tool_calls:
                    tool_call_id = str(tool_call.get("id", action))
                    try:
                        observation = tool.invoke(tool_call["args"])
                        state.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    except Exception as e:
                        last_error = f"Tool Error ({action}): {str(e)}"
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
        with open(self.log_path, "a") as f:
            _ = f.write(json.dumps(log) + "\n")


    def _route(self, state: list[BaseMessage]) -> tuple[str, SystemOneResponse]:
        """FDM decides next tool call """
        res = self.fdm_client.system_one(
            state=self._state_to_text(state),
            questions={"next_action": Choice(
                instructions=f"You are a react agent. Given the state of the task, decide the next action. You can use the following tools: {list(self.criteria.keys())}",
                criteria=self.criteria,
            )}
        )
        return str(res.answers["next_action"].choice), res

    def _generate_args(self, state: list[BaseMessage], action: str) -> AIMessage:
        """LLM generates arguments for tool call"""
        tool = self.tools_by_name[action]
        bound_llm = cast(ChatOllama, self.llm_client.bind_tools([tool]))
        prompt = list(state) + [HumanMessage(content=f"Call the tool '{action}' with the required arguments.")]
        return cast(AIMessage, bound_llm.invoke(prompt))

    def _get_criteria(self) -> dict[str, str]:
        criteria = {}
        for tool_name in self.tools_by_name.keys():
            tool = self.tools_by_name[tool_name]
            criteria[tool_name] = tool.description or tool_name
        criteria["Finish"] = "Stop when task is complete"
        return criteria


    def _state_to_text(self, state: list[BaseMessage]) -> str:
        lines: list[str] = []
        for msg in state:
            content_str = str(msg.content)
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {content_str}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {content_str}")
            elif isinstance(msg, ToolMessage):
                lines.append(f"Tool Output: {content_str}")
            else:
                msg_type: str = str(getattr(msg, "type", "Message"))
                lines.append(f"{msg_type}: {content_str}")
        return "\n".join(lines)
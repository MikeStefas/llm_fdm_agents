import json
import time
from typing import cast
from typing_extensions import override
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv, get_chat_model
from harness.utils.prompts import (
    FDM_REACT_FINISH_DESCRIPTION,
    FDM_REACT_INSTRUCTIONS_TEMPLATE,
    FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE,
)


class FDMReactAgent(BaseReactAgent):
    base_url: str
    model_name: str
    reasoning: bool
    fdm_client: TypeSafeClient
    llm_client: ChatOpenAI
    criteria: dict[str, str]

    def __init__(
        self,
        subagent_name: str,
        subagent_model: str,
        tools: list[Tool],
        description: str,
        log_path: str = "subagents.csv",
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
        self.llm_client = get_chat_model(
            reasoning=self.reasoning,
        )

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

                # fdm decides next action and calc usage
                action, res = self._route(messages)
                tot_tokens_in += res.usage.input_tokens 
                tot_tokens_out += res.usage.output_tokens 

                # If finish action, generate final response directly
                if action == "Finish":
                    final_res = self.llm_client.invoke(messages)
                    usage = final_res.usage_metadata 
                    tot_tokens_in += usage["input_tokens"]
                    tot_tokens_out += usage["output_tokens"]
                    break

                if action not in self.tools_by_name:
                    last_error = f"Tool Error: Tool '{action}' not found."
                    messages.append(ToolMessage(content=last_error, tool_call_id=action))
                    continue

                tool = self.tools_by_name[action]

                # llm generates arguments for tool call and calc usage
                arg_res = self._generate_args(messages, action)
                usage = arg_res.usage_metadata 
                tot_tokens_in += usage["input_tokens"]
                tot_tokens_out += usage["output_tokens"]

                # If no tool calls, this is the final natural response directly
                if not arg_res.tool_calls:
                    final_res = arg_res
                    break

                messages.append(arg_res)

                # tool/function call happens here
                for tool_call in arg_res.tool_calls:
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


    def _route(self, state: list[BaseMessage]) -> tuple[str, SystemOneResponse]:
        """FDM decides next tool call """
        res = self.fdm_client.system_one(
            state=self.state_to_text(state),
            questions={"next_action": Choice(
                instructions=FDM_REACT_INSTRUCTIONS_TEMPLATE.format(tools=list(self.criteria.keys())),
                criteria=self.criteria,
            )}
        )
        return str(res.answers["next_action"].choice), res

    def _generate_args(self, state: list[BaseMessage], action: str) -> AIMessage:
        """LLM generates arguments for tool call"""
        tool = self.tools_by_name[action]
        bound_llm = self.llm_client.bind_tools([tool])
        prompt = list(state) + [HumanMessage(content=FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE.format(action=action))]
        return cast(AIMessage, bound_llm.invoke(prompt))

    def _get_criteria(self) -> dict[str, str]:
        criteria = {}
        for tool_name, tool in self.tools_by_name.items():
            criteria[tool_name] = tool.description or tool_name
        criteria["Finish"] = FDM_REACT_FINISH_DESCRIPTION
        return criteria

    def _state_to_text(self, state: list[BaseMessage]) -> str:
        return self.state_to_text(state)
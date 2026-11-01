import json
import time
from typing import Optional, cast
from typing_extensions import override
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

from harness.react_agents.base import BaseReactAgent
from harness.utils.prompts import (
    FDM_REACT_FINISH_DESCRIPTION,
    FDM_REACT_INSTRUCTIONS_TEMPLATE,
    FDM_REACT_TOOL_ARGS_PROMPT_TEMPLATE,
)


class FDMReactAgent(BaseReactAgent):
    criteria: dict[str, str]
    fdm_client: TypeSafeClient
    llm_client: ChatOpenAI

    def __init__(
        self,
        subagent_name: str,
        tools: list[Tool],
        description: str,
        fdm_client: TypeSafeClient,
        llm_client: ChatOpenAI,
        log_path: Optional[str] = None,
        max_iter: int = 10,
  
    ) -> None:
        super().__init__(subagent_name, tools, description, log_path=log_path, max_iter=max_iter)

        self.criteria = self._get_criteria()
        self.fdm_client = fdm_client
        self.llm_client = llm_client


    @override
    def react(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before running react agent.")
        substep_count = 0
        final_res = None
        messages = list(state)
        t_substep_start = time.perf_counter()

        try:
            for i in range(self.max_iter):
                substep_count = i + 1
                t_substep_start = time.perf_counter()
                substep_error = None
                step_fdm_in = 0
                step_fdm_out = 0
                step_llm_in = 0
                step_llm_out = 0

                # fdm decides next action and calc usage
                action, res = self._route(messages)
                if res.usage is not None:
                    step_fdm_in += res.usage.input_tokens
                    step_fdm_out += res.usage.output_tokens

                # If finish action, generate final response directly
                if action == "Finish":
                    final_res = self.llm_client.invoke(messages)
                    usage = getattr(final_res, "usage_metadata", None) or {}
                    step_llm_in += usage.get("input_tokens", 0)
                    step_llm_out += usage.get("output_tokens", 0)
                    duration = time.perf_counter() - t_substep_start
                    self.log_step(
                        substep_count=substep_count,
                        llm_tkns_in=step_llm_in,
                        llm_tkns_out=step_llm_out,
                        fdm_tkns_in=step_fdm_in,
                        fdm_tkns_out=step_fdm_out,
                        duration=duration,
                        final_response=str(final_res.content),
                        error=None,
                        state_step=state_step,
                        task_id=task_id,
                    )
                    break

                if action not in self.tools_by_name:
                    substep_error = f"Tool Error: Tool '{action}' not found."
                    messages.append(ToolMessage(content=substep_error, tool_call_id=action))
                    duration = time.perf_counter() - t_substep_start
                    self.log_step(
                        substep_count=substep_count,
                        llm_tkns_in=step_llm_in,
                        llm_tkns_out=step_llm_out,
                        fdm_tkns_in=step_fdm_in,
                        fdm_tkns_out=step_fdm_out,
                        duration=duration,
                        final_response=f"Action: {action}",
                        error=substep_error,
                        state_step=state_step,
                        task_id=task_id,
                    )
                    continue

                tool = self.tools_by_name[action]

                # llm generates arguments for tool call and calc usage
                arg_res = self._generate_args(messages, action)
                usage = getattr(arg_res, "usage_metadata", None) or {}
                step_llm_in += usage.get("input_tokens", 0)
                step_llm_out += usage.get("output_tokens", 0)

                # If no tool calls, this is the final natural response directly
                if not arg_res.tool_calls:
                    final_res = arg_res
                    duration = time.perf_counter() - t_substep_start
                    self.log_step(
                        substep_count=substep_count,
                        llm_tkns_in=step_llm_in,
                        llm_tkns_out=step_llm_out,
                        fdm_tkns_in=step_fdm_in,
                        fdm_tkns_out=step_fdm_out,
                        duration=duration,
                        final_response=str(final_res.content),
                        error=None,
                        state_step=state_step,
                        task_id=task_id,
                    )
                    break

                messages.append(arg_res)

                # tool/function call happens here
                for tool_call in arg_res.tool_calls:
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
                step_response = arg_res.content if arg_res.content else json.dumps(arg_res.tool_calls)
                self.log_step(
                    substep_count=substep_count,
                    llm_tkns_in=step_llm_in,
                    llm_tkns_out=step_llm_out,
                    fdm_tkns_in=step_fdm_in,
                    fdm_tkns_out=step_fdm_out,
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
                usage = getattr(final_res, "usage_metadata", None) or {}
                step_llm_in = usage.get("input_tokens", 0)
                step_llm_out = usage.get("output_tokens", 0)
                duration = time.perf_counter() - t_substep_start
                output_text = str(final_res.content)
                self.log_step(
                    substep_count=substep_count,
                    llm_tkns_in=step_llm_in,
                    llm_tkns_out=step_llm_out,
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

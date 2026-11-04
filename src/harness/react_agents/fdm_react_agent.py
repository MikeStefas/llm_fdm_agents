import json
import time
from typing import Optional, cast
from typing_extensions import override
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
        max_iter: int = 30,
  
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
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
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
                if action.lower() == "finish":
                    final_res = self.llm_client.invoke(messages)
                    if final_res.usage_metadata:
                        step_llm_in += final_res.usage_metadata["input_tokens"]
                        step_llm_out += final_res.usage_metadata["output_tokens"]
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

                tool = self.tools_by_name[action]

                # llm generates arguments for tool call and calc usage
                arg_res = self._generate_args(messages, action)
                if arg_res.usage_metadata:
                    step_llm_in += arg_res.usage_metadata["input_tokens"]
                    step_llm_out += arg_res.usage_metadata["output_tokens"]

                # If no tool calls were produced by the LLM
                if not arg_res.tool_calls:
                    substep_error = f"LLM did not produce arguments for chosen action '{action}'."
                    duration = time.perf_counter() - t_substep_start
                    self.log_step(
                        substep_count=substep_count,
                        llm_tkns_in=step_llm_in,
                        llm_tkns_out=step_llm_out,
                        fdm_tkns_in=step_fdm_in,
                        fdm_tkns_out=step_fdm_out,
                        duration=duration,
                        final_response=str(arg_res.content),
                        error=substep_error,
                        state_step=state_step,
                        task_id=task_id,
                    )
                    messages.append(arg_res)
                    messages.append(
                        HumanMessage(
                            content=f"Error: You did not call the tool '{action}'. You must invoke '{action}' with valid arguments."
                        )
                    )
                    continue

                messages.append(arg_res)

                # Single tool execution per step
                tool_call = arg_res.tool_calls[0]
                name = str(tool_call["name"])
                tool_call_id = str(tool_call["id"])

                if name not in self.tools_by_name:
                    substep_error = f"Tool Error: Tool '{name}' not found."
                    messages.append(ToolMessage(content=substep_error, tool_call_id=tool_call_id))
                else:
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
                if final_res.usage_metadata:
                    step_llm_in = final_res.usage_metadata["input_tokens"]
                    step_llm_out = final_res.usage_metadata["output_tokens"]
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
        action = str(res.answers["next_action"].choice).strip().strip("'\"")
        return action, res

    def _generate_args(self, state: list[BaseMessage], action: str) -> AIMessage:
        """LLM generates arguments for tool call"""
        tool = self.tools_by_name[action]
        bound_llm = self.llm_client.bind_tools([tool], parallel_tool_calls=False)
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

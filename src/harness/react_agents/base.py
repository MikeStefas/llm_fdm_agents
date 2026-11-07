import time
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from harness.utils.prompts import get_subagent_system_prompt, state_to_text
from harness.utils import append_to_csv


class BaseReactAgent(ABC):
    subagent_name: str
    reasoning: bool = False
    log_path: Optional[str]
    tools: list[Tool]
    tools_by_name: dict[str, Tool]
    description: str
    system_prompt: str
    max_iter: int
    llm_client: ChatOpenAI

    def __init__(
        self,
        subagent_name: str,
        tools: list[Tool],
        description: str,
        llm_client: ChatOpenAI,
        log_path: Optional[str] = None,
        max_iter: int = 10,
    ) -> None:
        self.subagent_name = subagent_name
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.description = description
        self.llm_client = llm_client
        self.system_prompt = get_subagent_system_prompt(subagent_name, description)
        self.max_iter = max_iter
        self.log_path = log_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def get_tools(self) -> list[Tool]:
        return self.tools

    @staticmethod
    def state_to_text(state: list[BaseMessage]) -> str:
        return state_to_text(state)

    def prepare_messages(self, state: list[BaseMessage]) -> list[BaseMessage]:
        messages = list(state)
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        return messages

    def execute_tool_call(
        self,
        tool_call: dict,
        last_input: tuple[str, str] | None = None,
    ) -> tuple[ToolMessage, str | None, tuple[str, str], bool]:
        """Executes a tool call with built-in consecutive loop detection.

        Returns:
            (ToolMessage, error_str, current_input, is_repeated)
        """
        import json

        name = str(tool_call["name"])
        tool_call_id = str(tool_call["id"])
        args = tool_call["args"]
        current_input = (name, json.dumps(args, sort_keys=True))

        # Loop  skip if repeated consecutively with same args. qwen with tmp = 0 will just repeat.
        if current_input == last_input:
            warn_msg = (
                f"Error: You just called tool '{name}' with identical arguments {args} "
                f"and received that exact result. Do NOT call '{name}' with these same arguments again. "
                "Either use different arguments, try a different tool, or finish by outputting no tool call."
            )
            return (
                ToolMessage(content=warn_msg, tool_call_id=tool_call_id),
                "Skipped duplicate consecutive tool call.",
                current_input,
                True,
            )

        if name not in self.tools_by_name:
            err = f"Tool Error: Tool '{name}' not found."
            return ToolMessage(content=err, tool_call_id=tool_call_id), err, current_input, False

        tool = self.tools_by_name[name]
        try:
            observation = tool.invoke(args)
            return ToolMessage(content=str(observation), tool_call_id=tool_call_id), None, current_input, False
        except Exception as e:
            err = f"Tool Error ({name}): {str(e)}"
            return ToolMessage(content=err, tool_call_id=tool_call_id), err, current_input, False

    def synthesize_fallback(
        self,
        messages: list[BaseMessage],
        substep_count: int,
        state_step: int = 0,
        task_id: str | None = None,
    ) -> str:
        messages.append(
            HumanMessage(
                content="You have reached the maximum number of tool iterations. "
                "Do NOT call any more tools. Please provide a concise bulleted summary of your findings (max 4-5 bullet points) "
                "in plain text based on the observations above."
            )
        )
        t_api_start = time.perf_counter()
        final_res = self.llm_client.invoke(messages, tool_choice="none")
        duration = time.perf_counter() - t_api_start
        step_tkns_in = 0
        step_tkns_out = 0
        if final_res.usage_metadata:
            step_tkns_in = final_res.usage_metadata["input_tokens"]
            step_tkns_out = final_res.usage_metadata["output_tokens"]
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

    def handle_unhandled_exception(
        self,
        e: Exception,
        substep_count: int,
        t_substep_start: float,
        state_step: int = 0,
        task_id: str | None = None,
    ) -> str:
        duration = time.perf_counter() - t_substep_start
        err_msg = f"Unhandled Subagent Exception: {str(e)}"
        self.log_step(
            substep_count=substep_count,
            llm_tkns_in=0,
            llm_tkns_out=0,
            fdm_tkns_in=0,
            fdm_tkns_out=0,
            duration=duration,
            final_response=None,
            error=err_msg,
            state_step=state_step,
            task_id=task_id,
        )
        return f"Error encountered in subagent '{self.subagent_name}': {str(e)}"

    @abstractmethod
    def react(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        pass

    def log_step(
        self,
        substep_count: int,
        llm_tkns_in: int,
        llm_tkns_out: int,
        fdm_tkns_in: int,
        fdm_tkns_out: int,
        duration: float,
        final_response: str | None = None,
        tool_name: str | None = None,
        tool_args: dict | str | None = None,
        error: str | None = None,
        state_step: int = 0,
        task_id: str | None = None,
    ) -> None:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before logging.")

        log = {
            "task_id": task_id,
            "node": self.subagent_name,
            "subagent_steps": substep_count,
            "state_step": state_step,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "compute_time_seconds": duration,
            "llm_tkns_in": llm_tkns_in,
            "llm_tkns_out": llm_tkns_out,
            "fdm_tkns_in": fdm_tkns_in,
            "fdm_tkns_out": fdm_tkns_out,
            "response": final_response,
            "error": error,
        }
        append_to_csv(self.log_path, log)

    def __call__(self, state: list[BaseMessage], state_step: int = 0, task_id: str | None = None) -> str:
        if self.log_path is None:
            raise ValueError("log_path is not set. Call set_log_path() before running react agent.")
        return self.react(state, state_step=state_step, task_id=task_id)

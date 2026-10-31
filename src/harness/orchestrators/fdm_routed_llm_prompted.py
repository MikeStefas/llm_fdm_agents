import json
import time
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typesafe_sdk import TypeSafeClient, Choice, ChoiceAnswer, SystemOneResponse
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent
from harness.utils import append_to_csv, get_chat_model
from harness.utils.prompts import (
    FDM_ROUTED_LLM_PROMPTED_ROUTING_INSTRUCTIONS,
    FDM_ROUTED_LLM_PROMPTED_SUBAGENT_PROMPT_TEMPLATE,
    FINALIZER_OPTION_DESCRIPTION,
)


class FDMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    fdm_model: str
    base_url: str
    fdm_client: TypeSafeClient
    reasoning: bool
    llm_client: ChatOpenAI
    criteria: dict[str, str]

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        log_path: str = "orchestrator.csv",
        fdm_model: str = "clef-flash",
        fdm_base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents, log_path)
        self.fdm_model = fdm_model
        self.fdm_base_url = fdm_base_url
        self.fdm_client = TypeSafeClient(
            base_url=self.fdm_base_url,
            api_key="local",
        )
        self.reasoning = reasoning
        self.llm_client = get_chat_model(
            reasoning=self.reasoning,
        )

    @override
    def orchestrate(self, state: object) -> dict[str, str]:
        # stage 1: fdm routes and decides next subagent
        t_start_routing = time.perf_counter()

        available_options = dict(self.agent_descriptions)
        available_options["finalizer"] = FINALIZER_OPTION_DESCRIPTION

        state_text = self.state_to_text(state)
        fdm_res = self.fdm_client.system_one(
            state=state_text,
            questions={
                "next_subagent": Choice(
                    instructions=FDM_ROUTED_LLM_PROMPTED_ROUTING_INSTRUCTIONS,
                    criteria=available_options,
                )
            },
        )
        t_end_routing = time.perf_counter()

        chosen_agent = str(fdm_res.answers["next_subagent"].choice).strip().strip("'\"").lower()
        if chosen_agent == "finish":
            chosen_agent = "finalizer"

        allowed = self.subagents_names + ["finalizer"]
        if chosen_agent in allowed:
            error = None
        else:
            error = f"Invalid action: {chosen_agent}. Allowed: {allowed}"

        # stage 2: llm gens txt
        t_start_prompting = time.perf_counter()
        agent_desc = available_options.get(chosen_agent, "")
        prompt_res = self.llm_client.invoke([
            SystemMessage(content=FDM_ROUTED_LLM_PROMPTED_SUBAGENT_PROMPT_TEMPLATE.format(
                chosen_agent=chosen_agent,
                agent_desc=agent_desc,
            )),
            HumanMessage(content=state_text),
        ])
        t_end_prompting = time.perf_counter()
        subagent_prompt = str(prompt_res.content).strip()

        step_val = state.state_step
        self.log_step(
            step_val,
            chosen_agent,
            subagent_prompt,
            fdm_res,
            prompt_res,
            t_end_routing - t_start_routing,
            t_end_prompting - t_start_prompting,
            error,
        )

        return {
            "next_subagent": chosen_agent,
            "subagent_prompt": subagent_prompt,
        }

    def log_step(
        self,
        state_step: int,
        chosen_agent: str,
        subagent_prompt: str,
        fdm_res: SystemOneResponse | None,
        prompt_res: AIMessage | None,
        fdm_duration: float,
        llm_duration: float,
        error: str | None,
    ) -> None:
        ans = fdm_res.answers["next_subagent"]
        confidence = ans.confidence 
        probabilities = ans.probabilities

        fdm_tokens_in = fdm_res.usage.input_tokens
        fdm_tokens_out = fdm_res.usage.output_tokens

        prompt_usage = prompt_res.usage_metadata
        llm_tokens_in = prompt_usage["input_tokens"]
        llm_tokens_out = prompt_usage["output_tokens"]

        tokens_in = fdm_tokens_in + llm_tokens_in
        tokens_out = fdm_tokens_out + llm_tokens_out

        response = {
            "next_subagent": chosen_agent,
            "subagent_prompt": subagent_prompt,
        }

        log = {
            "node": "orchestrator",
            "subagent_steps": None,
            "state_step": state_step,
            "reasoning": self.reasoning,
            "confidence": confidence,
            "probabilities": probabilities,
            "compute_time_seconds": fdm_duration + llm_duration,
            "fdm_time_seconds": fdm_duration,
            "llm_time_seconds": llm_duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response": response,
            "error": error,
        }
        print(log)
        append_to_csv(self.log_path, log)

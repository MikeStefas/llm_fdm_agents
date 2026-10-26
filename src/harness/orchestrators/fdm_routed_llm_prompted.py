import json
import time
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from typesafe_sdk import TypeSafeClient, Choice, ChoiceAnswer, SystemOneResponse
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent


class FDMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    fdm_model: str
    base_url: str
    fdm_client: TypeSafeClient
    reasoning: bool
    llm_client: ChatOllama
    criteria: dict[str, str]

    def __init__(
        self,
        model_name: str,
        subagents: list[BaseReactAgent],
        fdm_model: str = "clef-flash",
        fdm_base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents)
        self.fdm_model = fdm_model
        self.fdm_base_url = fdm_base_url
        self.fdm_client = TypeSafeClient(
            base_url=self.fdm_base_url,
            api_key="local",
        )
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )

    @override
    def orchestrate(self, state: object) -> dict[str, str]:
        #stage 1 fdm routes and decides next subagent
        t_start_routing = time.perf_counter()

        fdm_res = self.fdm_client.system_one(
            state=str(state),
            questions={
                "next_subagent": Choice(
                    instructions="You are the orchestrator of a team of subagents. Given the task state, decide which subagent should act next.",
                    criteria=self.agent_descriptions,
                )
            },
        )
        t_end_routing = time.perf_counter()

        chosen_agent = str(fdm_res.answers["next_subagent"].choice)

        if chosen_agent in self.subagents_names:
            error = None
        else:
            error = f"Invalid action: {chosen_agent}. Allowed: {self.subagents_names}"

        #stage 2 llm gens txt
        t_start_prompting = time.perf_counter()
        agent_desc = self.agent_descriptions.get(chosen_agent, "")
        prompt_res = self.llm_client.invoke([
            SystemMessage(content=f"""
            You are the orchestrator of a team of subagents.
            The subagent '{chosen_agent}' ({agent_desc}) has been selected to act next.
            Write a clear, concise instruction/prompt for '{chosen_agent}' based on the task state.
            """),
            HumanMessage(content=f"""{state}"""),
        ])
        t_end_prompting = time.perf_counter()
        subagent_prompt = str(prompt_res.content).strip()

        self.log_step(
            state,
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
        state_step: object,
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

        llm_tokens_in = prompt_res.usage_metadata["input_tokens"]
        llm_tokens_out = prompt_res.usage_metadata["output_tokens"]

        tokens_in = fdm_tokens_in + llm_tokens_in
        tokens_out = fdm_tokens_out + llm_tokens_out

        log = {
            "orchestrator_type": "FDMRoutedLLMPromptedOrchestrator",
            "model_name": self.model_name,
            "fdm_model": self.fdm_model,
            "reasoning": self.reasoning,
            "state_step": state_step,
            "next_subagent": chosen_agent,
            "confidence": confidence,
            "probabilities": probabilities,
            "subagent_prompt": subagent_prompt,
            "compute_time_seconds": fdm_duration + llm_duration,
            "fdm_time_seconds": fdm_duration,
            "llm_time_seconds": llm_duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            _ = f.write(json.dumps(log) + "\n")

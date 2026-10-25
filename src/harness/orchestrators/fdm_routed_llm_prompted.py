import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from typesafe_sdk import TypeSafeClient, Choice
from harness.orchestrators.base import BaseOrchestrator


class FDMRoutedLLMPromptedOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        model_name: str,
        subagents: list,
        dfm_model: str = "clef-flash",
        base_url: str = "http://localhost:9931",
        reasoning: bool = False,
    ) -> None:
        super().__init__(model_name, subagents)
        self.dfm_model = dfm_model
        self.base_url = base_url
        self.dfm_client = TypeSafeClient(
            base_url=self.base_url,
            api_key="local",
        )
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )
        self.criteria = self.agent_descriptions

    def orchestrate(self, state):
        t_start_routing = time.perf_counter()
        #Route
        fdm_res = self.dfm_client.system_one(
            state=state,
            questions={"next_subagent": Choice(
                instructions="You are the orchestrator of a team of subagents. Given the task state, decide which subagent should act next.",
                criteria=self.criteria,
            )}
        )
        t_end_routing = time.perf_counter()
        chosen_agent = fdm_res.answers["next_subagent"].choice
        error = None if chosen_agent in self.actions else f"Invalid action: {chosen_agent}. Allowed: {self.actions}"

        #Gen
        t_start_prompting = time.perf_counter()
        agent_desc = self.agent_descriptions.get(chosen_agent, "")
        prompt_res = self.llm_client.invoke([
            SystemMessage(content=f"""You are the orchestrator of a team of subagents.
The subagent '{chosen_agent}' ({agent_desc}) has been selected to act next.
Write a clear, concise instruction/prompt for '{chosen_agent}' based on the task state."""),
            HumanMessage(content=f"""{state}""")
        ])
        t_end_prompting = time.perf_counter()
        subagent_prompt = str(prompt_res.content).strip()

        
        self.log_step(state, fdm_res, prompt_res, t_end_routing - t_start_routing, t_end_prompting - t_start_prompting, error)

        return {
            "next_subagent": chosen_agent,
            "subagent_prompt": subagent_prompt,
        }

    def log_step(self, state_step, fdm_res, prompt_res, fdm_duration, llm_duration, error: str | None = None):
        subagent = fdm_res.answers["next_subagent"].choice
        tokens_in = fdm_res.usage.input_tokens + prompt_res.usage_metadata.get("input_tokens", 0)
        tokens_out = fdm_res.usage.output_tokens + prompt_res.usage_metadata.get("output_tokens", 0)
        tokens_tot = (
            fdm_res.usage.input_tokens
            + fdm_res.usage.output_tokens
            + prompt_res.usage_metadata.get("total_tokens", 0)
        )

        log = {
            "timestamp": time.time(),
            "orchestrator_type": "FDMRoutedLLMPromptedOrchestrator",
            "model_name": self.model_name,
            "dfm_model": self.dfm_model,
            "reasoning": self.reasoning,
            "step": state_step,
            "next_subagent": subagent,
            "confidence": fdm_res.answers["next_subagent"].confidence,
            "probabilities": fdm_res.answers["next_subagent"].probabilities,
            "subagent_prompt": str(prompt_res.content).strip(),
            "compute_time_seconds": fdm_duration + llm_duration,
            "fdm_time_seconds": fdm_duration,
            "llm_time_seconds": llm_duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_total": tokens_tot,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

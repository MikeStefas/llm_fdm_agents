import json
import time
from typing_extensions import override
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from harness.orchestrators.base import BaseOrchestrator
from harness.react_agents.base import BaseReactAgent


class LLMOrchestrator(BaseOrchestrator):
    reasoning: bool
    llm_client: ChatOllama

    def __init__(self, model_name: str, subagents: list[BaseReactAgent], reasoning: bool = False) -> None:
        super().__init__(model_name, subagents)
        self.reasoning = reasoning
        self.llm_client = ChatOllama(
            model=self.model_name,
            reasoning=self.reasoning,
        )

    @override
    def orchestrate(self, state: object) -> str:
        
        t_start = time.perf_counter()
        
        res = self.llm_client.invoke([
            SystemMessage(content=f"""
            You are the orchestrator of a team of subagents.
            Given the task state, decide which subagent should act next.

            Available subagents:
            {json.dumps(self.agent_descriptions)}
            
            Reply ONLY with a single word which is the name of the subagent to act next.
            """),
            HumanMessage(content=f"""{state}""")
        ])
        t_end = time.perf_counter()
        
        decision = str(res.content).strip()
        if decision in self.subagents_names:
            error = None
        else:
            error = f"Invalid action: {decision}. Allowed: {self.subagents_names}"

        if res:
            self.log_step(state, decision, res, t_end - t_start, error)
        return decision
    
    def log_step(
        self,
        state_step: object,
        decision: str,
        res: AIMessage,
        duration: float,
        error: str | None,
    ) -> None:

        tokens_in = res.usage_metadata["input_tokens"]
        tokens_out = res.usage_metadata["output_tokens"]

        log= {
            "orchestrator_type": "LLMOrchestrator",
            "model_name": self.model_name,
            "reasoning": self.reasoning,
            "state_step": state_step,
            "next_subagent": decision,
            "compute_time_seconds": duration,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "error": error,
        }
        print(log)
        with open("orchestrator.jsonl", "a") as f:
            _ = f.write(json.dumps(log) + "\n")

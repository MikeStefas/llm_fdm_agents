"""Test script running all 4 orchestrators with simulated subagents."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure src is on sys.path regardless of how python is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from harness.orchestrators import (
    FDMOrchestrator,
    FDMRoutedLLMPromptedOrchestrator,
    LLMOrchestrator,
    LLMRoutedLLMPromptedOrchestrator,
)


def create_subagents():
    """Create simulated subagent objects with name and description."""
    return [
        SimpleNamespace(
            subagent="websearch",
            description="Searches the web for up-to-date information, documentation, and facts.",
        ),
        SimpleNamespace(
            subagent="calculator",
            description="Performs mathematical calculations, evaluations, and numerical computations.",
        ),
        SimpleNamespace(
            subagent="finalizer",
            description="Synthesizes findings, composes the final answer, and concludes the task.",
        ),
    ]


def test_llm_orchestrator(subagents, model_name: str, state: str):
    print("\n--- 1. Testing LLMOrchestrator ---")
    orch = LLMOrchestrator(model_name=model_name, subagents=subagents)
    decision = orch(state)
    print(f"Result -> Chosen Subagent: {decision}")
    assert decision in [a.subagent for a in subagents], f"Unexpected decision: {decision}"
    return decision


def test_llm_routed_llm_prompted_orchestrator(subagents, model_name: str, state: str):
    print("\n--- 2. Testing LLMRoutedLLMPromptedOrchestrator ---")
    orch = LLMRoutedLLMPromptedOrchestrator(model_name=model_name, subagents=subagents)
    decision = orch(state)
    print(f"Result -> Chosen Subagent: {decision['next_subagent']}")
    print(f"Result -> Subagent Prompt: {decision['subagent_prompt']}")
    assert decision["next_subagent"] in [a.subagent for a in subagents], f"Unexpected decision: {decision}"
    return decision


def test_fdm_orchestrator(subagents, fdm_url: str, model_name: str, state: str):
    print("\n--- 3. Testing FDMOrchestrator ---")
    orch = FDMOrchestrator(model_name=model_name, subagents=subagents, base_url=fdm_url)
    decision = orch(state)
    print(f"Result -> Chosen Subagent: {decision}")
    assert decision in [a.subagent for a in subagents], f"Unexpected decision: {decision}"
    return decision


def test_fdm_routed_llm_prompted_orchestrator(subagents, fdm_url: str, fdm_model: str, llm_model: str, state: str):
    print("\n--- 4. Testing FDMRoutedLLMPromptedOrchestrator ---")
    orch = FDMRoutedLLMPromptedOrchestrator(
        model_name=llm_model,
        subagents=subagents,
        dfm_model=fdm_model,
        base_url=fdm_url,
    )
    decision = orch(state)
    print(f"Result -> Chosen Subagent: {decision['next_subagent']}")
    print(f"Result -> Subagent Prompt: {decision['subagent_prompt']}")
    assert decision["next_subagent"] in [a.subagent for a in subagents], f"Unexpected decision: {decision}"
    return decision


def main():
    llm_model = os.getenv("LLM_MODEL", "qwen3.5:2b ")
    fdm_model = os.getenv("FDM_MODEL", "clef-flash")
    fdm_url = os.getenv("FDM_BASE_URL", "http://localhost:9931")

    subagents = create_subagents()
    state = "The user wants to find the current stock price of Apple and calculate a 15% discount on 10 shares."

    print(f"State: {state}")
    print(f"Available Subagents: {[a.subagent for a in subagents]}")

    # Test 1: LLM Pure Router
    try:
        test_llm_orchestrator(subagents, model_name=llm_model, state=state)
    except Exception as e:
        print(f"LLMOrchestrator error: {e}")

    # Test 2: LLM Routed + LLM Prompted
    # try:
    #     test_llm_routed_llm_prompted_orchestrator(subagents, model_name=llm_model, state=state)
    # except Exception as e:
    #     print(f"LLMRoutedLLMPromptedOrchestrator error: {e}")

    # Test 3: FDM Pure Router (requires decision_server running on fdm_url)
    try:
        test_fdm_orchestrator(subagents, fdm_url=fdm_url, model_name=fdm_model, state=state)
    except Exception as e:
        print(f"FDMOrchestrator note/error: {e}")

    # # Test 4: FDM Routed + LLM Prompted
    # try:
    #     test_fdm_routed_llm_prompted_orchestrator(
    #         subagents,
    #         fdm_url=fdm_url,
    #         fdm_model=fdm_model,
    #         llm_model=llm_model,
    #         state=state,
    #     )
    # except Exception as e:
    #     print(f"FDMRoutedLLMPromptedOrchestrator note/error: {e}")


if __name__ == "__main__":
    main()

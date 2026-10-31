import os
from langchain_openai import ChatOpenAI

from harness.config import (
    LLM_MODEL,
    LLM_SEED,
    LLM_TOP_P,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def get_chat_model(
    reasoning: bool = False,
    seed: int | None = LLM_SEED,
    top_p: float | None = LLM_TOP_P,
) -> ChatOpenAI:
    api_key = OPENROUTER_API_KEY
    model = LLM_MODEL
    base_url = OPENROUTER_BASE_URL

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        top_p=top_p,
        seed=seed,
        reasoning_effort="medium" if reasoning else "none",
    )

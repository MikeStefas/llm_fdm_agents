import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typesafe_sdk import TypeSafeClient

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_FDM_BASE_URL = "https://openrouter.ai/api"

DEFAULT_LLM_MODEL = "qwen/qwen3.5-9b"
DEFAULT_FDM_MODEL = "cloudflare/clef-flash"

LOCAL_LLAMACPP_BASE_URL = "http://127.0.0.1:8080/v1"
LOCAL_LLAMACPP_MODEL = "/home/mike/.lmstudio/models/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf"

PINNED_PROVIDER_EXTRA_BODY = {
    "provider": {
        "order": ["SiliconFlow"],
        "allow_fallbacks": False,
    }
}


class DeterministicFDMClient(TypeSafeClient):
    def system_one(self, state: Any, questions: Any) -> Any:
        return super().system_one(
            state=state,
            questions=questions,
            extra_body={"temperature": 0.0, "seed": 42},
        )


local_llm_client = ChatOpenAI(
    model=LOCAL_LLAMACPP_MODEL,
    api_key="not-needed",
    base_url=LOCAL_LLAMACPP_BASE_URL,
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    request_timeout=120.0,
    max_retries=1,
)

llm_client_thinking_off = ChatOpenAI(
    model=DEFAULT_LLM_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    request_timeout=90.0,
    max_retries=2,
    reasoning_effort="none",
    frequency_penalty=0.3,
    presence_penalty=0.3,
    extra_body=PINNED_PROVIDER_EXTRA_BODY,
)

llm_client_thinking_on = ChatOpenAI(
    model=DEFAULT_LLM_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    request_timeout=90.0,
    max_retries=2,
    frequency_penalty=0.3,
    presence_penalty=0.3,
    extra_body=PINNED_PROVIDER_EXTRA_BODY,
)

llm_client = llm_client_thinking_off

fdm_client = DeterministicFDMClient(
    model=DEFAULT_FDM_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_FDM_BASE_URL,
    timeout=90.0,
)

__all__ = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_FDM_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_FDM_MODEL",
    "LOCAL_LLAMACPP_BASE_URL",
    "LOCAL_LLAMACPP_MODEL",
    "PINNED_PROVIDER_EXTRA_BODY",
    "DeterministicFDMClient",
    "llm_client",
    "local_llm_client",
    "llm_client_thinking_off",
    "llm_client_thinking_on",
    "fdm_client",
]

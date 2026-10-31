import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

LLM_MODEL: str = "qwen/qwen-3.5-9b"


FDM_MODEL: str = "clef-flash"
FDM_BASE_URL: str = os.getenv("FDM_BASE_URL")

SUBAGENTS_LOG_PATH: str = "subagents.csv"
ORCHESTRATOR_LOG_PATH: str = "orchestrator.csv"

LLM_SEED: int = 42
LLM_TOP_P: float = 0.01

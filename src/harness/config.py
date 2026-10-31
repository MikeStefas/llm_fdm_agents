import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter Configuration
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# LLM Model (single configuration variable)
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")

# Aliases for backwards compatibility
SUBAGENT_MODEL: str = LLM_MODEL
ORCHESTRATOR_LLM_MODEL: str = LLM_MODEL

# FDM (System-1) Configuration
FDM_MODEL: str = os.getenv("FDM_MODEL", "clef-flash")
ORCHESTRATOR_FDM_MODEL: str = FDM_MODEL
FDM_BASE_URL: str = os.getenv("FDM_BASE_URL", "http://localhost:9931")

# Logging
SUBAGENTS_LOG_PATH: str = "subagents.csv"
ORCHESTRATOR_LOG_PATH: str = "orchestrator.csv"

# Reproducibility settings
LLM_SEED: int = int(os.getenv("LLM_SEED", "42"))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "1.0"))

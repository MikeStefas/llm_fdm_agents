import os
from dotenv import load_dotenv

load_dotenv()

SUBAGENT_MODEL: str = os.getenv("SUBAGENT_MODEL")
SUBAGENTS_LOG_PATH: str = "subagents.json"

ORCHESTRATOR_LLM_MODEL: str = os.getenv("ORCHESTRATOR_LLM_MODEL") or os.getenv("LLM_MODEL")
ORCHESTRATOR_FDM_MODEL: str = os.getenv("ORCHESTRATOR_FDM_MODEL") or os.getenv("FDM_MODEL")
ORCHESTRATOR_LOG_PATH: str = "orchestrator.json"

LLM_MODEL: str = ORCHESTRATOR_LLM_MODEL
FDM_MODEL: str = ORCHESTRATOR_FDM_MODEL

# Service Endpoints
FDM_BASE_URL: str = os.getenv("FDM_BASE_URL")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST")

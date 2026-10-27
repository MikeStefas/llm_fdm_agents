
## Start

1. **Environment Setup**:
   ```bash
   cp example.env .env
   ```
   Ensure local servers (Ollama & FDM) are running and models are available. (Will be fixed once the repo is dockerised)

2. **Run graph experiment**:
   ```bash
   python src/harness/experiments/graphbuilder.py
   ```

---

## Project Structure (`src/harness`)

- **`config.py`**: Loads `.env` variables (models, server endpoints, and log paths).

- **`orchestrators/`**: Multi-agent routing strategies:
  - `base.py`: `BaseOrchestrator` abstract class and decision schemas.
  - `llm_orchestrator.py`: Direct LLM subagent selection.
  - `fdm_orchestrator.py`: FDM (system-1) fast routing.
  - `llm_routed_llm_prompted.py`: LLM routes subagent + LLM generates prompt instructions.
  - `fdm_routed_llm_prompted.py`: FDM routes subagent + LLM generates prompt instructions.
- **`react_agents/`**: ReAct subagent implementations:
  - `base.py`: `BaseReactAgent` base class with tool-execution loop and logging.
  - `llm_react_agent.py`: Tool-calling agent powered by LLM.
  - `fdm_react_agent.py`: Tool-calling agent powered by FDM decisions.
- **`tools/`**: Domain toolkits for subagents.
- **`experiments/`**:
  - `initialised_subagents.py`: Pre-configured subagent instances.
  - `initialised_orchs.py`: Pre-configured orchestrator instances.
  - `states.py`: LangGraph state schemas.
  - `graphbuilder.py`: LangGraph StateGraph wiring orchestrators, subagents.

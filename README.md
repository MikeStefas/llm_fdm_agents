# Getting Started

Follow these steps to run the benchmark on your machine:

### 1. Prerequisites
- **Python >= 3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Docker](https://www.docker.com/)** (daemon must be running)

---

### 2. Configure Environment
Copy the example environment file and add your OpenRouter API key:
```bash
cp example.env .env
```
Open `.env` and set:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

### 3. Build the Docker Sandbox Image
Build the isolated container sandbox used by the code execution and bash tools:
```bash
docker build -t gaia-sandbox -f Dockerfile.sandbox .
```

---

### 4. Install Dependencies
Sync and install the project virtual environment:
```bash
uv sync
```

---

### 5. Run
Execute the benchmark:
```bash
uv run python main.py
```
*(You can select which configuration to run by changing `active_conf_id = 1` through `6` inside `main.py`)*

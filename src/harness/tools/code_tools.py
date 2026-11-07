import os
import subprocess
import warnings

from langchain_core.tools import tool
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=DeprecationWarning)

HARNESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR = os.path.join(HARNESS_ROOT, "data")
TASKS_DIR = os.path.join(HARNESS_ROOT, "tasks")


class PythonCodeInput(BaseModel):
    code: str = Field(description="The Python code to execute in the isolated sandbox.")


class BashCommandInput(BaseModel):
    command: str = Field(description="The bash command to run in the isolated sandbox.")


@tool("python_interpreter", args_schema=PythonCodeInput)
def python_repl(code: str) -> str:
    """Execute Python code safely inside an isolated Docker container with pre-installed scientific libraries.

    Available libraries: pandas, numpy, scipy, sympy, mpmath, pyarrow, fastparquet, openpyxl,
    pypdf, pdfplumber, python-docx, python-pptx, biopython, python-chess, pillow, opencv, pydub, bs4, requests.
    The GAIA data directory is mounted read-only at 'data/' and can be accessed directly.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--net=none",  
        "-m",
        "1024m",  
        "--cpus",
        "2.0",  
        "-v",
        f"{DATA_DIR}:/workspace/data:ro",
        "-v",
        f"{TASKS_DIR}:/workspace/tasks:ro",  
        "-w",
        "/workspace",
        "gaia-sandbox",
        "python",
        "-c",
        code,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = res.stdout if res.returncode == 0 else res.stderr
        return output.strip() or "Code executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Execution timed out after 60 seconds."
    except Exception as e:
        return f"Sandbox execution error: {str(e)}"


@tool("bash_shell", args_schema=BashCommandInput)
def shell_tool(command: str) -> str:
    """Execute a bash command safely inside an isolated Docker container."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "--net=none",
        "-m",
        "1024m",
        "--cpus",
        "2.0",
        "-v",
        f"{DATA_DIR}:/workspace/data:ro",
        "-v",
        f"{TASKS_DIR}:/workspace/tasks:ro",
        "-w",
        "/workspace",
        "gaia-sandbox",
        "bash",
        "-c",
        command,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = res.stdout if res.returncode == 0 else res.stderr
        return output.strip() or "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Sandbox execution error: {str(e)}"


CODE_TOOLS = [
    python_repl,
    shell_tool,
]

# Alias for backwards compatibility
python_ast_repl = python_repl

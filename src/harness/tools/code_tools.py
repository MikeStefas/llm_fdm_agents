import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools.shell.tool import ShellTool
from langchain_experimental.tools.python.tool import (
    PythonAstREPLTool,
    PythonREPLTool,
)

python_ast_repl = PythonAstREPLTool()
python_repl = PythonREPLTool()
shell_tool = ShellTool()

CODE_TOOLS = [
    python_ast_repl,
    python_repl,
    shell_tool,
]

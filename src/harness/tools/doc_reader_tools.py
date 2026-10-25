"""Document reading and extraction tools."""

import os
import tempfile
import warnings
import zipfile

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools.file_management import (
    FileSearchTool,
    ListDirectoryTool,
    ReadFileTool,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from markitdown import MarkItDown

_markitdown_converter = MarkItDown()

read_file_tool = ReadFileTool()
file_search_tool = FileSearchTool()
list_directory_tool = ListDirectoryTool()


@tool
def convert_file_to_markdown_and_read(path: str, task : str) -> str:
    """Converts almost any document, presentation, or spreadsheet (PDF, DOCX, PPTX, XLSX, CSV, HTML, etc.)
    into clean, readable Markdown.

    Args:
        path: Path to the target file.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    try:
        result = _markitdown_converter.convert(path)
        content = result.text_content or ""
        content = content[:12000] if len(content) > 12000 else content

        llm = ChatOllama(model='qwen3.5-2b')

        llm_res = llm.invoke({"messages": task + "\n\n" + content})
        return llm_res.content
    except Exception as e:
        return f"MarkItDown conversion failed for '{path}': {e}"


convert_file_to_markdown = convert_file_to_markdown_and_read



DOC_READER_TOOLS = [
    convert_file_to_markdown_and_read,
    file_search_tool,
    list_directory_tool,
]

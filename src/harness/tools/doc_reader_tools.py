import os
import warnings
import zipfile

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools.file_management import (
    FileSearchTool,
    ListDirectoryTool,
    ReadFileTool,
)
from langchain_core.tools import tool

from harness.utils.markitdown_config import shared_markitdown

_markitdown_converter = shared_markitdown

read_file_tool = ReadFileTool()
file_search_tool = FileSearchTool()
list_directory_tool = ListDirectoryTool()


@tool
def convert_file_to_markdown(path: str) -> str:
    """Converts a document, presentation, or spreadsheet (PDF, DOCX, PPTX, XLSX, CSV, HTML, XML, etc.)
    directly into clean Markdown text and returns it.

    Args:
        path: Path to the target file.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    _, ext = os.path.splitext(path.lower())
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".mp3", ".wav", ".m4a", ".ogg"):
        return (
            f"Error: '{path}' is a multimedia file ({ext}). It cannot be parsed by doc_reader. "
        )
    try:
        result = _markitdown_converter.convert(path)
        content = result.text_content or ""
        return content
    except Exception as e:
        return f"MarkItDown conversion failed for '{path}': {e}"


# Backward compatibility alias
convert_file_to_markdown_and_read = convert_file_to_markdown


@tool
def unzip_file(zip_path: str, extract_to: str | None = None) -> str:
    """Extracts a ZIP archive to a destination directory and returns the list of extracted files.

    Args:
        zip_path: Path to the ZIP file.
        extract_to: Directory to extract into. If omitted, extracts to a directory named after the zip file.
    """
    if not os.path.exists(zip_path):
        return f"Error: Zip file '{zip_path}' not found."
    if not zipfile.is_zipfile(zip_path):
        return f"Error: File '{zip_path}' is not a valid zip archive."

    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]

    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)
            extracted_files = archive.namelist()
        return (
            f"Successfully extracted {len(extracted_files)} files to '{extract_to}':\n"
            + "\n".join(f"- {f}" for f in extracted_files)
        )
    except Exception as e:
        return f"Failed to unzip '{zip_path}': {e}"


DOC_READER_TOOLS = [
    convert_file_to_markdown,
    unzip_file,
    read_file_tool,
    file_search_tool,
    list_directory_tool,
]

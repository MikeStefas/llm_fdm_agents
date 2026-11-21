import os
import shutil
import subprocess
import warnings
import zipfile

import pypdfium2 as pdfium
from langchain_community.tools.file_management import (
    FileSearchTool,
    ListDirectoryTool,
    ReadFileTool,
)
from langchain_core.tools import tool

from harness.utils.markitdown_config import shared_markitdown
from harness.utils.workspace import SHARED_DIR

warnings.filterwarnings("ignore", category=DeprecationWarning)

_markitdown_converter = shared_markitdown

read_file_tool = ReadFileTool()
file_search_tool = FileSearchTool()
list_directory_tool = ListDirectoryTool()


@tool
def convert_file_to_markdown_and_read(path: str) -> str:
    """Converts a document, presentation, or spreadsheet (PDF, DOCX, PPTX, XLSX, CSV, HTML, XML, etc.)
    directly into clean Markdown text and markdown tables.

    IMPORTANT: This tool is STRICTLY for extracting text and tabular data. It CANNOT see images, figures,
    charts, diagrams, visual slide layouts, or cell background styling/colors. For visual understanding of
    pages, slides, or sheets, use 'render_document_to_image'.

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


@tool
def render_document_to_image(file_path: str, page_number: int = 1) -> str:
    """Renders a specific page or slide of a PDF, PPTX presentation, or Excel workbook
    into a high-resolution PNG image in the shared directory for visual understanding
    (e.g. inspecting diagrams, charts, figures, flowcharts, slide layouts, or colors).

    Use this tool whenever visual understanding is needed that raw text extraction cannot provide.

    Args:
        file_path: Path to the target document (.pdf, .pptx, .ppt, .xlsx, .xls, .docx, .doc).
        page_number: 1-indexed page or slide number to render (defaults to 1).

    Returns:
        The file path to the rendered PNG image in the shared directory and page count.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    file_path = os.path.abspath(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if ext == ".pdf":
        pdf_path = file_path
    elif ext in (".pptx", ".ppt", ".xlsx", ".xls", ".docx", ".doc"):
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            file_path,
            "--outdir",
            SHARED_DIR,
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=45,
            )
            if res.returncode != 0:
                return f"Error converting '{file_path}' to PDF: {res.stderr}"
        except Exception as e:
            return f"Error running LibreOffice conversion: {e}"
        pdf_path = os.path.join(SHARED_DIR, f"{base_name}.pdf")
    else:
        return f"Error: Unsupported file format '{ext}' for visual rendering."

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf)
        if page_number < 1 or page_number > total_pages:
            return f"Error: Page number {page_number} is out of range. The document has {total_pages} page(s)."

        page = pdf[page_number - 1]
        image = page.render(scale=2.0).to_pil()

        out_filename = f"{base_name}_page_{page_number}.png"
        out_path = os.path.join(SHARED_DIR, out_filename)
        image.save(out_path)
        os.chmod(out_path, 0o666)

        return (
            f"Successfully rendered page {page_number} of {total_pages} to: {out_path}\n"
            f"Total pages in document: {total_pages}. You can now visually inspect this image."
        )
    except Exception as e:
        return f"Error rendering page {page_number} from '{file_path}': {e}"




@tool
def unzip_file(archive_path: str, extract_to: str | None = None) -> str:
    """Extracts an archive (ZIP, TAR, TAR.GZ, TGZ, TAR.BZ2, TAR.XZ) into a destination directory
    in the shared workspace and returns the list of extracted files.

    Args:
        archive_path: Path to the archive file (.zip, .tar.gz, .tgz, .tar.bz2, .tar).
        extract_to: Directory to extract into. If omitted, extracts into a folder in shared/ named after the archive.
    """
    if not os.path.exists(archive_path):
        return f"Error: Archive file '{archive_path}' not found."

    if extract_to is None:
        base_name = os.path.basename(archive_path)
        for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
            if base_name.lower().endswith(ext):
                base_name = base_name[: -len(ext)]
                break
        else:
            base_name = os.path.splitext(base_name)[0]
        extract_to = os.path.join(SHARED_DIR, base_name)

    os.makedirs(extract_to, exist_ok=True)
    try:
        shutil.unpack_archive(archive_path, extract_to)
        extracted_files = []
        for root, dirs, files in os.walk(extract_to):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o777)
                except Exception:
                    pass
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, extract_to)
                extracted_files.append(rel_p)
                try:
                    os.chmod(full_p, 0o666)
                except Exception:
                    pass

        return (
            f"Successfully extracted {len(extracted_files)} file(s) to '{extract_to}':\n"
            + "\n".join(f"- {f}" for f in extracted_files[:50])
            + (f"\n... and {len(extracted_files) - 50} more files." if len(extracted_files) > 50 else "")
        )
    except Exception as e:
        return f"Failed to unpack archive '{archive_path}': {e}"


unpack_archive = unzip_file


DOC_READER_TOOLS = [
    convert_file_to_markdown_and_read,
    render_document_to_image,
    unzip_file,
    read_file_tool,
    file_search_tool,
    list_directory_tool,
]

__all__ = [
    "DOC_READER_TOOLS",
    "convert_file_to_markdown_and_read",
    "render_document_to_image",
    "unzip_file",
    "unpack_archive",
    "read_file_tool",
    "file_search_tool",
    "list_directory_tool",
]

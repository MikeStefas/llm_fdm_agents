"""Subagent descriptions for harness orchestrators and agents."""

WEB_SEARCH_DESCRIPTION = (
    "Searches the web for up-to-date information, documentation, articles, and Wikipedia entries."
)
BROWSER_DESCRIPTION = (
    "Navigates and interacts with live web pages using a browser, clicks elements, and reads dynamic web content."
)
DOC_READER_DESCRIPTION = (
    "Reads, searches, and extracts content from text documents and spreadsheets (such as PDF, DOCX, XLSX, CSV, TXT, Markdown). "
    "Do NOT use this for images (JPG, PNG) or audio (MP3) files."
)
MEDIA_READER_DESCRIPTION = (
    "Inspects visual images, photos, labels, nutrition facts, and audio files (such as JPG, PNG, MP3) using OCR, vision analysis, and audio transcription. "
    "ALWAYS use this for any image, photo, or audio file."
)
CODE_INTERPRETER_DESCRIPTION = (
    "Executes Python code and shell commands to perform calculations, data processing, and scripting tasks."
)

__all__ = [
    "WEB_SEARCH_DESCRIPTION",
    "BROWSER_DESCRIPTION",
    "DOC_READER_DESCRIPTION",
    "MEDIA_READER_DESCRIPTION",
    "CODE_INTERPRETER_DESCRIPTION",
]

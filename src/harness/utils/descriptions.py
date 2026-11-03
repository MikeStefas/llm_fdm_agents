WEB_SEARCH_DESCRIPTION = (
    "Searches the web and Wikipedia for facts, regulations, standards, quotes, and references. "
    "Use this to look up information needed by the user or to search for answers once media is transcribed."
)
BROWSER_DESCRIPTION = (
    "Navigates and interacts with live web pages using a browser, clicks elements, and reads dynamic web content."
)
DOC_READER_DESCRIPTION = (
    "Reads, searches, and extracts content from text documents and spreadsheets (such as PDF, DOCX, XLSX, CSV, TXT, Markdown). "
    "Do NOT use this for images (JPG, PNG) or audio (MP3) files. "
    "Do NOT call if the document has already been read in the conversation."
)
MEDIA_READER_DESCRIPTION = (
    "Inspects visual images, photos, labels, and audio files (such as JPG, PNG, MP3) using OCR, vision analysis, and audio transcription to extract raw text and speech. "
    "Do NOT call this agent if the media file has ALREADY been transcribed or inspected in the conversation."
)
CODE_INTERPRETER_DESCRIPTION = (
    "Executes Python code and shell commands to perform calculations, solve anagrams, process data, and run algorithms."
)

__all__ = [
    "WEB_SEARCH_DESCRIPTION",
    "BROWSER_DESCRIPTION",
    "DOC_READER_DESCRIPTION",
    "MEDIA_READER_DESCRIPTION",
    "CODE_INTERPRETER_DESCRIPTION",
]

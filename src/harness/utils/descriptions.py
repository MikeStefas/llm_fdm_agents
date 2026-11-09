WEB_SEARCH_DESCRIPTION = (
    "Searches the web and Wikipedia for facts, documentation, citations, and reference data. "
    "Visits and extracts readable text from specific URLs, and downloads files from web links. "
    "Use when external information must be fetched from the internet."
)

DOC_READER_DESCRIPTION = (
    "Reads, searches, and extracts content from text documents and spreadsheets (such as PDF, DOCX, XLSX, CSV, TXT, Markdown). "
    "Explores directory listings, locates files, and unpacks archive files (such as ZIP or TAR). "
    "Use when examining local documents, workspace files, or extracted archives."
)

MEDIA_READER_DESCRIPTION = (
    "Inspects visual images, diagrams, photos, and audio files (such as JPG, PNG, MP3, WAV). "
    "Performs OCR to extract visible text, visual reasoning on image contents, and audio transcription. "
    "Use when image or audio files need to be transcribed, read, or visually analyzed."
)

CODE_INTERPRETER_DESCRIPTION = (
    "Executes Python code and shell commands to perform calculations, programmatic data manipulation (e.g. Pandas/CSV/JSON), and algorithmic problem solving. "
    "Use when numerical computation, data transformations, or custom script execution is required."
)

FINALIZER_DESCRIPTION = (
    "Synthesizes the gathered findings and produces the final answer when all required information has been collected and the task is complete."
)

FINALIZER_OPTION_DESCRIPTION = FINALIZER_DESCRIPTION

__all__ = [
    "WEB_SEARCH_DESCRIPTION",
    "DOC_READER_DESCRIPTION",
    "MEDIA_READER_DESCRIPTION",
    "CODE_INTERPRETER_DESCRIPTION",
    "FINALIZER_DESCRIPTION",
    "FINALIZER_OPTION_DESCRIPTION",
]

import os

from dotenv import load_dotenv

load_dotenv()

from harness.react_agents import FDMReactAgent, LLMReactAgent
from harness.tools import (
    BROWSER_TOOLS,
    CODE_TOOLS,
    DOC_READER_TOOLS,
    MEDIA_READER_TOOLS,
    WEB_SEARCH_TOOLS,
)

MODEL_NAME = os.getenv("SUBAGENT_MODEL") 
FDM_BASE_URL = os.getenv("FDM_BASE_URL") 

WEB_SEARCH_DESCRIPTION = (
    "Searches the web for up-to-date information, documentation, articles, and Wikipedia entries."
)
BROWSER_DESCRIPTION = (
    "Navigates and interacts with live web pages using a browser, clicks elements, and reads dynamic web content."
)
DOC_READER_DESCRIPTION = (
    "Reads, searches, and extracts content from local documents and files (such as PDF, DOCX, TXT, Markdown)."
)
MEDIA_READER_DESCRIPTION = (
    "Extracts text and inspects multimedia files using OCR, image analysis, and audio transcription."
)
CODE_INTERPRETER_DESCRIPTION = (
    "Executes Python code and shell commands to perform calculations, data processing, and scripting tasks."
)

llm_web_search_subagent = LLMReactAgent(
    subagent_name="web_search",
    subagent_model=MODEL_NAME,
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
)

llm_browser_subagent = LLMReactAgent(
    subagent_name="browser",
    subagent_model=MODEL_NAME,
    tools=BROWSER_TOOLS,
    description=BROWSER_DESCRIPTION,
)

llm_doc_reader_subagent = LLMReactAgent(
    subagent_name="doc_reader",
    subagent_model=MODEL_NAME,
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
)

llm_media_reader_subagent = LLMReactAgent(
    subagent_name="media_reader",
    subagent_model=MODEL_NAME,
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
)

llm_code_interpreter_subagent = LLMReactAgent(
    subagent_name="code_interpreter",
    subagent_model=MODEL_NAME,
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
)
llm_code_subagent = llm_code_interpreter_subagent

LLM_SUBAGENTS = [
    llm_web_search_subagent,
    llm_browser_subagent,
    llm_doc_reader_subagent,
    llm_media_reader_subagent,
    llm_code_interpreter_subagent,
]

fdm_web_search_subagent = FDMReactAgent(
    subagent_name="web_search",
    subagent_model=MODEL_NAME,
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    base_url=FDM_BASE_URL,
)

fdm_browser_subagent = FDMReactAgent(
    subagent_name="browser",
    subagent_model=MODEL_NAME,
    tools=BROWSER_TOOLS,
    description=BROWSER_DESCRIPTION,
    base_url=FDM_BASE_URL,
)

fdm_doc_reader_subagent = FDMReactAgent(
    subagent_name="doc_reader",
    subagent_model=MODEL_NAME,
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    base_url=FDM_BASE_URL,
)

fdm_media_reader_subagent = FDMReactAgent(
    subagent_name="media_reader",
    subagent_model=MODEL_NAME,
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    base_url=FDM_BASE_URL,
)

fdm_code_interpreter_subagent = FDMReactAgent(
    subagent_name="code_interpreter",
    subagent_model=MODEL_NAME,
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    base_url=FDM_BASE_URL,
)
fdm_code_subagent = fdm_code_interpreter_subagent

FDM_SUBAGENTS = [
    fdm_web_search_subagent,
    fdm_browser_subagent,
    fdm_doc_reader_subagent,
    fdm_media_reader_subagent,
    fdm_code_interpreter_subagent,
]

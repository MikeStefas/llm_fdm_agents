from config import (
    fdm_client,
    llm_client_thinking_off,
    llm_client_thinking_on,
)
from harness.react_agents.fdm_react_agent import FDMReactAgent
from harness.react_agents.llm_react_agent import LLMReactAgent
from harness.tools import (
    CODE_TOOLS,
    DOC_READER_TOOLS,
    MEDIA_READER_TOOLS,
    WEB_SEARCH_TOOLS,
)
from harness.utils import (
    CODE_INTERPRETER_DESCRIPTION,
    DOC_READER_DESCRIPTION,
    MEDIA_READER_DESCRIPTION,
    WEB_SEARCH_DESCRIPTION,
)

#LLM THINKING OFF
llm_web_search_thinking_off = LLMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    llm_client=llm_client_thinking_off,
)

llm_doc_reader_thinking_off = LLMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    llm_client=llm_client_thinking_off,
)

llm_media_reader_thinking_off = LLMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    llm_client=llm_client_thinking_off,
)

llm_code_interpreter_thinking_off = LLMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    llm_client=llm_client_thinking_off,
)

llm_subagents_thinking_off = [
    llm_web_search_thinking_off,
    llm_doc_reader_thinking_off,
    llm_media_reader_thinking_off,
    llm_code_interpreter_thinking_off,
]

#LLM THINKING ON
llm_web_search_thinking_on = LLMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    llm_client=llm_client_thinking_on,
)

llm_doc_reader_thinking_on = LLMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    llm_client=llm_client_thinking_on,
)

llm_media_reader_thinking_on = LLMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    llm_client=llm_client_thinking_on,
)

llm_code_interpreter_thinking_on = LLMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    llm_client=llm_client_thinking_on,
)

llm_subagents_thinking_on = [
    llm_web_search_thinking_on,
    llm_doc_reader_thinking_on,
    llm_media_reader_thinking_on,
    llm_code_interpreter_thinking_on,
]

#FDM THINKING OFF
fdm_web_search_thinking_off = FDMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_off,
)

fdm_doc_reader_thinking_off = FDMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_off,
)

fdm_media_reader_thinking_off = FDMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_off,
)

fdm_code_interpreter_thinking_off = FDMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_off,
)

fdm_subagents_thinking_off = [
    fdm_web_search_thinking_off,
    fdm_doc_reader_thinking_off,
    fdm_media_reader_thinking_off,
    fdm_code_interpreter_thinking_off,
]

#FDM THINKING ON
fdm_web_search_thinking_on = FDMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_on,
)

fdm_doc_reader_thinking_on = FDMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_on,
)

fdm_media_reader_thinking_on = FDMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_on,
)

fdm_code_interpreter_thinking_on = FDMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client_thinking_on,
)

fdm_subagents_thinking_on = [
    fdm_web_search_thinking_on,
    fdm_doc_reader_thinking_on,
    fdm_media_reader_thinking_on,
    fdm_code_interpreter_thinking_on,
]


SUBAGENTS_BY_CONFIG = {
    "llm_thinking_off": llm_subagents_thinking_off,
    "llm_thinking_on": llm_subagents_thinking_on,
    "fdm_thinking_off": fdm_subagents_thinking_off,
    "fdm_thinking_on": fdm_subagents_thinking_on,
}


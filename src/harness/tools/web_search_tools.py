import os
import tempfile
import warnings
import urllib.request
import requests
import wikipedia
from ddg import Duckduckgo
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool

from harness.utils.markitdown_config import shared_markitdown
from harness.utils.vcr_config import my_vcr

warnings.filterwarnings("ignore", category=DeprecationWarning)

wikipedia.set_user_agent("GAIABenchmarkBot/1.0 (mike.research@gmail.com)")

_ddg_client = Duckduckgo()
_markitdown_web_converter = shared_markitdown

_raw_wiki_run = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
)


@tool
@my_vcr.use_cassette("duckduckgo_search.yaml")
def duckduckgo_search(query: str) -> str:
    """Searches DuckDuckGo for the query and returns titles, snippets, and page URLs."""
    res = _ddg_client.search(query)
    if not res.get("success") or not res.get("data"):
        return f"No results found for '{query}'."

    formatted = []
    for item in res["data"][:5]:
        title = item["title"]
        url = item["url"]
        desc = item["description"]
        formatted.append(f"Title: {title}\nSnippet: {desc}\nLink: {url}")

    return "\n\n".join(formatted)


@tool
@my_vcr.use_cassette("wikipedia.yaml")
def wikipedia_tool(query: str) -> str:
    """Searches Wikipedia for the given query and returns page summaries."""
    return _raw_wiki_run.invoke(query)


@tool
@my_vcr.use_cassette("webpage_to_markdown.yaml")
def readable_webpage(url: str) -> str:
    """Fetches a webpage URL and converts its content into clean, readable Markdown text and tables."""
    try:
        clean_url = url.strip()
        result = _markitdown_web_converter.convert(clean_url)
        content = result.text_content or ""
        return content[:12000] if len(content) > 12000 else content
    except Exception as e:
        return f"Error converting webpage to markdown: {e}"


convert_webpage_to_markdown = readable_webpage
duckduckgo_search_run = duckduckgo_search


@tool
@my_vcr.use_cassette("file_downloads.yaml")
def file_from_url(url: str) -> str:
    """Downloads a file from a URL to a local temporary file and returns its path."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        _, ext = os.path.splitext(url.split("?")[0])
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp") as tmp:
            tmp.write(response.content)
            return tmp.name
    except Exception as e:
        return f"Error downloading file: {e}"


WEB_SEARCH_TOOLS = [
    duckduckgo_search,
    wikipedia_tool,
    readable_webpage,
    file_from_url,
]

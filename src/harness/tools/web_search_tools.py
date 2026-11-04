import os
import tempfile
import warnings
import urllib.request
import requests
import vcr
import wikipedia
from ddg import Duckduckgo
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool
from markitdown import MarkItDown

warnings.filterwarnings("ignore", category=DeprecationWarning)

wikipedia.set_user_agent("GAIABenchmarkBot/1.0 (mike.research@gmail.com)")

CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "cache",
)
CASSETTE_DIR = os.path.join(CACHE_ROOT, "vcr_cassettes")
os.makedirs(CASSETTE_DIR, exist_ok=True)

# VCR instance with recording decorator
my_vcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode="new_episodes",
    match_on=["method", "scheme", "host", "port", "path", "query"],
    filter_headers=["authorization", "user-agent"],
)

_ddg_client = Duckduckgo()

_web_session = requests.Session()
_web_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})
_markitdown_web_converter = MarkItDown(requests_session=_web_session)

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



WEB_SEARCH_TOOLS = [
    duckduckgo_search,
    wikipedia_tool,
    readable_webpage,
]

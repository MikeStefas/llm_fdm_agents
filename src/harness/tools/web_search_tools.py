"""Web search and informational retrieval tools."""

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from bs4 import BeautifulSoup
from langchain_community.tools.ddg_search.tool import (
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool
import requests
import wikipedia

wikipedia.set_user_agent("GAIABenchmarkBot/1.0 (mike.research@gmail.com)")


ddg_api_wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
duckduckgo_search_run = DuckDuckGoSearchRun(api_wrapper=ddg_api_wrapper)
duckduckgo_search_results = DuckDuckGoSearchResults(api_wrapper=ddg_api_wrapper)
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
)




@tool
def readable_webpage(url: str) -> str:
    """Fetches a webpage and extracts clean, readable main body text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
            tag.decompose()

        candidates = []
        selectors = [
            "article", "main", "div#content", "div#main", "div.article", "section", "div.content", "div.post"
        ]
        for sel in selectors:
            for el in soup.select(sel):
                txt = el.get_text(" ", strip=True)
                if txt and len(txt) > 200:
                    candidates.append((len(txt), txt))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1][:8000]
        return soup.get_text(" ", strip=True)[:8000]
    except Exception as e:
        return f"Error extracting readable webpage: {e}"


@tool
def file_from_url(url: str) -> str:
    """Downloads a file from a URL to a local temporary file and returns its path."""
    import tempfile

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
    t
    for t in [
        duckduckgo_search_run,
        wikipedia_tool,
        readable_webpage,
        file_from_url,
    ]
    if t is not None
]

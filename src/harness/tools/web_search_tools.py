"""Web search and informational retrieval tools."""

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from bs4 import BeautifulSoup
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_community.tools.ddg_search.tool import (
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool
import arxiv
import requests
import wikipedia

wikipedia.set_user_agent("GAIABenchmarkBot/1.0 (mike.research@gmail.com)")

duckduckgo_search_run = DuckDuckGoSearchRun()
duckduckgo_search_results = DuckDuckGoSearchResults()
_arxiv_client = arxiv.Client(page_size=3, delay_seconds=1.0, num_retries=3)

@tool("arxiv")
def arxiv_search_tool(query: str) -> str:
    """Search academic research papers on arXiv.
    
    Args:
        query: Search term, title, authors, or arXiv ID.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(_arxiv_client.results(search))
        if not results:
            return "No arXiv results found."

        entries = []
        for r in results:
            authors = ", ".join(a.name for a in r.authors[:5])
            published = r.published.strftime("%Y-%m-%d") if r.published else "N/A"
            entries.append(
                f"Title: {r.title}\n"
                f"Published: {published}\n"
                f"Authors: {authors}\n"
                f"Summary: {r.summary[:800]}...\n"
                f"Link: {r.entry_id}"
            )
        return "\n\n".join(entries)
    except Exception as e:
        return f"Error searching arXiv: {e}"

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
        arxiv_search_tool,
        wikipedia_tool,
        readable_webpage,
        file_from_url,
    ]
    if t is not None
]

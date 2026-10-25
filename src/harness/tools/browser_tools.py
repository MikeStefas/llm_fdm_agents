
import warnings


from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,
)


def get_async_playwright_tools(headless: bool = True):
    """Instantiate and return the PlayWright browser tools with an async browser."""
    browser = create_async_playwright_browser(headless=headless)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools()


BROWSER_TOOLS = get_async_playwright_tools(headless=True)

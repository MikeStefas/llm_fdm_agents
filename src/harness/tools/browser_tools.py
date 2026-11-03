
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)


def get_playwright_tools(headless: bool = True):
    """Instantiate and return the PlayWright browser tools with a synchronous browser."""
    browser = create_sync_playwright_browser(headless=headless)
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
    return toolkit.get_tools()


get_async_playwright_tools = get_playwright_tools
BROWSER_TOOLS = get_playwright_tools(headless=True)

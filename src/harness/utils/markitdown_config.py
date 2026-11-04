import requests
from markitdown import MarkItDown

_web_session = requests.Session()
_web_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

# Single shared MarkItDown converter for files and web URLs
shared_markitdown = MarkItDown(requests_session=_web_session)

__all__ = ["shared_markitdown"]


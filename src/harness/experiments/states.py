from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel


class MessageState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    task_id: Optional[str] = None


class MessagePromptState(BaseModel):
    prompt: str | None = None
    messages: Annotated[list[AnyMessage], add_messages]
    task_id: Optional[str] = None

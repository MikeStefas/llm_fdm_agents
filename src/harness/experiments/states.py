from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel


class MessageState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


class MessagePromptState(BaseModel):
    prompt: Annotated[str, add_messages]
    messages: Annotated[list[AnyMessage], add_messages]

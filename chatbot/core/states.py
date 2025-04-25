from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence


class CustomState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chunks_retrieved: str
    allergy_result: str
    digestive_result: str
    vision_loss_result: str    

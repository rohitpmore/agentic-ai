"""
Agent State management for the multi-agent workflow
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    State structure for the multi-agent workflow.
    
    Attributes:
        messages: Sequence of messages that get accumulated throughout the workflow
    """
    messages: Annotated[Sequence[BaseMessage], operator.add] 
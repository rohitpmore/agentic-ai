"""
Pydantic parsers for structured output parsing
"""

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class TopicSelectionParser(BaseModel):
    """
    Parser for topic classification in the supervisor node.
    
    Attributes:
        Topic: The selected topic category
        Reasoning: Explanation for the classification decision
    """
    Topic: str = Field(description="selected topic from: USA Economy, General Knowledge, Real-time/Current Events")
    Reasoning: str = Field(description="reasoning behind the selected topic")


def get_topic_parser() -> PydanticOutputParser:
    """
    Get a configured topic selection parser.
    
    Returns:
        PydanticOutputParser: Configured parser for topic selection
    """
    return PydanticOutputParser(pydantic_object=TopicSelectionParser) 
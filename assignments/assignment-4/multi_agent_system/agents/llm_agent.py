"""
LLM agent for general knowledge questions
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from ..core.state import AgentState
from config import config


class LLMAgent:
    """
    LLM agent for general knowledge questions.
    
    This agent handles general questions that don't require real-time data
    or specialized knowledge bases, using the base language model capabilities.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI):
        """
        Initialize the LLM agent.
        
        Args:
            model: Google Gemini model instance
        """
        self.model = model
        self.output_parser = StrOutputParser()
    
    def answer_question(self, question: str) -> str:
        """
        Answer a general knowledge question using the LLM.
        
        Args:
            question: The question to answer
            
        Returns:
            str: The generated answer
        """
        try:
            print(f"🤖 LLM Node (General Knowledge) processing: {question}")
            
            # Create a complete query with clear instructions
            complete_query = f"Answer the following general knowledge question clearly and concisely: {question}"
            
            response = self.model.invoke(complete_query)
            print("✅ LLM Response generated")
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"LLM processing failed: {str(e)}"
            print(f"❌ LLM Error: {error_msg}")
            return f"I'm sorry, I couldn't process your question due to a technical issue: {error_msg}"
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the agent state and generate a response.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: Updated state with response
        """
        question = state["messages"][0]
        response = self.answer_question(str(question))
        return {"messages": [response]}
    
    def generate_explanation(self, topic: str, detail_level: str = "medium") -> str:
        """
        Generate an explanation for a given topic.
        
        Args:
            topic: The topic to explain
            detail_level: Level of detail ("simple", "medium", "detailed")
            
        Returns:
            str: Generated explanation
        """
        detail_instructions = {
            "simple": "Explain this in simple terms suitable for a beginner.",
            "medium": "Provide a balanced explanation with key concepts.",
            "detailed": "Give a comprehensive explanation with examples and context."
        }
        
        instruction = detail_instructions.get(detail_level, detail_instructions["medium"])
        query = f"{instruction} Topic: {topic}"
        
        return self.answer_question(query) 
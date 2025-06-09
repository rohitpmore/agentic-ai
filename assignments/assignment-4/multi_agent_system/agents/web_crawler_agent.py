"""
Web crawler agent for real-time/current events using web search
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.state import AgentState
from ..utils.web_search import WebSearchManager
from config import config


class WebCrawlerAgent:
    """
    Web crawler agent for real-time/current events questions.
    
    This agent specializes in answering questions that require up-to-date information
    by performing web searches and synthesizing the results.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI, web_search_manager: WebSearchManager):
        """
        Initialize the web crawler agent.
        
        Args:
            model: Google Gemini model instance
            web_search_manager: Web search manager for performing searches
        """
        self.model = model
        self.web_search_manager = web_search_manager
    
    def answer_question(self, question: str) -> str:
        """
        Answer a real-time/current events question using web search.
        
        Args:
            question: The question to answer
            
        Returns:
            str: The generated answer based on web search results
        """
        try:
            print(f"🌐 Web Crawler Node (Real-time/Current Events) processing: {question}")
            
            # Perform web search to get current information
            search_results = self.web_search_manager.perform_search(question, max_results=3)
            
            # Generate a concise answer based on search results
            query = f"""Based on the following web search results, provide a concise and accurate answer to the question: "{question}"
            
            Search Results:
            {search_results}
            
            Please provide a clear, factual answer based on the search results above."""
            
            response = self.model.invoke(query)
            print("✅ Web Crawler Response generated")
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"Web crawler processing failed: {str(e)}"
            print(f"❌ Web Crawler Error: {error_msg}")
            return f"I'm sorry, I couldn't get current information for your question due to a technical issue: {error_msg}"
    
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
    
    def get_latest_news(self, topic: str, max_results: int = 5) -> str:
        """
        Get latest news about a specific topic.
        
        Args:
            topic: The topic to search for news
            max_results: Maximum number of news results
            
        Returns:
            str: Formatted news results
        """
        news_query = f"latest news about {topic}"
        return self.web_search_manager.perform_search(news_query, max_results)
    
    def get_current_data(self, data_type: str, entity: str) -> str:
        """
        Get current data for a specific entity.
        
        Args:
            data_type: Type of data (e.g., "price", "value", "status")
            entity: The entity to get data for
            
        Returns:
            str: Current data information
        """
        query = f"current {data_type} of {entity}"
        return self.web_search_manager.perform_search(query, max_results=3) 
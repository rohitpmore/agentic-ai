"""
Web search utilities using Tavily API
"""

from typing import Dict, Any, Optional
from tavily import TavilyClient
from config import config


class WebSearchManager:
    """
    Manages web search operations using Tavily API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search manager.
        
        Args:
            api_key: Tavily API key (defaults to config value)
        """
        self.api_key = api_key or config.tavily_api_key
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = config.max_search_results
    
    def search(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform web search using Tavily.
        
        Args:
            query: Search query
            max_results: Maximum number of results (defaults to config value)
            
        Returns:
            Dict[str, Any]: Search response from Tavily
            
        Raises:
            Exception: If search fails
        """
        max_results = max_results or self.max_results
        
        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,
                include_images=False
            )
            return response
        except Exception as e:
            raise Exception(f"Web search failed for query '{query}': {str(e)}")
    
    def format_search_results(self, query: str, response: Dict[str, Any]) -> str:
        """
        Format search results for AI agent consumption.
        
        Args:
            query: Original search query
            response: Tavily search response
            
        Returns:
            str: Formatted search results
        """
        formatted_results = f"Web Search Results for: '{query}'\n\n"
        
        # Add the AI-generated answer if available
        if response.get('answer'):
            formatted_results += f"Summary Answer: {response['answer']}\n\n"
        
        # Add individual search results
        if response.get('results'):
            formatted_results += "Detailed Results:\n"
            for i, result in enumerate(response['results'], 1):
                formatted_results += f"{i}. {result['title']}\n"
                formatted_results += f"   Source: {result['url']}\n"
                formatted_results += f"   Content: {result['content'][:300]}...\n\n"
        else:
            formatted_results += "No detailed results found.\n"
        
        return formatted_results
    
    def perform_search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Perform search and return formatted results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            str: Formatted search results or error message
        """
        try:
            response = self.search(query, max_results)
            return self.format_search_results(query, response)
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            print(f"Error in web search: {error_msg}")
            return f"Unable to perform web search for '{query}'. Error: {error_msg}"
    
    def test_connection(self) -> bool:
        """
        Test the Tavily API connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_query = "test query"
            response = self.client.search(
                query=test_query,
                search_depth="basic",
                max_results=1,
                include_answer=True,
                include_raw_content=False
            )
            print("✅ Tavily API connection successful!")
            return True
        except Exception as e:
            print(f"❌ Tavily API connection failed: {str(e)}")
            return False 
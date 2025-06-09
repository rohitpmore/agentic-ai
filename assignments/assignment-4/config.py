"""
Configuration management for the multi-agent system
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """
    Configuration class for the multi-agent system.
    
    Attributes:
        gemini_api_key: Google Gemini API key
        tavily_api_key: Tavily search API key
        data_directory: Directory containing knowledge base files
        faiss_db_directory: Directory for FAISS vector database
        embedding_model: HuggingFace embedding model name
        chunk_size: Text splitter chunk size
        chunk_overlap: Text splitter chunk overlap
        max_search_results: Maximum web search results
        max_retries: Maximum retry attempts for failed validations
    """
    gemini_api_key: str
    tavily_api_key: str
    data_directory: str = "./data"
    faiss_db_directory: str = "./faiss_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 200
    chunk_overlap: int = 50
    max_search_results: int = 5
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.
        
        Returns:
            Config: Configuration instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
            
        return cls(
            gemini_api_key=gemini_key,
            tavily_api_key=tavily_key,
            data_directory=os.getenv("DATA_DIRECTORY", "./data"),
            faiss_db_directory=os.getenv("FAISS_DB_DIRECTORY", "./faiss_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "5")),
            max_retries=int(os.getenv("MAX_RETRIES", "3"))
        )


# Global configuration instance
config = Config.from_env() 
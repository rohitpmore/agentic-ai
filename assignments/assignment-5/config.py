"""
Configuration management for the Travel Agent System
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
    Configuration class for the Travel Agent System.
    
    Attributes:
        gemini_api_key: Google Gemini API key for LLM operations
        openweather_api_key: OpenWeatherMap API key for weather data
        foursquare_api_key: Foursquare Places API key for attractions/restaurants
        exchangerate_api_key: ExchangeRate-API key for currency conversion
        max_search_results: Maximum search results per API call
        max_retries: Maximum retry attempts for failed API calls
        request_timeout: Timeout for API requests in seconds
        cache_duration: Cache duration for API responses in minutes
    """
    gemini_api_key: str
    openweather_api_key: str
    foursquare_api_key: str
    exchangerate_api_key: str
    max_search_results: int = 10
    max_retries: int = 3
    request_timeout: int = 30
    cache_duration: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.
        
        Returns:
            Config: Configuration instance
            
        Note:
            Uses fallback values for missing API keys to allow system testing
        """
        gemini_key = os.getenv("GEMINI_API_KEY", "fallback_gemini_key")
        openweather_key = os.getenv("OPENWEATHER_API_KEY", "fallback_weather_key") 
        foursquare_key = os.getenv("FOURSQUARE_API_KEY", "fallback_foursquare_key")
        exchangerate_key = os.getenv("EXCHANGERATE_API_KEY", "free_api_key")
        
        return cls(
            gemini_api_key=gemini_key,
            openweather_api_key=openweather_key,
            foursquare_api_key=foursquare_key,
            exchangerate_api_key=exchangerate_key,
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "10")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            cache_duration=int(os.getenv("CACHE_DURATION", "60"))
        )


# Global configuration instance
config = Config.from_env()
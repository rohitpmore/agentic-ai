"""
API Client utilities for external services
"""

import requests
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from config import config


@dataclass
class APIResponse:
    """Standard API response wrapper"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class BaseAPIClient:
    """Base API client with rate limiting and error handling"""
    
    def __init__(self, base_url: str, api_key: str, rate_limit_delay: float = 1.0):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Make HTTP request with error handling"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(
                url, 
                params=params,
                timeout=config.request_timeout,
                headers={"User-Agent": "TravelAgent/1.0"}
            )
            
            if response.status_code == 200:
                return APIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code
                )
            else:
                return APIResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code
                )
                
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )


class OpenWeatherMapClient(BaseAPIClient):
    """OpenWeatherMap API client for weather data"""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.openweathermap.org/data/2.5",
            api_key=config.openweather_api_key,
            rate_limit_delay=1.0  # Free tier: 60 calls/minute
        )
    
    def get_current_weather(self, city: str) -> APIResponse:
        """Get current weather for a city"""
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        return self._make_request("weather", params)
    
    def get_weather_forecast(self, city: str, days: int = 5) -> APIResponse:
        """Get weather forecast for a city"""
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
        }
        return self._make_request("forecast", params)
    
    def test_connection(self) -> bool:
        """Test API connection"""
        response = self.get_current_weather("London")
        return response.success


class FoursquareClient(BaseAPIClient):
    """Foursquare Places API client for attractions and venues"""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.foursquare.com/v3/places",
            api_key=config.foursquare_api_key,
            rate_limit_delay=0.1  # Free tier: generous limits
        )
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """Override to add API key to headers"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {
                "Authorization": self.api_key,
                "Accept": "application/json"
            }
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=config.request_timeout
            )
            
            if response.status_code == 200:
                return APIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code
                )
            else:
                return APIResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code
                )
                
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )
    
    def search_places(self, query: str, near: str, categories: str = None) -> APIResponse:
        """Search for places near a location"""
        params = {
            "query": query,
            "near": near,
            "limit": config.max_search_results
        }
        if categories:
            params["categories"] = categories
            
        return self._make_request("search", params)
    
    def test_connection(self) -> bool:
        """Test API connection"""
        response = self.search_places("restaurant", "New York")
        return response.success


class ExchangeRateClient(BaseAPIClient):
    """ExchangeRate-API client for currency conversion"""
    
    def __init__(self):
        super().__init__(
            base_url="https://v6.exchangerate-api.com/v6",
            api_key=config.exchangerate_api_key,
            rate_limit_delay=1.0  # Free tier: 1500 requests/month
        )
    
    def get_exchange_rates(self, base_currency: str = "USD") -> APIResponse:
        """Get exchange rates for a base currency"""
        endpoint = f"{self.api_key}/latest/{base_currency}"
        return self._make_request(endpoint)
    
    def convert_currency(self, from_currency: str, to_currency: str, amount: float) -> APIResponse:
        """Convert amount from one currency to another"""
        endpoint = f"{self.api_key}/pair/{from_currency}/{to_currency}/{amount}"
        return self._make_request(endpoint)
    
    def test_connection(self) -> bool:
        """Test API connection"""
        response = self.get_exchange_rates("USD")
        return response.success


class APIClient:
    """Main API client facade"""
    
    def __init__(self):
        self.weather = OpenWeatherMapClient()
        self.places = FoursquareClient()
        self.currency = ExchangeRateClient()
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all API connections"""
        return {
            "weather": self.weather.test_connection(),
            "places": self.places.test_connection(),
            "currency": self.currency.test_connection()
        }
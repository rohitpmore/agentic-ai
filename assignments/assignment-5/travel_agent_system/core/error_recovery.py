"""
Enhanced Error Recovery for LangGraph Workflow
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def exponential_backoff_retry(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        break
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            # Return error result if all retries failed
            return {"error": f"Failed after {max_retries} retries: {str(last_exception)}"}
        
        return wrapper
    return decorator


class APIFallbackManager:
    """Manages fallback strategies for failed API calls."""
    
    def __init__(self):
        self.fallback_data = {
            "weather": {
                "temperature": 20,
                "condition": "mild",
                "forecast": ["Partly cloudy", "Pleasant weather expected"],
                "recommendations": ["Light jacket recommended", "Good weather for outdoor activities"]
            },
            "attractions": {
                "attractions": [
                    {"name": "City Center", "type": "area", "rating": 4.0},
                    {"name": "Local Museum", "type": "cultural", "rating": 4.2},
                    {"name": "Historic District", "type": "historical", "rating": 4.1}
                ],
                "restaurants": [
                    {"name": "Local Cuisine", "type": "restaurant", "rating": 4.0},
                    {"name": "Cafe Central", "type": "cafe", "rating": 4.1}
                ]
            },
            "hotels": {
                "hotel_options": [
                    {"name": "City Hotel", "rating": 4.0, "price_range": "mid-range"},
                    {"name": "Budget Inn", "rating": 3.5, "price_range": "budget"},
                    {"name": "Luxury Suites", "rating": 4.5, "price_range": "luxury"}
                ],
                "cost_estimates": {"budget": 80, "mid-range": 150, "luxury": 300}
            }
        }
    
    def get_fallback_data(self, data_type: str, destination: str = "destination") -> Dict[str, Any]:
        """
        Get fallback data for failed API calls.
        
        Args:
            data_type: Type of data (weather, attractions, hotels)
            destination: Destination name for context
            
        Returns:
            Fallback data dictionary
        """
        base_data = self.fallback_data.get(data_type, {})
        
        # Add context information
        fallback_result = base_data.copy()
        fallback_result.update({
            "fallback_mode": True,
            "destination": destination,
            "message": f"Using fallback data for {data_type} due to API unavailability"
        })
        
        logger.info(f"Providing fallback data for {data_type} in {destination}")
        return fallback_result


# Global fallback manager instance
_fallback_manager = APIFallbackManager()


def get_fallback_manager() -> APIFallbackManager:
    """Get the global fallback manager instance."""
    return _fallback_manager


def with_api_fallback(data_type: str):
    """
    Decorator to add API fallback capability to agent methods.
    
    Args:
        data_type: Type of data for fallback (weather, attractions, hotels)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check if result indicates API failure
                if isinstance(result, dict) and result.get("error"):
                    logger.warning(f"API call failed in {func.__name__}, using fallback")
                    destination = kwargs.get("destination", args[1] if len(args) > 1 else "destination")
                    return _fallback_manager.get_fallback_data(data_type, destination)
                
                return result
                
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}, using fallback")
                destination = kwargs.get("destination", args[1] if len(args) > 1 else "destination")
                return _fallback_manager.get_fallback_data(data_type, destination)
        
        return wrapper
    return decorator


def graceful_degradation(critical_data: Optional[Dict[str, Any]] = None):
    """
    Decorator for graceful degradation when partial data is available.
    
    Args:
        critical_data: Minimum required data fields
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check if we have minimum required data
                if critical_data and isinstance(result, dict):
                    missing_fields = []
                    for field in critical_data:
                        if field not in result or result[field] is None:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        logger.warning(f"Missing critical data in {func.__name__}: {missing_fields}")
                        # Add default values for missing critical data
                        for field in missing_fields:
                            if field in critical_data:
                                result[field] = critical_data[field]
                
                return result
                
            except Exception as e:
                logger.error(f"Graceful degradation triggered in {func.__name__}: {e}")
                return critical_data or {"error": str(e), "partial_data": True}
        
        return wrapper
    return decorator
"""
LangGraph Tools - Convert CostCalculator and CurrencyConverter to LangGraph tools
"""

from typing import Dict, Any, List, Optional
import logging
from langchain_core.tools import tool
from .cost_calculator import CostCalculator
from .currency_converter import CurrencyConverter

logger = logging.getLogger(__name__)

# Global instances for tool functions
_cost_calculator = CostCalculator()
_currency_converter = CurrencyConverter()


@tool
def add_costs(costs: List[float]) -> float:
    """Add multiple costs together for travel expenses."""
    return _cost_calculator.add_costs(costs)


@tool  
def multiply_daily_cost(daily_cost: float, days: int) -> float:
    """Calculate total cost for multiple days."""
    return _cost_calculator.multiply_daily_cost(daily_cost, days)


@tool
def calculate_total_trip_cost(costs_breakdown: Dict[str, float]) -> float:
    """Calculate total trip cost from breakdown of categories."""
    return _cost_calculator.calculate_total_trip_cost(costs_breakdown)


@tool
def calculate_daily_budget(total_budget: float, days: int) -> float:
    """Calculate daily budget from total budget."""
    return _cost_calculator.calculate_daily_budget(total_budget, days)


@tool
def calculate_cost_per_person(total_cost: float, num_people: int) -> float:
    """Calculate cost per person for group travel."""
    return _cost_calculator.calculate_cost_per_person(total_cost, num_people)


@tool
def calculate_budget_remaining(total_budget: float, spent_amount: float) -> float:
    """Calculate remaining budget after expenses."""
    return _cost_calculator.calculate_budget_remaining(total_budget, spent_amount)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> Optional[float]:
    """Convert amount from one currency to another using live exchange rates."""
    return _currency_converter.convert_currency(amount, from_currency, to_currency)


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """Get current exchange rate between two currencies."""
    return _currency_converter.get_exchange_rate(from_currency, to_currency)


@tool
def format_currency_amount(amount: float, currency: str) -> str:
    """Format currency amount with proper symbol and formatting."""
    return _currency_converter.format_currency_amount(amount, currency)


@tool
def extract_trip_highlights(trip_data: Dict[str, Any]) -> List[str]:
    """Extract key highlights from trip data for summary generation."""
    highlights = []
    
    # Extract attractions highlights
    attractions_data = trip_data.get("attractions_data", {})
    if attractions_data and attractions_data.get("attractions"):
        for attraction in attractions_data["attractions"][:3]:
            name = attraction.get("name", "Attraction")
            highlights.append(f"Visit {name}")
    
    # Extract hotel highlights
    hotels_data = trip_data.get("hotels_data", {})
    if hotels_data and hotels_data.get("hotels"):
        hotel_count = len(hotels_data["hotels"])
        highlights.append(f"{hotel_count} accommodation options available")
        
        # Add featured hotel
        if hotels_data["hotels"]:
            top_hotel = hotels_data["hotels"][0]
            hotel_name = top_hotel.get("name", "Featured accommodation")
            highlights.append(f"Stay at {hotel_name}")
    
    # Extract weather highlights
    weather_data = trip_data.get("weather_data", {})
    if weather_data and weather_data.get("forecast"):
        forecast = weather_data["forecast"]
        if forecast:
            first_day = forecast[0]
            temp = first_day.get("temperature", {}).get("avg")
            weather_desc = first_day.get("description", "")
            if temp and weather_desc:
                highlights.append(f"Weather: {weather_desc}, {temp}°C")
    
    # Extract budget highlights
    itinerary_data = trip_data.get("itinerary_data", {})
    if itinerary_data and itinerary_data.get("total_cost"):
        highlights.append(f"Total budget: ${itinerary_data['total_cost']:.2f}")
    
    return highlights


@tool
def calculate_trip_statistics(trip_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive trip statistics for summary generation."""
    stats = {}
    
    # Basic trip info
    stats["destination"] = trip_data.get("destination", "Unknown")
    
    # Itinerary statistics
    itinerary_data = trip_data.get("itinerary_data", {})
    if itinerary_data:
        stats["total_days"] = itinerary_data.get("total_days", 0)
        stats["total_cost"] = itinerary_data.get("total_cost", 0)
        if stats["total_days"] > 0:
            stats["daily_average"] = stats["total_cost"] / stats["total_days"]
        else:
            stats["daily_average"] = 0
        stats["currency"] = itinerary_data.get("currency", "USD")
    
    # Agent data counts
    attractions_data = trip_data.get("attractions_data", {})
    hotels_data = trip_data.get("hotels_data", {})
    weather_data = trip_data.get("weather_data", {})
    
    stats["attraction_count"] = len(attractions_data.get("attractions", [])) if attractions_data else 0
    stats["hotel_count"] = len(hotels_data.get("hotels", [])) if hotels_data else 0
    stats["weather_available"] = bool(weather_data and weather_data.get("forecast"))
    
    return stats


@tool
def format_cost_breakdown_display(cost_breakdown: Dict[str, float]) -> Dict[str, Dict[str, str]]:
    """Format cost breakdown for display with percentages and formatted amounts."""
    if not cost_breakdown:
        return {"categories": {}}
    
    total = sum(cost_breakdown.values()) if cost_breakdown.values() else 1
    categories = {}
    
    for category, amount in cost_breakdown.items():
        percentage = (amount / total * 100) if total > 0 else 0
        categories[category] = {
            "formatted": f"${amount:.2f}",
            "percentage": f"{percentage:.1f}"
        }
    
    return {"categories": categories}


# List of all available tools
TRAVEL_TOOLS = [
    add_costs,
    multiply_daily_cost, 
    calculate_total_trip_cost,
    calculate_daily_budget,
    calculate_cost_per_person,
    calculate_budget_remaining,
    convert_currency,
    get_exchange_rate,
    format_currency_amount,
    extract_trip_highlights,
    calculate_trip_statistics,
    format_cost_breakdown_display
]
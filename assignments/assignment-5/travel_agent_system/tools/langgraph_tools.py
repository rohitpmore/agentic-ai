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
    format_currency_amount
]
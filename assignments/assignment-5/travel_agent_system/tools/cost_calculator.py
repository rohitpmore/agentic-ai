"""
Cost Calculator Tool - Mathematical operations for travel planning
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Utility tool for cost calculations and budget planning.
    Performs specific mathematical operations for travel expense calculations.
    """
    
    def __init__(self):
        """Initialize cost calculator"""
        self.calculations_history: List[Dict[str, Any]] = []
    
    def add_costs(self, costs: List[float]) -> float:
        """
        Add multiple costs together.
        
        Args:
            costs: List of cost values to sum
            
        Returns:
            Total sum of all costs
        """
        if not costs:
            return 0.0
        
        # Filter out None values and ensure all are numeric
        valid_costs = [cost for cost in costs if cost is not None and isinstance(cost, (int, float))]
        
        if not valid_costs:
            logger.warning("No valid costs provided for addition")
            return 0.0
        
        total = sum(valid_costs)
        
        # Record calculation
        self.calculations_history.append({
            "operation": "add_costs",
            "input": valid_costs,
            "result": total
        })
        
        logger.info(f"Added {len(valid_costs)} costs: {total}")
        return total
    
    def multiply_daily_cost(self, daily_cost: float, days: int) -> float:
        """
        Calculate total cost for multiple days.
        
        Args:
            daily_cost: Cost per day
            days: Number of days
            
        Returns:
            Total cost for all days
        """
        if not isinstance(daily_cost, (int, float)) or not isinstance(days, int):
            logger.error("Invalid input types for multiplication")
            return 0.0
        
        if daily_cost < 0 or days < 0:
            logger.error("Negative values not allowed for cost calculation")
            return 0.0
        
        total = daily_cost * days
        
        # Record calculation
        self.calculations_history.append({
            "operation": "multiply_daily_cost",
            "input": {"daily_cost": daily_cost, "days": days},
            "result": total
        })
        
        logger.info(f"Daily cost {daily_cost} × {days} days = {total}")
        return total
    
    def calculate_total_trip_cost(self, costs_breakdown: Dict[str, float]) -> float:
        """
        Calculate total trip cost from breakdown.
        
        Args:
            costs_breakdown: Dictionary of cost categories and amounts
            
        Returns:
            Total cost across all categories
        """
        if not costs_breakdown:
            return 0.0
        
        # Filter out None values and ensure all are numeric
        valid_costs = {}
        for category, cost in costs_breakdown.items():
            if cost is not None and isinstance(cost, (int, float)) and cost >= 0:
                valid_costs[category] = cost
            else:
                logger.warning(f"Invalid cost for category '{category}': {cost}")
        
        if not valid_costs:
            logger.warning("No valid costs in breakdown")
            return 0.0
        
        total = sum(valid_costs.values())
        
        # Record calculation
        self.calculations_history.append({
            "operation": "calculate_total_trip_cost",
            "input": valid_costs,
            "result": total
        })
        
        logger.info(f"Total trip cost from {len(valid_costs)} categories: {total}")
        return total
    
    def calculate_daily_budget(self, total_budget: float, days: int) -> float:
        """
        Calculate daily budget from total.
        
        Args:
            total_budget: Total available budget
            days: Number of days for the trip
            
        Returns:
            Daily budget amount
        """
        if not isinstance(total_budget, (int, float)) or not isinstance(days, int):
            logger.error("Invalid input types for daily budget calculation")
            return 0.0
        
        if total_budget <= 0 or days <= 0:
            logger.error("Budget and days must be positive values")
            return 0.0
        
        daily_budget = total_budget / days
        
        # Record calculation
        self.calculations_history.append({
            "operation": "calculate_daily_budget",
            "input": {"total_budget": total_budget, "days": days},
            "result": daily_budget
        })
        
        logger.info(f"Daily budget: {total_budget} ÷ {days} days = {daily_budget}")
        return daily_budget
    
    def calculate_cost_per_person(self, total_cost: float, num_people: int) -> float:
        """
        Calculate cost per person for group travel.
        
        Args:
            total_cost: Total trip cost
            num_people: Number of people in the group
            
        Returns:
            Cost per person
        """
        if not isinstance(total_cost, (int, float)) or not isinstance(num_people, int):
            logger.error("Invalid input types for per-person calculation")
            return 0.0
        
        if total_cost <= 0 or num_people <= 0:
            logger.error("Total cost and number of people must be positive")
            return 0.0
        
        cost_per_person = total_cost / num_people
        
        # Record calculation
        self.calculations_history.append({
            "operation": "calculate_cost_per_person",
            "input": {"total_cost": total_cost, "num_people": num_people},
            "result": cost_per_person
        })
        
        logger.info(f"Cost per person: {total_cost} ÷ {num_people} people = {cost_per_person}")
        return cost_per_person
    
    def calculate_budget_remaining(self, total_budget: float, spent_amount: float) -> float:
        """
        Calculate remaining budget after expenses.
        
        Args:
            total_budget: Total available budget
            spent_amount: Amount already spent
            
        Returns:
            Remaining budget (can be negative if over budget)
        """
        if not isinstance(total_budget, (int, float)) or not isinstance(spent_amount, (int, float)):
            logger.error("Invalid input types for budget remaining calculation")
            return 0.0
        
        remaining = total_budget - spent_amount
        
        # Record calculation
        self.calculations_history.append({
            "operation": "calculate_budget_remaining",
            "input": {"total_budget": total_budget, "spent_amount": spent_amount},
            "result": remaining
        })
        
        if remaining < 0:
            logger.warning(f"Over budget by {abs(remaining)}")
        else:
            logger.info(f"Remaining budget: {remaining}")
        
        return remaining
    
    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all calculations performed.
        
        Returns:
            List of calculation records
        """
        return self.calculations_history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.calculations_history.clear()
        logger.info("Calculation history cleared")
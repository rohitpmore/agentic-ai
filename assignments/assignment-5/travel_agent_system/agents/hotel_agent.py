"""
Hotel Agent - Accommodation search and cost estimation
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class HotelAgent:
    """
    Reasoning agent for hotel search and accommodation recommendations.
    Since most hotel APIs require payment, this agent uses estimation logic
    combined with general hotel data to provide realistic recommendations.
    """
    
    def __init__(self):
        """Initialize hotel agent"""
        self.search_history: List[Dict[str, Any]] = []
        
        # Hotel price estimation data (average prices per night in USD)
        self.base_prices = {
            "budget": {"min": 30, "max": 80, "avg": 55},
            "mid_range": {"min": 80, "max": 200, "avg": 140},
            "luxury": {"min": 200, "max": 500, "avg": 350},
            "ultra_luxury": {"min": 500, "max": 1500, "avg": 800}
        }
        
        # City price multipliers (relative to base prices)
        self.city_multipliers = {
            # Major expensive cities
            "new york": 1.8, "san francisco": 1.7, "london": 1.6, "paris": 1.5,
            "tokyo": 1.4, "singapore": 1.3, "sydney": 1.3, "zurich": 1.6,
            "oslo": 1.5, "copenhagen": 1.4, "stockholm": 1.3, "amsterdam": 1.3,
            
            # Medium-cost cities
            "barcelona": 1.2, "madrid": 1.1, "rome": 1.2, "berlin": 1.1,
            "vienna": 1.2, "prague": 0.9, "budapest": 0.8, "warsaw": 0.7,
            "lisbon": 1.0, "dublin": 1.3, "edinburgh": 1.2, "brussels": 1.2,
            
            # Lower-cost cities
            "bangkok": 0.6, "ho chi minh city": 0.5, "hanoi": 0.5, "mumbai": 0.4,
            "delhi": 0.4, "cairo": 0.5, "istanbul": 0.7, "mexico city": 0.6,
            "buenos aires": 0.6, "lima": 0.7, "bogota": 0.6, "kiev": 0.5,
            
            # Default multiplier for unknown cities
            "default": 1.0
        }
    
    def search_hotels(self, destination: str, budget_range: Optional[Dict[str, float]] = None,
                     travel_dates: Optional[Dict[str, str]] = None,
                     preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for hotel accommodations and provide cost estimates.
        
        Args:
            destination: Destination city name
            budget_range: Optional dict with 'min' and 'max' daily budget
            travel_dates: Optional dict with 'start_date' and 'end_date'
            preferences: Optional preferences (room_type, amenities, etc.)
            
        Returns:
            Hotel search results with cost estimates
        """
        if not destination:
            logger.error("Destination is required for hotel search")
            return {"error": "Destination is required"}
        
        search_result = {
            "destination": destination,
            "budget_range": budget_range,
            "travel_dates": travel_dates,
            "hotels": [],
            "cost_estimates": {},
            "recommendations": [],
            "search_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Calculate number of nights if dates provided
            nights = self._calculate_nights(travel_dates) if travel_dates else 3  # Default 3 nights
            
            # Generate hotel options based on budget and preferences
            hotels = self._generate_hotel_options(destination, budget_range, preferences, nights)
            search_result["hotels"] = hotels
            
            # Calculate cost estimates
            search_result["cost_estimates"] = self._calculate_cost_estimates(hotels, nights)
            
            # Generate recommendations
            search_result["recommendations"] = self._generate_hotel_recommendations(
                hotels, budget_range, destination, nights
            )
            
            # Record search
            self.search_history.append({
                "destination": destination,
                "budget_range": budget_range,
                "travel_dates": travel_dates,
                "nights": nights,
                "timestamp": datetime.now().isoformat(),
                "hotels_found": len(hotels),
                "success": True
            })
            
            logger.info(f"Completed hotel search for {destination} - found {len(hotels)} options")
            return search_result
            
        except Exception as e:
            logger.error(f"Error searching hotels for {destination}: {e}")
            search_result["error"] = str(e)
            
            # Record failed search
            self.search_history.append({
                "destination": destination,
                "budget_range": budget_range,
                "travel_dates": travel_dates,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return search_result
    
    def _calculate_nights(self, travel_dates: Dict[str, str]) -> int:
        """Calculate number of nights from travel dates."""
        try:
            start_date = datetime.fromisoformat(travel_dates["start_date"].replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(travel_dates["end_date"].replace("Z", "+00:00"))
            nights = (end_date - start_date).days
            return max(1, nights)  # At least 1 night
        except Exception as e:
            logger.warning(f"Could not parse travel dates: {e}")
            return 3  # Default fallback
    
    def _generate_hotel_options(self, destination: str, budget_range: Optional[Dict[str, float]],
                              preferences: Optional[Dict[str, Any]], nights: int) -> List[Dict[str, Any]]:
        """Generate realistic hotel options based on destination and budget."""
        
        # Get city multiplier
        city_key = destination.lower()
        multiplier = self.city_multipliers.get(city_key, self.city_multipliers["default"])
        
        # Determine budget categories to include
        if budget_range:
            daily_budget = budget_range.get("max", 200)
            if daily_budget <= 100:
                categories = ["budget", "mid_range"]
            elif daily_budget <= 300:
                categories = ["budget", "mid_range", "luxury"]
            else:
                categories = ["mid_range", "luxury", "ultra_luxury"]
        else:
            categories = ["budget", "mid_range", "luxury"]
        
        hotels = []
        hotel_names = self._get_hotel_names(destination)
        
        for i, category in enumerate(categories):
            # Generate 2-3 hotels per category
            num_hotels = min(3, len(hotel_names) - len(hotels))
            
            for j in range(num_hotels):
                hotel_idx = len(hotels)
                if hotel_idx >= len(hotel_names):
                    break
                
                base_price = self.base_prices[category]["avg"]
                adjusted_price = base_price * multiplier
                
                # Add some variation (±20%)
                price_variation = random.uniform(0.8, 1.2)
                final_price = round(adjusted_price * price_variation, 2)
                
                hotel = {
                    "name": hotel_names[hotel_idx],
                    "category": category,
                    "price_per_night": final_price,
                    "total_cost": final_price * nights,
                    "rating": self._generate_rating(category),
                    "amenities": self._generate_amenities(category),
                    "location": self._generate_location_info(destination),
                    "room_type": preferences.get("room_type", "Standard Room") if preferences else "Standard Room",
                    "booking_info": {
                        "cancellation": "Free cancellation" if category != "budget" else "Non-refundable",
                        "breakfast": "Included" if category in ["luxury", "ultra_luxury"] else "Available for extra cost",
                        "wifi": "Free" if category != "budget" else "Available"
                    },
                    "distance_to_center": round(random.uniform(0.5, 5.0), 1),
                    "estimated": True  # Mark as estimated data
                }
                
                hotels.append(hotel)
        
        return hotels
    
    def _get_hotel_names(self, destination: str) -> List[str]:
        """Generate realistic hotel names for the destination."""
        
        # Generic hotel names that work for any city
        generic_names = [
            f"{destination} Grand Hotel",
            f"Hotel {destination} Plaza",
            f"{destination} Boutique Inn",
            f"The {destination} Heritage",
            f"{destination} Business Hotel",
            f"City Center {destination}",
            f"{destination} Comfort Lodge",
            f"The Royal {destination}",
            f"{destination} Garden Hotel",
            f"Metropolitan {destination}",
            f"{destination} Executive Suites",
            f"The {destination} Palace"
        ]
        
        # Shuffle to add variety
        random.shuffle(generic_names)
        return generic_names
    
    def _generate_rating(self, category: str) -> float:
        """Generate realistic rating based on hotel category."""
        rating_ranges = {
            "budget": (6.5, 7.5),
            "mid_range": (7.5, 8.5),
            "luxury": (8.5, 9.2),
            "ultra_luxury": (9.0, 9.8)
        }
        
        min_rating, max_rating = rating_ranges[category]
        return round(random.uniform(min_rating, max_rating), 1)
    
    def _generate_amenities(self, category: str) -> List[str]:
        """Generate amenities based on hotel category."""
        
        base_amenities = ["24-hour front desk", "Room service"]
        
        amenities_by_category = {
            "budget": base_amenities + ["Free WiFi", "Air conditioning"],
            "mid_range": base_amenities + ["Free WiFi", "Fitness center", "Restaurant", "Business center"],
            "luxury": base_amenities + ["Free WiFi", "Spa", "Pool", "Concierge", "Restaurant", "Bar", "Fitness center", "Valet parking"],
            "ultra_luxury": base_amenities + ["Free WiFi", "Full-service spa", "Multiple restaurants", "Pool", "Concierge", "Butler service", "Limousine service", "Private club level"]
        }
        
        return amenities_by_category.get(category, base_amenities)
    
    def _generate_location_info(self, destination: str) -> Dict[str, Any]:
        """Generate location information."""
        return {
            "area": "City Center",  # Simplified
            "nearby_attractions": ["Main attractions within walking distance"],
            "transportation": "Near public transportation"
        }
    
    def _calculate_cost_estimates(self, hotels: List[Dict[str, Any]], nights: int) -> Dict[str, Any]:
        """Calculate comprehensive cost estimates."""
        
        if not hotels:
            return {}
        
        prices = [hotel["price_per_night"] for hotel in hotels]
        total_costs = [hotel["total_cost"] for hotel in hotels]
        
        estimates = {
            "per_night": {
                "min": min(prices),
                "max": max(prices),
                "avg": round(sum(prices) / len(prices), 2)
            },
            "total_stay": {
                "min": min(total_costs),
                "max": max(total_costs),
                "avg": round(sum(total_costs) / len(total_costs), 2)
            },
            "nights": nights,
            "currency": "USD"
        }
        
        # Cost breakdown by category
        categories = {}
        for hotel in hotels:
            cat = hotel["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(hotel["price_per_night"])
        
        for cat, prices in categories.items():
            estimates[f"{cat}_avg"] = round(sum(prices) / len(prices), 2)
        
        return estimates
    
    def _generate_hotel_recommendations(self, hotels: List[Dict[str, Any]], 
                                      budget_range: Optional[Dict[str, float]],
                                      destination: str, nights: int) -> List[str]:
        """Generate personalized hotel recommendations."""
        recommendations = []
        
        if not hotels:
            recommendations.append("No hotels found - try expanding your search criteria")
            return recommendations
        
        # Find best value option
        best_value = min(hotels, key=lambda h: h["price_per_night"] / h["rating"])
        recommendations.append(f"Best value: {best_value['name']} - ${best_value['price_per_night']}/night, {best_value['rating']}/10 rating")
        
        # Find highest rated
        highest_rated = max(hotels, key=lambda h: h["rating"])
        if highest_rated != best_value:
            recommendations.append(f"Highest rated: {highest_rated['name']} - {highest_rated['rating']}/10 rating")
        
        # Budget-specific recommendations
        if budget_range:
            max_budget = budget_range.get("max", float('inf'))
            affordable_options = [h for h in hotels if h["price_per_night"] <= max_budget]
            
            if not affordable_options:
                recommendations.append(f"No hotels found within ${max_budget}/night budget - consider expanding budget")
            elif len(affordable_options) == len(hotels):
                recommendations.append("All options are within your budget")
            else:
                recommendations.append(f"{len(affordable_options)} of {len(hotels)} hotels are within your budget")
        
        # Location recommendations
        city_key = destination.lower()
        if city_key in ["new york", "london", "paris", "tokyo"]:
            recommendations.append("Book early for better rates in this popular destination")
        
        # General recommendations
        luxury_count = len([h for h in hotels if h["category"] in ["luxury", "ultra_luxury"]])
        if luxury_count > 0:
            recommendations.append(f"{luxury_count} luxury options available for special occasions")
        
        budget_count = len([h for h in hotels if h["category"] == "budget"])
        if budget_count > 0:
            recommendations.append(f"{budget_count} budget-friendly options for cost-conscious travelers")
        
        # Seasonal advice
        recommendations.append("Consider booking refundable rates for flexible travel plans")
        
        return recommendations
    
    def estimate_accommodation_costs(self, destination: str, nights: int, 
                                   budget_level: str = "medium") -> Dict[str, Any]:
        """
        Provide quick accommodation cost estimates without full search.
        
        Args:
            destination: Destination city name
            nights: Number of nights
            budget_level: "low", "medium", or "high"
            
        Returns:
            Cost estimation summary
        """
        
        # Map budget levels to categories
        category_mapping = {
            "low": "budget",
            "medium": "mid_range", 
            "high": "luxury"
        }
        
        category = category_mapping.get(budget_level, "mid_range")
        
        # Get city multiplier
        city_key = destination.lower()
        multiplier = self.city_multipliers.get(city_key, self.city_multipliers["default"])
        
        # Calculate estimates
        base_prices = self.base_prices[category]
        
        estimates = {
            "destination": destination,
            "nights": nights,
            "budget_level": budget_level,
            "currency": "USD",
            "per_night": {
                "min": round(base_prices["min"] * multiplier, 2),
                "max": round(base_prices["max"] * multiplier, 2),
                "avg": round(base_prices["avg"] * multiplier, 2)
            },
            "total_stay": {
                "min": round(base_prices["min"] * multiplier * nights, 2),
                "max": round(base_prices["max"] * multiplier * nights, 2),
                "avg": round(base_prices["avg"] * multiplier * nights, 2)
            },
            "city_cost_level": self._get_cost_level_description(multiplier),
            "estimated": True
        }
        
        return estimates
    
    def _get_cost_level_description(self, multiplier: float) -> str:
        """Get description of city cost level."""
        if multiplier >= 1.5:
            return "Very expensive destination"
        elif multiplier >= 1.2:
            return "Expensive destination"
        elif multiplier >= 0.8:
            return "Moderate cost destination"
        else:
            return "Budget-friendly destination"
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get history of hotel searches performed."""
        return self.search_history.copy()
    
    def clear_history(self):
        """Clear search history."""
        self.search_history.clear()
        logger.info("Hotel search history cleared")
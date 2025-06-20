"""
Itinerary Agent - Day-by-day trip planning and integration
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


class ItineraryAgent:
    """
    Reasoning agent for creating comprehensive day-by-day travel itineraries.
    Integrates data from weather, attractions, and hotel agents to create optimized plans.
    Uses tools for cost calculations and currency conversions.
    """
    
    def __init__(self, tool_node: Optional[ToolNode] = None):
        """
        Initialize itinerary agent with LangGraph ToolNode
        
        Args:
            tool_node: LangGraph ToolNode with travel tools
        """
        self.tool_node = tool_node
        self.itinerary_history: List[Dict[str, Any]] = []
    
    def create_itinerary(self, trip_data: Dict[str, Any], 
                        preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create comprehensive day-by-day itinerary using integrated data.
        
        Args:
            trip_data: Integrated data from weather, attractions, and hotel agents
            preferences: Optional user preferences (budget, interests, pace, etc.)
            
        Returns:
            Complete itinerary with daily plans and cost breakdown
        """
        
        # Validate required data
        if not trip_data.get("destination"):
            logger.error("Destination is required for itinerary creation")
            return {"error": "Destination is required"}
        
        itinerary = {
            "destination": trip_data["destination"],
            "trip_dates": trip_data.get("travel_dates"),
            "total_days": self._calculate_trip_days(trip_data.get("travel_dates")),
            "daily_plans": [],
            "cost_breakdown": {},
            "total_cost": 0,
            "recommendations": [],
            "packing_list": [],
            "important_notes": [],
            "creation_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Extract data from integrated sources
            weather_data = trip_data.get("weather_data", {})
            attractions_data = trip_data.get("attractions_data", {})
            hotels_data = trip_data.get("hotels_data", {})
            
            # Create daily plans
            itinerary["daily_plans"] = self._create_daily_plans(
                itinerary["total_days"], weather_data, attractions_data, preferences
            )
            
            # Calculate costs using tools
            itinerary["cost_breakdown"] = self._calculate_trip_costs(
                itinerary["daily_plans"], hotels_data, preferences
            )
            
            # Convert currency if needed
            if preferences and preferences.get("currency") and preferences["currency"] != "USD":
                itinerary["cost_breakdown"] = self._convert_costs_to_currency(
                    itinerary["cost_breakdown"], "USD", preferences["currency"]
                )
            
            # Calculate total cost using cost calculator tool
            itinerary["total_cost"] = self._calculate_total_trip_cost(
                itinerary["cost_breakdown"]
            )
            
            # Generate recommendations and notes
            itinerary["recommendations"] = self._generate_itinerary_recommendations(
                itinerary, trip_data, preferences
            )
            
            itinerary["packing_list"] = self._create_packing_list(weather_data, itinerary["daily_plans"])
            itinerary["important_notes"] = self._generate_important_notes(trip_data)
            
            # Record successful creation
            self.itinerary_history.append({
                "destination": trip_data["destination"],
                "days": itinerary["total_days"],
                "timestamp": datetime.now().isoformat(),
                "total_cost": itinerary["total_cost"],
                "success": True
            })
            
            logger.info(f"Created {itinerary['total_days']}-day itinerary for {trip_data['destination']}")
            return itinerary
            
        except Exception as e:
            logger.error(f"Error creating itinerary: {e}")
            itinerary["error"] = str(e)
            
            # Record failed creation
            self.itinerary_history.append({
                "destination": trip_data.get("destination"),
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return itinerary
    
    def _calculate_trip_days(self, travel_dates: Optional[Dict[str, str]]) -> int:
        """Calculate number of days for the trip."""
        if not travel_dates or not travel_dates.get("start_date") or not travel_dates.get("end_date"):
            return 3  # Default 3-day trip
        
        try:
            start_date = datetime.fromisoformat(travel_dates["start_date"].replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(travel_dates["end_date"].replace("Z", "+00:00"))
            days = (end_date - start_date).days + 1  # Include both start and end days
            return max(1, min(days, 14))  # Limit to reasonable range
        except Exception as e:
            logger.warning(f"Could not parse travel dates: {e}")
            return 3
    
    def _create_daily_plans(self, total_days: int, weather_data: Dict[str, Any],
                          attractions_data: Dict[str, Any], 
                          preferences: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create day-by-day plans based on available data."""
        
        daily_plans = []
        
        # Get weather forecast if available
        weather_forecast = weather_data.get("forecast", [])
        
        # Get attractions by category
        attractions = attractions_data.get("attractions", [])
        restaurants = attractions_data.get("restaurants", [])
        activities = attractions_data.get("activities", [])
        entertainment = attractions_data.get("entertainment", [])
        
        # Determine trip pace
        pace = preferences.get("pace", "moderate") if preferences else "moderate"
        activities_per_day = {"slow": 2, "moderate": 3, "fast": 4}.get(pace, 3)
        
        for day in range(1, total_days + 1):
            # Get weather for this day if available
            day_weather = None
            if day <= len(weather_forecast):
                day_weather = weather_forecast[day - 1]
            
            # Create daily plan
            daily_plan = {
                "day": day,
                "date": self._calculate_date_for_day(day, preferences),
                "weather": day_weather,
                "morning": [],
                "afternoon": [],
                "evening": [],
                "meals": {"breakfast": None, "lunch": None, "dinner": None},
                "estimated_costs": {"activities": 0, "meals": 0, "transportation": 30},
                "notes": []
            }
            
            # Plan activities based on weather and available attractions
            planned_activities = self._plan_daily_activities(
                day, day_weather, attractions, activities, activities_per_day
            )
            
            # Distribute activities throughout the day
            self._distribute_activities_by_time(daily_plan, planned_activities)
            
            # Plan meals
            daily_plan["meals"] = self._plan_daily_meals(day, restaurants, preferences)
            
            # Estimate costs
            daily_plan["estimated_costs"] = self._estimate_daily_costs(
                daily_plan, preferences
            )
            
            # Add weather-specific notes
            if day_weather:
                daily_plan["notes"].extend(self._generate_weather_notes(day_weather))
            
            daily_plans.append(daily_plan)
        
        return daily_plans
    
    def _calculate_date_for_day(self, day: int, preferences: Optional[Dict[str, Any]]) -> Optional[str]:
        """Calculate actual date for a given day number."""
        if not preferences or not preferences.get("travel_dates"):
            return None
        
        try:
            start_date = datetime.fromisoformat(
                preferences["travel_dates"]["start_date"].replace("Z", "+00:00")
            )
            target_date = start_date + timedelta(days=day - 1)
            return target_date.strftime("%Y-%m-%d")
        except Exception:
            return None
    
    def _plan_daily_activities(self, day: int, day_weather: Optional[Dict[str, Any]],
                             attractions: List[Dict[str, Any]], 
                             activities: List[Dict[str, Any]],
                             target_count: int) -> List[Dict[str, Any]]:
        """Plan activities for a specific day considering weather."""
        
        planned_activities = []
        all_options = attractions + activities
        
        if not all_options:
            # Fallback generic activities
            planned_activities = [
                {"name": "Explore city center", "type": "sightseeing", "duration": "2-3 hours", "indoor": False},
                {"name": "Visit local market", "type": "cultural", "duration": "1-2 hours", "indoor": False},
                {"name": "Museum visit", "type": "cultural", "duration": "2-3 hours", "indoor": True}
            ]
            return planned_activities[:target_count]
        
        # Filter activities based on weather
        suitable_activities = []
        
        if day_weather:
            weather_desc = day_weather.get("description", "").lower()
            is_rainy = "rain" in weather_desc
            is_cold = day_weather.get("temperature", {}).get("avg", 20) < 10
            
            for activity in all_options:
                # Assume outdoor activities if not specified
                is_indoor = activity.get("indoor", False)
                
                if is_rainy and not is_indoor:
                    continue  # Skip outdoor activities in rain
                if is_cold and not is_indoor and "outdoor" in activity.get("name", "").lower():
                    continue  # Skip cold outdoor activities
                
                suitable_activities.append(activity)
        else:
            suitable_activities = all_options
        
        # Select diverse activities
        selected = []
        categories_used = set()
        
        for activity in suitable_activities:
            if len(selected) >= target_count:
                break
            
            category = activity.get("category", "general")
            
            # Try to avoid repeating categories
            if category not in categories_used or len(selected) < target_count // 2:
                planned_activities.append({
                    "name": activity.get("name", "Activity"),
                    "type": category,
                    "rating": activity.get("rating"),
                    "price_level": activity.get("price_level", "medium"),
                    "address": activity.get("address"),
                    "duration": "2-3 hours",  # Default duration
                    "indoor": activity.get("indoor", False)
                })
                selected.append(activity)
                categories_used.add(category)
        
        # Fill remaining slots with generic activities if needed
        while len(planned_activities) < target_count:
            planned_activities.append({
                "name": f"Free time / Exploration",
                "type": "flexible",
                "duration": "1-2 hours",
                "indoor": False
            })
        
        return planned_activities
    
    def _distribute_activities_by_time(self, daily_plan: Dict[str, Any], 
                                     activities: List[Dict[str, Any]]):
        """Distribute activities across morning, afternoon, and evening."""
        
        for i, activity in enumerate(activities):
            if i == 0:  # First activity in morning
                daily_plan["morning"].append(activity)
            elif i == len(activities) - 1 and len(activities) > 2:  # Last activity in evening if 3+ activities
                daily_plan["evening"].append(activity)
            else:  # Middle activities in afternoon
                daily_plan["afternoon"].append(activity)
        
        # Ensure each time period has something if possible
        if not daily_plan["morning"] and daily_plan["afternoon"]:
            daily_plan["morning"].append(daily_plan["afternoon"].pop(0))
        
        if not daily_plan["evening"] and len(daily_plan["afternoon"]) > 1:
            daily_plan["evening"].append(daily_plan["afternoon"].pop())
    
    def _plan_daily_meals(self, day: int, restaurants: List[Dict[str, Any]],
                         preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan meals for the day."""
        
        meals = {"breakfast": None, "lunch": None, "dinner": None}
        
        if restaurants:
            # Select restaurants trying to vary by meal type and price
            available_restaurants = restaurants.copy()
            
            # Try to assign different restaurants to different meals
            if len(available_restaurants) >= 1:
                meals["dinner"] = {
                    "restaurant": available_restaurants[0]["name"],
                    "type": "dinner",
                    "estimated_cost": self._estimate_meal_cost("dinner", available_restaurants[0].get("price_level", "medium"))
                }
            
            if len(available_restaurants) >= 2:
                meals["lunch"] = {
                    "restaurant": available_restaurants[1]["name"],
                    "type": "lunch", 
                    "estimated_cost": self._estimate_meal_cost("lunch", available_restaurants[1].get("price_level", "medium"))
                }
            
            # Breakfast is often at hotel or simple cafe
            meals["breakfast"] = {
                "restaurant": "Local cafe / hotel",
                "type": "breakfast",
                "estimated_cost": self._estimate_meal_cost("breakfast", "low")
            }
        else:
            # Fallback meal planning
            meals = {
                "breakfast": {"restaurant": "Local cafe", "type": "breakfast", "estimated_cost": 15},
                "lunch": {"restaurant": "Local restaurant", "type": "lunch", "estimated_cost": 25},
                "dinner": {"restaurant": "Local restaurant", "type": "dinner", "estimated_cost": 40}
            }
        
        return meals
    
    def _estimate_meal_cost(self, meal_type: str, price_level: str) -> float:
        """Estimate cost for a meal based on type and restaurant price level."""
        
        base_costs = {
            "breakfast": {"low": 8, "medium": 15, "high": 25},
            "lunch": {"low": 15, "medium": 25, "high": 40},
            "dinner": {"low": 25, "medium": 40, "high": 70}
        }
        
        return base_costs.get(meal_type, {}).get(price_level, 25)
    
    def _estimate_daily_costs(self, daily_plan: Dict[str, Any],
                            preferences: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate costs for all daily activities."""
        
        costs = {"activities": 0, "meals": 0, "transportation": 30}  # Base transportation cost
        
        # Calculate meal costs
        meals = daily_plan.get("meals", {})
        for meal in meals.values():
            if meal and meal.get("estimated_cost"):
                costs["meals"] += meal["estimated_cost"]
        
        # Calculate activity costs
        all_activities = (daily_plan.get("morning", []) + 
                         daily_plan.get("afternoon", []) + 
                         daily_plan.get("evening", []))
        
        for activity in all_activities:
            price_level = activity.get("price_level", "medium")
            activity_cost = {"low": 10, "medium": 25, "high": 50}.get(price_level, 25)
            costs["activities"] += activity_cost
        
        return costs
    
    def _generate_weather_notes(self, day_weather: Dict[str, Any]) -> List[str]:
        """Generate weather-specific notes for the day."""
        notes = []
        
        if not day_weather:
            return notes
        
        weather_desc = day_weather.get("description", "").lower()
        temp = day_weather.get("temperature", {}).get("avg")
        
        if "rain" in weather_desc:
            notes.append("Rainy weather expected - bring umbrella and plan indoor backup activities")
        elif "clear" in weather_desc:
            notes.append("Clear weather - perfect day for outdoor sightseeing")
        elif "snow" in weather_desc:
            notes.append("Snow expected - dress warmly and check transportation schedules")
        
        if temp is not None:
            if temp < 5:
                notes.append("Very cold day - dress in layers and limit outdoor exposure")
            elif temp > 30:
                notes.append("Hot day - stay hydrated and seek shade during midday")
        
        return notes
    
    def _calculate_trip_costs(self, daily_plans: List[Dict[str, Any]], 
                            hotels_data: Dict[str, Any],
                            preferences: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive trip costs using cost calculator tool."""
        
        # Collect daily costs
        daily_activity_costs = []
        daily_meal_costs = []
        daily_transport_costs = []
        
        for day_plan in daily_plans:
            costs = day_plan.get("estimated_costs", {})
            daily_activity_costs.append(costs.get("activities", 0))
            daily_meal_costs.append(costs.get("meals", 0))
            daily_transport_costs.append(costs.get("transportation", 30))
        
        # Calculate totals using cost calculator tool
        total_activities = self._add_costs(daily_activity_costs)
        total_meals = self._add_costs(daily_meal_costs)
        total_transport = self._add_costs(daily_transport_costs)
        
        # Hotel costs
        hotel_cost = 0
        if hotels_data and hotels_data.get("cost_estimates"):
            hotel_estimates = hotels_data["cost_estimates"]
            hotel_cost = hotel_estimates.get("total_stay", {}).get("avg", 0)
        
        # Miscellaneous costs (shopping, tips, etc.)
        misc_cost = len(daily_plans) * 20  # $20 per day for miscellaneous
        
        cost_breakdown = {
            "accommodation": hotel_cost,
            "activities": total_activities,
            "meals": total_meals,
            "transportation": total_transport,
            "miscellaneous": misc_cost
        }
        
        return cost_breakdown
    
    def _convert_costs_to_currency(self, cost_breakdown: Dict[str, float],
                                 from_currency: str, to_currency: str) -> Dict[str, float]:
        """Convert cost breakdown to target currency using currency converter tool."""
        
        try:
            converted_costs = self._convert_cost_breakdown(
                cost_breakdown, from_currency, to_currency
            )
            logger.info(f"Converted costs from {from_currency} to {to_currency}")
            return converted_costs
        except Exception as e:
            logger.error(f"Failed to convert currency: {e}")
            return cost_breakdown  # Return original if conversion fails
    
    def _generate_itinerary_recommendations(self, itinerary: Dict[str, Any],
                                          trip_data: Dict[str, Any],
                                          preferences: Optional[Dict[str, Any]]) -> List[str]:
        """Generate personalized recommendations for the itinerary."""
        
        recommendations = []
        
        # Budget recommendations
        total_cost = itinerary.get("total_cost", 0)
        daily_avg = total_cost / max(itinerary.get("total_days", 1), 1)
        
        if preferences and preferences.get("budget"):
            budget = preferences["budget"]
            if total_cost > budget:
                overage = total_cost - budget
                recommendations.append(f"Trip exceeds budget by ${overage:.2f} - consider reducing activity costs or accommodation level")
            elif total_cost < budget * 0.8:
                recommendations.append(f"Trip is under budget - consider upgrading accommodation or adding premium activities")
            else:
                recommendations.append("Trip cost aligns well with your budget")
        
        recommendations.append(f"Average daily cost: ${daily_avg:.2f}")
        
        # Weather-based recommendations
        weather_data = trip_data.get("weather_data", {})
        if weather_data.get("weather_alerts"):
            recommendations.append("Check weather alerts before travel - some activities may need adjustment")
        
        # Activity recommendations
        total_days = itinerary.get("total_days", 0)
        if total_days >= 5:
            recommendations.append("Consider designating one day as a rest/flexible day for longer trips")
        elif total_days <= 2:
            recommendations.append("Short trip - focus on must-see attractions to maximize your time")
        
        # Practical recommendations
        recommendations.extend([
            "Book popular attractions in advance to avoid disappointment",
            "Keep digital and physical copies of important documents",
            "Research local customs and tipping practices",
            "Consider purchasing travel insurance for international trips"
        ])
        
        return recommendations
    
    def _create_packing_list(self, weather_data: Dict[str, Any], 
                           daily_plans: List[Dict[str, Any]]) -> List[str]:
        """Create packing list based on weather and planned activities."""
        
        packing_list = []
        
        # Weather-based items
        if weather_data.get("packing_suggestions"):
            packing_list.extend(weather_data["packing_suggestions"])
        
        # Activity-based items
        activity_types = set()
        for plan in daily_plans:
            for period in ["morning", "afternoon", "evening"]:
                for activity in plan.get(period, []):
                    activity_types.add(activity.get("type", "general"))
        
        if "cultural" in activity_types:
            packing_list.append("Modest clothing for cultural sites")
        if "outdoor" in activity_types or "sightseeing" in activity_types:
            packing_list.extend(["Comfortable walking shoes", "Small backpack for day trips"])
        
        # Essential items
        essentials = [
            "Travel documents and copies",
            "Phone charger and power adapter",
            "Basic first aid kit",
            "Reusable water bottle",
            "Camera or smartphone for photos"
        ]
        
        packing_list.extend(essentials)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_packing_list = []
        for item in packing_list:
            if item not in seen:
                seen.add(item)
                unique_packing_list.append(item)
        
        return unique_packing_list
    
    def _generate_important_notes(self, trip_data: Dict[str, Any]) -> List[str]:
        """Generate important notes and reminders for the trip."""
        
        notes = []
        
        # Weather alerts
        weather_data = trip_data.get("weather_data", {})
        if weather_data.get("weather_alerts"):
            notes.extend(weather_data["weather_alerts"])
        
        # General travel notes
        notes.extend([
            "Verify passport validity (6+ months remaining) for international travel",
            "Check visa requirements for your destination",
            "Notify your bank of travel plans to avoid card blocks", 
            "Research local emergency numbers and embassy contacts",
            "Consider downloading offline maps and translation apps"
        ])
        
        return notes
    
    def get_itinerary_history(self) -> List[Dict[str, Any]]:
        """Get history of itineraries created."""
        return self.itinerary_history.copy()
    
    def clear_history(self):
        """Clear itinerary creation history."""
        self.itinerary_history.clear()
        logger.info("Itinerary creation history cleared")
    
    def _add_costs(self, costs: List[float]) -> float:
        """Use LangGraph tool to add costs."""
        if self.tool_node:
            from ..tools.langgraph_tools import add_costs
            return add_costs.invoke({"costs": costs})
        return sum(cost for cost in costs if cost is not None)
    
    def _calculate_total_trip_cost(self, costs_breakdown: Dict[str, float]) -> float:
        """Use LangGraph tool to calculate total trip cost."""
        if self.tool_node:
            from ..tools.langgraph_tools import calculate_total_trip_cost
            return calculate_total_trip_cost.invoke({"costs_breakdown": costs_breakdown})
        return sum(cost for cost in costs_breakdown.values() if cost is not None)
    
    def _convert_cost_breakdown(self, cost_breakdown: Dict[str, float], from_currency: str, to_currency: str) -> Dict[str, float]:
        """Use LangGraph tool to convert cost breakdown."""
        if self.tool_node:
            from ..tools.langgraph_tools import convert_currency
            converted = {}
            for category, amount in cost_breakdown.items():
                result = convert_currency.invoke({"amount": amount, "from_currency": from_currency, "to_currency": to_currency})
                converted[category] = result if isinstance(result, (int, float)) else amount
            return converted
        return cost_breakdown
"""
Attraction Agent - Points of interest discovery and recommendations
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from ..utils.api_clients import FoursquareClient

logger = logging.getLogger(__name__)


class AttractionAgent:
    """
    Reasoning agent for discovering attractions, restaurants, and activities.
    Uses Foursquare API to find points of interest and provides recommendations.
    """
    
    def __init__(self, api_client: Optional[FoursquareClient] = None):
        """
        Initialize attraction agent
        
        Args:
            api_client: Optional FoursquareClient instance for dependency injection
        """
        self.api_client = api_client or FoursquareClient()
        self.search_history: List[Dict[str, Any]] = []
    
    def discover_attractions(self, destination: str, categories: Optional[List[str]] = None, 
                           budget_level: str = "medium") -> Dict[str, Any]:
        """
        Discover attractions and activities for a destination.
        
        Args:
            destination: Destination city name
            categories: Optional list of specific categories to search for
            budget_level: Budget level - "low", "medium", or "high"
            
        Returns:
            Comprehensive attraction recommendations
        """
        if not destination:
            logger.error("Destination is required for attraction discovery")
            return {"error": "Destination is required"}
        
        # Default categories if none provided
        if not categories:
            categories = ["attractions", "restaurants", "activities", "entertainment"]
        
        discovery_result = {
            "destination": destination,
            "budget_level": budget_level,
            "categories_searched": categories,
            "attractions": [],
            "restaurants": [],
            "activities": [],
            "entertainment": [],
            "recommendations": [],
            "discovery_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Search for each category
            for category in categories:
                places = self._search_places_by_category(destination, category, budget_level)
                if places:
                    discovery_result[category] = places
                    logger.info(f"Found {len(places)} {category} in {destination}")
            
            # Generate recommendations based on findings
            discovery_result["recommendations"] = self._generate_attraction_recommendations(
                discovery_result, budget_level
            )
            
            # Record search
            self.search_history.append({
                "destination": destination,
                "categories": categories,
                "budget_level": budget_level,
                "timestamp": datetime.now().isoformat(),
                "results_count": sum(len(discovery_result.get(cat, [])) for cat in categories),
                "success": True
            })
            
            logger.info(f"Completed attraction discovery for {destination}")
            return discovery_result
            
        except Exception as e:
            logger.error(f"Error discovering attractions for {destination}: {e}")
            discovery_result["error"] = str(e)
            
            # Record failed search
            self.search_history.append({
                "destination": destination,
                "categories": categories,
                "budget_level": budget_level,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return discovery_result
    
    def _search_places_by_category(self, destination: str, category: str, 
                                 budget_level: str) -> List[Dict[str, Any]]:
        """Search for places in a specific category."""
        try:
            # Map our categories to Foursquare categories
            category_mapping = {
                "attractions": "10000",  # Arts & Entertainment
                "restaurants": "13000",  # Food & Dining
                "activities": "18000",   # Travel & Transport (includes tours)
                "entertainment": "10000", # Arts & Entertainment
                "museums": "10003",      # Museums
                "parks": "16000",        # Outdoors & Recreation
                "shopping": "17000",     # Retail
                "nightlife": "10032"     # Bars & Nightlife
            }
            
            foursquare_category = category_mapping.get(category, "10000")
            
            # Search places using Foursquare API
            response = self.api_client.search_places(
                near=destination,
                query=category,
                categories=foursquare_category
            )
            
            if not response.success:
                logger.warning(f"API call failed for category {category}: {response.error}")
                return []
            
            places_data = response.data
            
            if not places_data or "results" not in places_data:
                logger.warning(f"No places found for category {category} in {destination}")
                return []
            
            # Process and format results
            places = []
            for place in places_data["results"]:
                place_info = self._format_place_info(place, category, budget_level)
                if place_info:
                    places.append(place_info)
            
            return places
            
        except Exception as e:
            logger.error(f"Failed to search places for category {category}: {e}")
            return []
    
    def _get_fallback_places(self, destination: str, category: str) -> List[Dict[str, Any]]:
        """Provide fallback place data when API is unavailable"""
        fallback_data = {
            "attractions": [
                {"name": f"{destination} City Center", "category": "attractions", "rating": 8.5, "price_level": "free"},
                {"name": f"Historical District", "category": "attractions", "rating": 8.0, "price_level": "free"},
                {"name": f"{destination} Museum", "category": "attractions", "rating": 7.5, "price_level": "medium"}
            ],
            "restaurants": [
                {"name": f"Local {destination} Restaurant", "category": "restaurants", "rating": 8.0, "price_level": "medium"},
                {"name": f"Traditional Cuisine", "category": "restaurants", "rating": 7.8, "price_level": "medium"},
                {"name": f"Street Food Market", "category": "restaurants", "rating": 8.2, "price_level": "low"}
            ],
            "activities": [
                {"name": f"Walking Tour", "category": "activities", "rating": 8.3, "price_level": "low"},
                {"name": f"Local Market Visit", "category": "activities", "rating": 7.9, "price_level": "free"},
                {"name": f"Cultural Experience", "category": "activities", "rating": 8.1, "price_level": "medium"}
            ],
            "entertainment": [
                {"name": f"{destination} Theater", "category": "entertainment", "rating": 8.0, "price_level": "high"},
                {"name": f"Live Music Venue", "category": "entertainment", "rating": 7.7, "price_level": "medium"},
                {"name": f"Local Nightlife", "category": "entertainment", "rating": 7.5, "price_level": "medium"}
            ]
        }
        
        return fallback_data.get(category, [])
    
    def _format_place_info(self, place_data: Dict[str, Any], category: str, 
                          budget_level: str) -> Optional[Dict[str, Any]]:
        """Format place information from Foursquare API response."""
        try:
            place_info = {
                "name": place_data.get("name", "Unknown"),
                "category": category,
                "address": self._format_address(place_data.get("location", {})),
                "rating": place_data.get("rating"),  # Corrected: Rating is already on a 0-10 scale
                "price_level": self._estimate_price_level(place_data, category),
                "description": place_data.get("description", ""),
                "website": place_data.get("website"),
                "phone": place_data.get("tel"),
                "hours": self._format_hours(place_data.get("hours", {})),
                "budget_friendly": self._is_budget_friendly(place_data, budget_level),
                "foursquare_id": place_data.get("fsq_id"),
                "distance": place_data.get("distance"),
                "coordinates": {
                    "lat": place_data.get("geocodes", {}).get("main", {}).get("latitude"),
                    "lng": place_data.get("geocodes", {}).get("main", {}).get("longitude")
                }
            }
            
            return place_info
            
        except Exception as e:
            logger.error(f"Failed to format place info: {e}")
            return None
    
    def _format_address(self, location_data: Dict[str, Any]) -> str:
        """Format address from location data."""
        address_parts = []
        
        if location_data.get("address"):
            address_parts.append(location_data["address"])
        if location_data.get("locality"):
            address_parts.append(location_data["locality"])
        if location_data.get("region"):
            address_parts.append(location_data["region"])
        if location_data.get("country"):
            address_parts.append(location_data["country"])
        
        return ", ".join(address_parts) if address_parts else "Address not available"
    
    def _estimate_price_level(self, place_data: Dict[str, Any], category: str) -> str:
        """Estimate price level based on available data."""
        # Foursquare may provide price tier in some regions
        price_tier = place_data.get("price")
        if price_tier:
            price_mapping = {1: "low", 2: "medium", 3: "high", 4: "very_high"}
            return price_mapping.get(price_tier, "medium")
        
        # Fallback estimation based on category and rating
        rating = place_data.get("rating", 0) / 10 if place_data.get("rating") else 5
        
        if category == "restaurants":
            if rating < 6:
                return "low"
            elif rating < 8:
                return "medium"
            else:
                return "high"
        elif category == "attractions":
            # Most attractions have variable pricing
            return "medium"
        else:
            return "medium"
    
    def _format_hours(self, hours_data: Dict[str, Any]) -> Optional[str]:
        """Format operating hours information."""
        if not hours_data:
            return None
        
        # Foursquare hours format can be complex, simplify for now
        if "display" in hours_data:
            return hours_data["display"]
        elif "regular" in hours_data:
            # Try to format regular hours
            regular_hours = hours_data["regular"]
            if regular_hours:
                # This is a simplified approach - real implementation would need more complex parsing
                return "Hours available - check website"
        
        return None
    
    def _is_budget_friendly(self, place_data: Dict[str, Any], budget_level: str) -> bool:
        """Determine if a place is budget-friendly based on user's budget level."""
        place_price_level = self._estimate_price_level(place_data, "general")
        
        budget_mapping = {
            "low": ["low"],
            "medium": ["low", "medium"],
            "high": ["low", "medium", "high", "very_high"]
        }
        
        return place_price_level in budget_mapping.get(budget_level, ["medium"])
    
    def _generate_attraction_recommendations(self, discovery_data: Dict[str, Any], 
                                           budget_level: str) -> List[str]:
        """Generate personalized recommendations based on discovered places."""
        recommendations = []
        
        # Count places by category
        total_places = sum(len(discovery_data.get(cat, [])) for cat in 
                          ["attractions", "restaurants", "activities", "entertainment"])
        
        if total_places == 0:
            recommendations.append("Limited attraction data available - consider checking local tourism websites")
            return recommendations
        
        # Analyze attractions
        attractions = discovery_data.get("attractions", [])
        if attractions:
            high_rated = [a for a in attractions if a.get("rating", 0) >= 8]
            if high_rated:
                recommendations.append(f"Highly recommended: {high_rated[0]['name']} - top-rated attraction")
            
            budget_friendly = [a for a in attractions if a.get("budget_friendly", False)]
            if budget_friendly and budget_level in ["low", "medium"]:
                recommendations.append(f"Budget-friendly option: {budget_friendly[0]['name']}")
        
        # Analyze restaurants
        restaurants = discovery_data.get("restaurants", [])
        if restaurants:
            high_rated_restaurants = [r for r in restaurants if r.get("rating", 0) >= 8]
            if high_rated_restaurants:
                recommendations.append(f"Must-try dining: {high_rated_restaurants[0]['name']}")
            
            if len(restaurants) >= 3:
                recommendations.append("Good variety of dining options available")
        
        # Analyze activities
        activities = discovery_data.get("activities", [])
        if activities:
            recommendations.append(f"Activities available: {len(activities)} options found")
        
        # General recommendations based on findings
        if total_places >= 10:
            recommendations.append("Rich selection of attractions and activities - plan multiple days")
        elif total_places >= 5:
            recommendations.append("Good selection of places to visit - ideal for weekend trip")
        else:
            recommendations.append("Limited attractions found - consider combining with nearby areas")
        
        # Budget-specific recommendations
        if budget_level == "low":
            free_or_cheap = sum(1 for cat in ["attractions", "activities"] 
                              for place in discovery_data.get(cat, []) 
                              if place.get("budget_friendly", False))
            if free_or_cheap > 0:
                recommendations.append(f"{free_or_cheap} budget-friendly options available")
        
        return recommendations
    
    def search_specific_type(self, destination: str, place_type: str, 
                           preferences: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for specific type of places (e.g., museums, parks, restaurants).
        
        Args:
            destination: Destination city name
            place_type: Specific type of place to search for
            preferences: Optional preferences (rating, price, etc.)
            
        Returns:
            List of places matching the criteria
        """
        try:
            places = self._search_places_by_category(destination, place_type, 
                                                   preferences.get("budget_level", "medium") if preferences else "medium")
            
            # Apply additional filters based on preferences
            if preferences:
                places = self._filter_places_by_preferences(places, preferences)
            
            logger.info(f"Found {len(places)} {place_type} places in {destination}")
            return places
            
        except Exception as e:
            logger.error(f"Error searching for {place_type} in {destination}: {e}")
            return []
    
    def _filter_places_by_preferences(self, places: List[Dict[str, Any]], 
                                    preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter places based on user preferences."""
        filtered_places = places
        
        # Filter by minimum rating
        if preferences.get("min_rating"):
            min_rating = preferences["min_rating"]
            filtered_places = [p for p in filtered_places 
                             if p.get("rating", 0) >= min_rating]
        
        # Filter by budget
        if preferences.get("budget_level"):
            budget_level = preferences["budget_level"]
            filtered_places = [p for p in filtered_places 
                             if self._is_budget_friendly({"price": p.get("price_level")}, budget_level)]
        
        # Limit results
        if preferences.get("limit"):
            filtered_places = filtered_places[:preferences["limit"]]
        
        return filtered_places
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get history of attraction searches performed."""
        return self.search_history.copy()
    
    def clear_history(self):
        """Clear search history."""
        self.search_history.clear()
        logger.info("Attraction search history cleared")
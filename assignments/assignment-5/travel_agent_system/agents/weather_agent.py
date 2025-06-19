"""
Weather Agent - Climate analysis and travel recommendations
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from ..utils.api_clients import OpenWeatherMapClient

logger = logging.getLogger(__name__)


class WeatherAgent:
    """
    Reasoning agent for weather analysis and travel recommendations.
    Analyzes current weather conditions and forecasts to provide travel advice.
    """
    
    def __init__(self, api_client: Optional[OpenWeatherMapClient] = None):
        """
        Initialize weather agent
        
        Args:
            api_client: Optional OpenWeatherMapClient instance for dependency injection
        """
        self.api_client = api_client or OpenWeatherMapClient()
        self.analysis_history: List[Dict[str, Any]] = []
    
    def analyze_weather_for_travel(self, destination: str, travel_dates: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze weather conditions for travel planning.
        
        Args:
            destination: Destination city name
            travel_dates: Optional dict with 'start_date' and 'end_date' keys
            
        Returns:
            Comprehensive weather analysis for travel planning
        """
        if not destination:
            logger.error("Destination is required for weather analysis")
            return {"error": "Destination is required"}
        
        analysis = {
            "destination": destination,
            "current_weather": None,
            "forecast": None,
            "travel_recommendations": [],
            "packing_suggestions": [],
            "weather_alerts": [],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Get current weather
            current_weather = self._get_current_weather(destination)
            if current_weather:
                analysis["current_weather"] = current_weather
                logger.info(f"Retrieved current weather for {destination}")
            
            # Get weather forecast
            forecast = self._get_weather_forecast(destination)
            if forecast:
                analysis["forecast"] = forecast
                logger.info(f"Retrieved weather forecast for {destination}")
            
            # Generate travel recommendations
            if current_weather or forecast:
                analysis["travel_recommendations"] = self._generate_travel_recommendations(
                    current_weather, forecast, travel_dates
                )
                analysis["packing_suggestions"] = self._generate_packing_suggestions(
                    current_weather, forecast
                )
                analysis["weather_alerts"] = self._generate_weather_alerts(
                    current_weather, forecast
                )
            
            # Record analysis
            self.analysis_history.append({
                "destination": destination,
                "travel_dates": travel_dates,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            logger.info(f"Completed weather analysis for {destination}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weather for {destination}: {e}")
            analysis["error"] = str(e)
            
            # Record failed analysis
            self.analysis_history.append({
                "destination": destination,
                "travel_dates": travel_dates,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return analysis
    
    def _get_current_weather(self, destination: str) -> Optional[Dict[str, Any]]:
        """Get current weather conditions."""
        try:
            weather_data = self.api_client.get_current_weather(destination)
            if not weather_data:
                return None
            
            # Extract and format relevant weather information
            current = {
                "temperature": weather_data.get("main", {}).get("temp"),
                "feels_like": weather_data.get("main", {}).get("feels_like"),
                "humidity": weather_data.get("main", {}).get("humidity"),
                "pressure": weather_data.get("main", {}).get("pressure"),
                "description": weather_data.get("weather", [{}])[0].get("description", ""),
                "main": weather_data.get("weather", [{}])[0].get("main", ""),
                "wind_speed": weather_data.get("wind", {}).get("speed"),
                "wind_direction": weather_data.get("wind", {}).get("deg"),
                "visibility": weather_data.get("visibility"),
                "clouds": weather_data.get("clouds", {}).get("all"),
                "timestamp": datetime.now().isoformat()
            }
            return current
            
        except Exception as e:
            logger.error(f"Failed to get current weather for {destination}: {e}")
            return None
    
    def _get_weather_forecast(self, destination: str, days: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Get weather forecast for specified number of days."""
        try:
            forecast_data = self.api_client.get_weather_forecast(destination)
            if not forecast_data or "list" not in forecast_data:
                return None
            
            # Process forecast data (API returns 3-hour intervals for 5 days)
            forecast_list = []
            processed_dates = set()
            
            for item in forecast_data["list"]:
                # Get date for this forecast item
                dt_txt = item.get("dt_txt", "")
                if not dt_txt:
                    continue
                
                forecast_date = dt_txt.split(" ")[0]  # Extract date part
                
                # Skip if we've already processed this date (take first forecast of the day)
                if forecast_date in processed_dates:
                    continue
                
                processed_dates.add(forecast_date)
                
                forecast_item = {
                    "date": forecast_date,
                    "temperature": {
                        "min": item.get("main", {}).get("temp_min"),
                        "max": item.get("main", {}).get("temp_max"),
                        "avg": item.get("main", {}).get("temp")
                    },
                    "humidity": item.get("main", {}).get("humidity"),
                    "description": item.get("weather", [{}])[0].get("description", ""),
                    "main": item.get("weather", [{}])[0].get("main", ""),
                    "wind_speed": item.get("wind", {}).get("speed"),
                    "clouds": item.get("clouds", {}).get("all"),
                    "pop": item.get("pop", 0)  # Probability of precipitation
                }
                
                forecast_list.append(forecast_item)
                
                # Limit to requested number of days
                if len(forecast_list) >= days:
                    break
            
            return forecast_list
            
        except Exception as e:
            logger.error(f"Failed to get weather forecast for {destination}: {e}")
            return None
    
    def _generate_travel_recommendations(self, current_weather: Optional[Dict], 
                                       forecast: Optional[List[Dict]], 
                                       travel_dates: Optional[Dict[str, str]]) -> List[str]:
        """Generate travel recommendations based on weather conditions."""
        recommendations = []
        
        if not current_weather and not forecast:
            return ["Weather data unavailable - check local forecasts before traveling"]
        
        # Analyze current conditions
        if current_weather:
            temp = current_weather.get("temperature")
            description = current_weather.get("description", "").lower()
            main = current_weather.get("main", "").lower()
            wind_speed = current_weather.get("wind_speed", 0)
            
            if temp is not None:
                if temp < 0:
                    recommendations.append("Very cold conditions - plan for winter activities and warm clothing")
                elif temp < 10:
                    recommendations.append("Cold weather - ideal for indoor attractions and cozy activities")
                elif temp > 30:
                    recommendations.append("Hot weather - stay hydrated and plan indoor activities during peak hours")
                elif 15 <= temp <= 25:
                    recommendations.append("Pleasant temperature - great for outdoor sightseeing and walking tours")
            
            if "rain" in description or "rain" in main:
                recommendations.append("Rainy conditions expected - plan indoor activities and carry an umbrella")
            elif "clear" in description or "clear" in main:
                recommendations.append("Clear skies - perfect for outdoor activities and photography")
            elif "snow" in description or "snow" in main:
                recommendations.append("Snowy conditions - check transportation and consider winter sports")
            
            if wind_speed and wind_speed > 10:  # m/s
                recommendations.append("Windy conditions - be cautious with outdoor activities")
        
        # Analyze forecast trends
        if forecast and len(forecast) > 1:
            temps = [f.get("temperature", {}).get("avg") for f in forecast if f.get("temperature", {}).get("avg")]
            if temps:
                avg_temp = sum(temps) / len(temps)
                temp_range = max(temps) - min(temps)
                
                if temp_range > 15:
                    recommendations.append("Variable temperatures expected - pack layers for different conditions")
                
                rainy_days = sum(1 for f in forecast if "rain" in f.get("description", "").lower())
                if rainy_days > len(forecast) // 2:
                    recommendations.append("Several rainy days forecasted - plan indoor alternatives")
        
        return recommendations
    
    def _generate_packing_suggestions(self, current_weather: Optional[Dict], 
                                    forecast: Optional[List[Dict]]) -> List[str]:
        """Generate packing suggestions based on weather conditions."""
        suggestions = []
        
        if not current_weather and not forecast:
            return ["Check local weather forecasts for packing guidance"]
        
        # Analyze temperature ranges
        temps = []
        if current_weather and current_weather.get("temperature"):
            temps.append(current_weather["temperature"])
        
        if forecast:
            for f in forecast:
                temp_data = f.get("temperature", {})
                if temp_data.get("min"):
                    temps.append(temp_data["min"])
                if temp_data.get("max"):
                    temps.append(temp_data["max"])
        
        if temps:
            min_temp = min(temps)
            max_temp = max(temps)
            
            if min_temp < 0:
                suggestions.extend([
                    "Heavy winter coat and thermal layers",
                    "Insulated boots and warm socks",
                    "Gloves, hat, and scarf"
                ])
            elif min_temp < 10:
                suggestions.extend([
                    "Warm jacket or coat",
                    "Long pants and sweaters",
                    "Closed-toe shoes"
                ])
            
            if max_temp > 25:
                suggestions.extend([
                    "Light, breathable clothing",
                    "Sunscreen and sunglasses",
                    "Hat for sun protection"
                ])
            
            if max_temp - min_temp > 15:
                suggestions.append("Layered clothing for temperature changes")
        
        # Check for precipitation
        rainy_conditions = False
        if current_weather and "rain" in current_weather.get("description", "").lower():
            rainy_conditions = True
        
        if forecast:
            for f in forecast:
                if "rain" in f.get("description", "").lower() or f.get("pop", 0) > 0.3:
                    rainy_conditions = True
                    break
        
        if rainy_conditions:
            suggestions.extend([
                "Waterproof jacket or raincoat",
                "Umbrella",
                "Water-resistant shoes"
            ])
        
        return suggestions
    
    def _generate_weather_alerts(self, current_weather: Optional[Dict], 
                               forecast: Optional[List[Dict]]) -> List[str]:
        """Generate weather alerts and warnings."""
        alerts = []
        
        # Check current conditions for alerts
        if current_weather:
            temp = current_weather.get("temperature")
            main = current_weather.get("main", "").lower()
            wind_speed = current_weather.get("wind_speed", 0)
            
            if temp is not None:
                if temp < -10:
                    alerts.append("ALERT: Extreme cold conditions - risk of frostbite")
                elif temp > 35:
                    alerts.append("ALERT: Very hot conditions - heat exhaustion risk")
            
            if wind_speed and wind_speed > 15:  # m/s (about 54 km/h)
                alerts.append("ALERT: Strong winds - outdoor activities may be dangerous")
            
            if "thunderstorm" in main:
                alerts.append("ALERT: Thunderstorms - stay indoors and avoid outdoor activities")
        
        # Check forecast for potential issues
        if forecast:
            for f in forecast:
                main = f.get("main", "").lower()
                if "thunderstorm" in main:
                    alerts.append(f"WARNING: Thunderstorms expected on {f.get('date')}")
                elif "snow" in main:
                    alerts.append(f"WARNING: Snow expected on {f.get('date')} - check transportation")
        
        return alerts
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of weather analyses performed."""
        return self.analysis_history.copy()
    
    def clear_history(self):
        """Clear analysis history."""
        self.analysis_history.clear()
        logger.info("Weather analysis history cleared")
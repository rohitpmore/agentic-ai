"""
Summary Agent - LLM-powered trip summary generation
"""

from typing import Dict, Any, Optional, List
import logging
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from config import config
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    Reasoning agent for generating comprehensive trip summaries using LLM.
    Creates natural, readable summaries from structured travel data.
    """
    
    def __init__(self, tool_node: Optional[ToolNode] = None):
        """Initialize summary agent with Gemini LLM and LangGraph tools"""
        # Configure Gemini if available
        if GEMINI_AVAILABLE and config.gemini_api_key and config.gemini_api_key != "fallback_gemini_key":
            genai.configure(api_key=config.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.llm_available = True
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini library not available. LLM summary generation disabled.")
            else:
                logger.warning("Gemini API key not configured. LLM summary generation disabled.")
            self.llm_available = False
        
        # Store tool node for LangGraph tools integration
        self.tool_node = tool_node
    
    def generate_trip_summary(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trip summary using LLM with LangGraph tools.
        Implements Agent→Tools→Agent circular pattern.
        
        Args:
            trip_data: Combined data from all travel agents
            
        Returns:
            Dict containing LLM-generated summary sections
        """
        if not trip_data:
            logger.error("Trip data is required for summary generation")
            return {"error": "Trip data is required"}
        
        try:
            # Use LangGraph tools for data processing (Agent→Tools)
            processed_data = self._process_data_with_tools(trip_data)
            
            # Generate different summary sections using LLM + processed data (Tools→Agent)
            summary_result = {
                "destination": processed_data.get("destination", "Unknown"),
                "executive_summary": self._generate_executive_summary(trip_data, processed_data),
                "overview": processed_data.get("trip_statistics", {}),
                "highlights": processed_data.get("highlights", []),
                "cost_breakdown": processed_data.get("cost_breakdown", {}),
                "detailed_description": self._generate_detailed_description(trip_data, processed_data)
            }
            
            logger.info(f"Successfully generated LLM summary for {processed_data.get('destination', 'Unknown')}")
            return summary_result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {"error": f"Summary generation failed: {str(e)}"}
    
    def _process_data_with_tools(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process trip data using LangGraph tools (Agent→Tools phase).
        
        Args:
            trip_data: Raw trip data from agents
            
        Returns:
            Dict with processed data from tools
        """
        processed_data = {}
        
        try:
            # Import tools directly since we may not have tool_node
            from ..tools.langgraph_tools import (
                extract_trip_highlights, 
                calculate_trip_statistics,
                format_cost_breakdown_display
            )
            
            # Use tools to process data
            processed_data["highlights"] = extract_trip_highlights.func(trip_data)
            processed_data["trip_statistics"] = calculate_trip_statistics.func(trip_data)
            
            # Process cost breakdown if available
            itinerary_data = trip_data.get("itinerary_data", {})
            if itinerary_data and itinerary_data.get("cost_breakdown"):
                processed_data["cost_breakdown"] = format_cost_breakdown_display.func(
                    itinerary_data["cost_breakdown"]
                )
            else:
                processed_data["cost_breakdown"] = {"categories": {}}
            
            processed_data["destination"] = trip_data.get("destination", "Unknown")
            
            logger.info("Successfully processed trip data using LangGraph tools")
            return processed_data
            
        except Exception as e:
            logger.error(f"Tool processing failed: {str(e)}")
            # Fallback to raw data
            return {
                "destination": trip_data.get("destination", "Unknown"),
                "highlights": [],
                "trip_statistics": {},
                "cost_breakdown": {"categories": {}}
            }
    
    def _generate_executive_summary(self, trip_data: Dict[str, Any], processed_data: Dict[str, Any]) -> str:
        """Generate executive summary using LLM with processed data"""
        if not self.llm_available:
            return self._fallback_executive_summary(trip_data)
            
        try:
            prompt = self._build_executive_summary_prompt(trip_data, processed_data)
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Executive summary generation failed: {str(e)}")
            return self._fallback_executive_summary(trip_data)
    
    def _generate_overview(self, itinerary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overview section with cost calculations"""
        overview = {}
        
        if itinerary_data:
            total_cost = itinerary_data.get("total_cost", 0)
            total_days = itinerary_data.get("total_days", 0)
            daily_average = total_cost / max(total_days, 1) if total_days > 0 else 0
            
            overview = {
                "total_days": total_days,
                "total_cost": total_cost,
                "daily_average": daily_average,
                "currency": itinerary_data.get("currency", "USD")
            }
        
        return overview
    
    def _generate_highlights(self, trip_data: Dict[str, Any]) -> List[str]:
        """Generate trip highlights using LLM"""
        if not self.llm_available:
            return self._fallback_highlights(trip_data)
            
        try:
            prompt = self._build_highlights_prompt(trip_data)
            response = self.model.generate_content(prompt)
            
            # Parse LLM response into list
            highlights = []
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    # Remove bullet point markers
                    highlight = line.lstrip('•-* ').strip()
                    if highlight:
                        highlights.append(highlight)
            
            return highlights[:6] if highlights else self._fallback_highlights(trip_data)
            
        except Exception as e:
            logger.error(f"Highlights generation failed: {str(e)}")
            return self._fallback_highlights(trip_data)
    
    def _generate_cost_breakdown(self, itinerary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost breakdown section"""
        cost_breakdown = {"categories": {}}
        
        if itinerary_data and itinerary_data.get("cost_breakdown"):
            breakdown_data = itinerary_data["cost_breakdown"]
            total = sum(breakdown_data.values()) if breakdown_data.values() else 1
            
            for category, amount in breakdown_data.items():
                percentage = (amount / total * 100) if total > 0 else 0
                cost_breakdown["categories"][category] = {
                    "formatted": f"${amount:.2f}",
                    "percentage": f"{percentage:.1f}"
                }
        
        return cost_breakdown
    
    def _generate_detailed_description(self, trip_data: Dict[str, Any], processed_data: Dict[str, Any]) -> str:
        """Generate detailed trip description using LLM with processed data"""
        if not self.llm_available:
            return self._fallback_detailed_description(trip_data)
            
        try:
            prompt = self._build_detailed_description_prompt(trip_data, processed_data)
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Detailed description generation failed: {str(e)}")
            return self._fallback_detailed_description(trip_data)
    
    def _build_executive_summary_prompt(self, trip_data: Dict[str, Any], processed_data: Dict[str, Any]) -> str:
        """Build prompt for executive summary generation using processed data"""
        destination = processed_data.get("destination", "Unknown")
        stats = processed_data.get("trip_statistics", {})
        highlights = processed_data.get("highlights", [])
        
        prompt = f"""Create a compelling executive summary for a trip to {destination}.

Trip Statistics (from LangGraph tools):
- Duration: {stats.get('total_days', 'Unknown')} days
- Budget: ${stats.get('total_cost', 0):.2f}
- Daily Average: ${stats.get('daily_average', 0):.2f}
- Hotels: {stats.get('hotel_count', 0)} options found
- Attractions: {stats.get('attraction_count', 0)} attractions discovered
- Weather Analysis: {'Available' if stats.get('weather_available') else 'Not available'}

Key Highlights:
{chr(10).join([f"- {highlight}" for highlight in highlights[:4]])}

Write a concise, engaging 2-3 sentence summary that captures the essence of this trip experience. Focus on value, experiences, and what makes this trip special. Use the statistics and highlights to create compelling copy. Do not use bullet points."""
        
        return prompt
    
    def _build_highlights_prompt(self, trip_data: Dict[str, Any]) -> str:
        """Build prompt for highlights generation"""
        destination = trip_data.get("destination", "Unknown")
        hotels_data = trip_data.get("hotels_data", {})
        attractions_data = trip_data.get("attractions_data", {})
        weather_data = trip_data.get("weather_data", {})
        itinerary_data = trip_data.get("itinerary_data", {})
        
        prompt = f"""Generate 4-6 compelling trip highlights for {destination}.

Available Data:
Hotels: {hotels_data.get('hotels', [])[:2] if hotels_data else []}
Top Attractions: {[a.get('name', 'Unknown') for a in attractions_data.get('attractions', [])[:3]] if attractions_data else []}
Weather: {weather_data.get('forecast', [{}])[0].get('description', 'Unknown') if weather_data else 'Unknown'}
Total Budget: ${itinerary_data.get('total_cost', 0):.2f}

Create highlights that include:
- Top hotel recommendations (if available)
- Must-visit attractions  
- Weather insights
- Budget summary
- Unique experiences

Format as bullet points starting with • or -. Keep each highlight concise but engaging."""
        
        return prompt
    
    def _build_detailed_description_prompt(self, trip_data: Dict[str, Any], processed_data: Dict[str, Any]) -> str:
        """Build prompt for detailed description generation using processed data"""
        destination = processed_data.get("destination", "Unknown")
        stats = processed_data.get("trip_statistics", {})
        highlights = processed_data.get("highlights", [])
        
        prompt = f"""Write a detailed, engaging description of this {destination} trip experience.

Trip Statistics:
- {stats.get('total_days', 3)} days total
- ${stats.get('total_cost', 0):.2f} total budget (${stats.get('daily_average', 0):.2f}/day)
- {stats.get('hotel_count', 0)} accommodation options
- {stats.get('attraction_count', 0)} attractions and activities
- Weather analysis: {'Available' if stats.get('weather_available') else 'General planning'}

Top Highlights:
{chr(10).join([f"• {highlight}" for highlight in highlights[:5]])}

Write in an enthusiastic but professional tone, as if recommending this trip to a friend. Include information about:
- The destination's unique character and what makes it special
- Value proposition (cost vs. experiences)
- Variety of accommodation and activity options
- Overall trip experience and benefits

Keep it informative yet engaging, around 100-150 words. Use the statistics and highlights to create compelling narrative."""
        
        return prompt
    
    def _fallback_executive_summary(self, trip_data: Dict[str, Any]) -> str:
        """Fallback executive summary when LLM is unavailable"""
        destination = trip_data.get("destination", "Unknown")
        itinerary_data = trip_data.get("itinerary_data", {})
        
        summary = f"A {itinerary_data.get('total_days', 3)}-day trip to {destination}"
        
        if itinerary_data.get("total_cost"):
            summary += f" with a total budget of ${itinerary_data['total_cost']:.2f}"
        
        summary += ". Includes accommodation, activities, meals, and transportation with comprehensive planning."
        
        return summary
    
    def _fallback_highlights(self, trip_data: Dict[str, Any]) -> List[str]:
        """Fallback highlights when LLM is unavailable"""
        highlights = []
        
        # Add attractions
        attractions_data = trip_data.get("attractions_data", {})
        if attractions_data and attractions_data.get("attractions"):
            for attraction in attractions_data["attractions"][:2]:
                name = attraction.get("name", "Attraction")
                highlights.append(f"Visit {name}")
        
        # Add hotel info
        hotels_data = trip_data.get("hotels_data", {})
        if hotels_data and hotels_data.get("hotels"):
            hotel_count = len(hotels_data["hotels"])
            highlights.append(f"{hotel_count} carefully selected accommodation options")
        
        # Add weather
        weather_data = trip_data.get("weather_data", {})
        if weather_data and weather_data.get("forecast"):
            highlights.append("Weather-optimized daily planning")
        
        # Add budget
        itinerary_data = trip_data.get("itinerary_data", {})
        if itinerary_data and itinerary_data.get("total_cost"):
            highlights.append(f"Total budget: ${itinerary_data['total_cost']:.2f}")
        
        return highlights
    
    def _fallback_detailed_description(self, trip_data: Dict[str, Any]) -> str:
        """Fallback detailed description when LLM is unavailable"""
        destination = trip_data.get("destination", "Unknown")
        return f"Experience the best of {destination} with this carefully curated travel package. This comprehensive trip includes professionally selected accommodations, must-see attractions, and weather-optimized planning to ensure you make the most of your time and budget."
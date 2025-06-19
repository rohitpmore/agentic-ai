"""
Travel Planner Workflow - Parallel orchestration using LangGraph
"""

from typing import Dict, Any, Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .state import TravelPlanState
from ..agents.weather_agent import WeatherAgent
from ..agents.attraction_agent import AttractionAgent
from ..agents.hotel_agent import HotelAgent
from ..agents.itinerary_agent import ItineraryAgent

logger = logging.getLogger(__name__)


class TravelPlannerWorkflow:
    """
    Main workflow orchestration class for travel planning.
    Implements parallel execution of agents for optimal performance.
    """
    
    def __init__(self):
        """Initialize workflow with agents"""
        self.weather_agent = WeatherAgent()
        self.attraction_agent = AttractionAgent()
        self.hotel_agent = HotelAgent()
        self.itinerary_agent = ItineraryAgent()
        
        # Thread pool for parallel execution
        self.max_workers = 3  # Weather, Attractions, Hotels run in parallel
        
    def process_travel_request(self, travel_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete travel planning request with parallel execution.
        
        Args:
            travel_request: Dictionary containing destination, dates, budget, etc.
            
        Returns:
            Complete travel plan with itinerary and cost breakdown
        """
        logger.info(f"Starting travel planning for {travel_request.get('destination')}")
        
        # Initialize state
        state = TravelPlanState(
            destination=travel_request.get("destination"),
            travel_dates=travel_request.get("travel_dates"),
            budget=travel_request.get("budget"),
            currency=travel_request.get("currency", "USD"),
            preferences=travel_request.get("preferences", {})
        )
        
        try:
            # Validate input
            validation_errors = state.validate_input()
            if validation_errors:
                state.add_error(f"Validation failed: {', '.join(validation_errors)}")
                return self._create_error_response(state, validation_errors)
            
            # Phase 1: Parallel data gathering (Weather, Attractions, Hotels)
            self._execute_parallel_agents(state)
            
            # Phase 2: Check if we have sufficient data for itinerary
            if not state.has_sufficient_data_for_itinerary():
                error_msg = "Insufficient data to create itinerary"
                state.add_error(error_msg)
                return self._create_error_response(state, [error_msg])
            
            # Phase 3: Generate itinerary using collected data
            self._execute_itinerary_agent(state)
            
            # Mark workflow complete
            state.mark_complete()
            
            # Return final result
            return self._create_success_response(state)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state.add_error(f"Workflow error: {str(e)}")
            return self._create_error_response(state, [str(e)])
    
    def _execute_parallel_agents(self, state: TravelPlanState):
        """Execute weather, attraction, and hotel agents in parallel"""
        logger.info("Starting parallel agent execution")
        
        # Create tasks for parallel execution
        tasks = [
            ("weather", self._execute_weather_agent, state),
            ("attractions", self._execute_attraction_agent, state),
            ("hotels", self._execute_hotel_agent, state)
        ]
        
        # Execute agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(task_func, task_state): agent_name 
                for agent_name, task_func, task_state in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                
                try:
                    result = future.result()
                    if result and not result.get("error"):
                        state.mark_agent_complete(agent_name, result)
                        logger.info(f"Agent {agent_name} completed successfully")
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "No result returned"
                        state.mark_agent_error(agent_name, error_msg)
                        
                except Exception as e:
                    state.mark_agent_error(agent_name, str(e))
                    logger.error(f"Agent {agent_name} failed with exception: {e}")
        
        logger.info("Parallel agent execution completed")
    
    def _execute_weather_agent(self, state: TravelPlanState) -> Optional[Dict[str, Any]]:
        """Execute weather agent"""
        state.mark_agent_processing("weather")
        
        try:
            result = self.weather_agent.analyze_weather_for_travel(
                destination=state.destination,
                travel_dates=state.travel_dates
            )
            
            if result.get("error"):
                logger.warning(f"Weather agent returned error: {result['error']}")
                return result
            
            logger.info("Weather agent completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Weather agent execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_attraction_agent(self, state: TravelPlanState) -> Optional[Dict[str, Any]]:
        """Execute attraction agent"""
        state.mark_agent_processing("attractions")
        
        try:
            # Determine budget level for attractions
            budget_level = "medium"  # Default
            if state.preferences and state.preferences.get("budget_level"):
                budget_level = state.preferences["budget_level"]
            elif state.budget:
                if state.budget <= 500:
                    budget_level = "low"
                elif state.budget >= 2000:
                    budget_level = "high"
            
            result = self.attraction_agent.discover_attractions(
                destination=state.destination,
                categories=["attractions", "restaurants", "activities", "entertainment"],
                budget_level=budget_level
            )
            
            if result.get("error"):
                logger.warning(f"Attraction agent returned error: {result['error']}")
                return result
            
            logger.info("Attraction agent completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Attraction agent execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_hotel_agent(self, state: TravelPlanState) -> Optional[Dict[str, Any]]:
        """Execute hotel agent"""
        state.mark_agent_processing("hotels")
        
        try:
            # Create budget range if budget is specified
            budget_range = None
            if state.budget and state.travel_dates:
                # Estimate nights
                nights = 3  # Default
                try:
                    from datetime import datetime
                    start = datetime.fromisoformat(state.travel_dates["start_date"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(state.travel_dates["end_date"].replace("Z", "+00:00"))
                    nights = max(1, (end - start).days)
                except Exception:
                    pass
                
                # Allocate ~40% of budget to accommodation
                hotel_budget = state.budget * 0.4
                daily_hotel_budget = hotel_budget / nights
                
                budget_range = {
                    "min": daily_hotel_budget * 0.7,
                    "max": daily_hotel_budget * 1.3
                }
            
            result = self.hotel_agent.search_hotels(
                destination=state.destination,
                budget_range=budget_range,
                travel_dates=state.travel_dates,
                preferences=state.preferences
            )
            
            if result.get("error"):
                logger.warning(f"Hotel agent returned error: {result['error']}")
                return result
            
            logger.info("Hotel agent completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Hotel agent execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_itinerary_agent(self, state: TravelPlanState):
        """Execute itinerary agent using aggregated data"""
        logger.info("Starting itinerary generation")
        
        state.mark_agent_processing("itinerary")
        
        try:
            # Get aggregated data for itinerary creation
            aggregated_data = state.get_aggregated_data()
            
            result = self.itinerary_agent.create_itinerary(
                trip_data=aggregated_data,
                preferences=state.preferences
            )
            
            if result.get("error"):
                state.mark_agent_error("itinerary", result["error"])
                logger.error(f"Itinerary agent failed: {result['error']}")
            else:
                state.mark_agent_complete("itinerary", result)
                logger.info("Itinerary agent completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            state.mark_agent_error("itinerary", error_msg)
            logger.error(f"Itinerary agent execution failed: {e}")
    
    def _create_success_response(self, state: TravelPlanState) -> Dict[str, Any]:
        """Create successful response from completed state"""
        # Generate comprehensive trip summary
        trip_summary = self._generate_comprehensive_summary(state)
        
        return {
            "status": "success",
            "destination": state.destination,
            "travel_plan": state.itinerary_data,
            "trip_summary": trip_summary,
            "data_sources": {
                "weather": state.weather_data,
                "attractions": state.attractions_data,
                "hotels": state.hotels_data
            },
            "processing_summary": state.get_processing_summary(),
            "errors": state.errors if state.errors else []
        }
    
    def _create_error_response(self, state: TravelPlanState, errors: list) -> Dict[str, Any]:
        """Create error response from failed state"""
        return {
            "status": "error",
            "destination": state.destination,
            "errors": errors,
            "partial_data": {
                "weather": state.weather_data,
                "attractions": state.attractions_data,
                "hotels": state.hotels_data
            },
            "processing_summary": state.get_processing_summary()
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process simple travel planning query.
        
        Args:
            question: Natural language travel planning question
            
        Returns:
            Travel plan response
        """
        # Simple query parser - extract destination from question
        # In a real implementation, this would use LLM for parsing
        destination = self._extract_destination_from_query(question)
        
        if not destination:
            return {
                "status": "error",
                "errors": ["Could not extract destination from query"],
                "question": question
            }
        
        # Create basic travel request
        travel_request = {
            "destination": destination,
            "preferences": {"pace": "moderate"}
        }
        
        # Process the request
        return self.process_travel_request(travel_request)
    
    def _extract_destination_from_query(self, question: str) -> Optional[str]:
        """Extract destination from natural language query"""
        # Simple extraction - look for common patterns
        question_lower = question.lower()
        
        # Common patterns: "trip to X", "visit X", "plan X", "travel to X"
        patterns = [
            "trip to ", "visit ", "travel to ", "go to ", "plan a trip to ",
            "vacation in ", "holiday in ", "tour of "
        ]
        
        for pattern in patterns:
            if pattern in question_lower:
                start_idx = question_lower.find(pattern) + len(pattern)
                # Extract the next word(s) as destination
                remaining = question[start_idx:].strip()
                # Take up to 3 words as destination
                words = remaining.split()[:3]
                if words:
                    return " ".join(words).rstrip(".,!?")
        
        # Fallback - return None if no pattern found
        return None
    
    def _generate_comprehensive_summary(self, state: TravelPlanState) -> Dict[str, Any]:
        """Generate comprehensive trip summary with all details"""
        summary = {
            "destination": state.destination,
            "overview": {},
            "cost_breakdown": {},
            "highlights": [],
            "recommendations": [],
            "practical_info": {}
        }
        
        try:
            # Extract key information from itinerary
            if state.itinerary_data:
                itinerary = state.itinerary_data
                
                # Overview
                summary["overview"] = {
                    "total_days": itinerary.get("total_days", 0),
                    "trip_dates": itinerary.get("trip_dates"),
                    "total_cost": itinerary.get("total_cost", 0),
                    "currency": state.currency or "USD",
                    "daily_average": round(itinerary.get("total_cost", 0) / max(itinerary.get("total_days", 1), 1), 2)
                }
                
                # Cost breakdown with detailed formatting
                cost_breakdown = itinerary.get("cost_breakdown", {})
                summary["cost_breakdown"] = self._format_cost_breakdown(cost_breakdown, state.currency or "USD")
                
                # Extract highlights from daily plans
                summary["highlights"] = self._extract_trip_highlights(itinerary)
                
                # Combine recommendations from all sources
                summary["recommendations"] = self._aggregate_recommendations(state)
                
                # Practical information
                summary["practical_info"] = self._generate_practical_info(state)
            
            # Executive summary
            summary["executive_summary"] = self._generate_executive_summary(summary, state)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive summary: {e}")
            return {
                "destination": state.destination,
                "error": f"Summary generation failed: {str(e)}"
            }
    
    def _format_cost_breakdown(self, cost_breakdown: Dict[str, float], currency: str) -> Dict[str, Any]:
        """Format cost breakdown with detailed information"""
        formatted = {
            "currency": currency,
            "categories": {},
            "total": sum(cost_breakdown.values()),
            "percentages": {}
        }
        
        total = sum(cost_breakdown.values())
        
        for category, amount in cost_breakdown.items():
            formatted["categories"][category] = {
                "amount": round(amount, 2),
                "percentage": round((amount / total * 100) if total > 0 else 0, 1),
                "formatted": f"{currency} {amount:.2f}"
            }
            formatted["percentages"][category] = round((amount / total * 100) if total > 0 else 0, 1)
        
        formatted["total_formatted"] = f"{currency} {total:.2f}"
        
        return formatted
    
    def _extract_trip_highlights(self, itinerary: Dict[str, Any]) -> List[str]:
        """Extract key highlights from the itinerary"""
        highlights = []
        
        try:
            daily_plans = itinerary.get("daily_plans", [])
            
            # Collect unique activities and top-rated places
            all_activities = []
            for day_plan in daily_plans:
                for period in ["morning", "afternoon", "evening"]:
                    activities = day_plan.get(period, [])
                    all_activities.extend(activities)
            
            # Extract top highlights
            unique_activities = []
            seen_names = set()
            
            for activity in all_activities:
                name = activity.get("name", "")
                if name and name not in seen_names:
                    unique_activities.append(activity)
                    seen_names.add(name)
            
            # Sort by rating if available, take top 5
            rated_activities = [a for a in unique_activities if a.get("rating")]
            if rated_activities:
                rated_activities.sort(key=lambda x: x.get("rating", 0), reverse=True)
                highlights.extend([f"{a['name']} (Rating: {a['rating']}/10)" for a in rated_activities[:3]])
            
            # Add non-rated activities
            other_activities = [a for a in unique_activities if not a.get("rating")]
            highlights.extend([a["name"] for a in other_activities[:2]])
            
            return highlights[:5]  # Limit to 5 highlights
            
        except Exception as e:
            logger.warning(f"Could not extract highlights: {e}")
            return ["Explore the destination", "Discover local culture", "Enjoy local cuisine"]
    
    def _aggregate_recommendations(self, state: TravelPlanState) -> List[str]:
        """Aggregate recommendations from all data sources"""
        all_recommendations = []
        
        # From weather data
        if state.weather_data and state.weather_data.get("travel_recommendations"):
            all_recommendations.extend(state.weather_data["travel_recommendations"][:2])
        
        # From attractions data
        if state.attractions_data and state.attractions_data.get("recommendations"):
            all_recommendations.extend(state.attractions_data["recommendations"][:2])
        
        # From hotels data
        if state.hotels_data and state.hotels_data.get("recommendations"):
            all_recommendations.extend(state.hotels_data["recommendations"][:2])
        
        # From itinerary
        if state.itinerary_data and state.itinerary_data.get("recommendations"):
            all_recommendations.extend(state.itinerary_data["recommendations"][:3])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:8]  # Limit to 8 recommendations
    
    def _generate_practical_info(self, state: TravelPlanState) -> Dict[str, Any]:
        """Generate practical travel information"""
        practical = {
            "packing_list": [],
            "important_notes": [],
            "weather_summary": None,
            "budget_tips": []
        }
        
        try:
            # From itinerary
            if state.itinerary_data:
                practical["packing_list"] = state.itinerary_data.get("packing_list", [])[:10]
                practical["important_notes"] = state.itinerary_data.get("important_notes", [])[:5]
            
            # Weather summary
            if state.weather_data:
                practical["weather_summary"] = self._summarize_weather(state.weather_data)
            
            # Budget tips based on cost breakdown
            if state.itinerary_data and state.itinerary_data.get("cost_breakdown"):
                practical["budget_tips"] = self._generate_budget_tips(state.itinerary_data["cost_breakdown"])
            
            return practical
            
        except Exception as e:
            logger.warning(f"Could not generate practical info: {e}")
            return practical
    
    def _summarize_weather(self, weather_data: Dict[str, Any]) -> str:
        """Create a concise weather summary"""
        try:
            current = weather_data.get("current_weather", {})
            forecast = weather_data.get("forecast", [])
            
            if current:
                temp = current.get("temperature", "Unknown")
                desc = current.get("description", "")
                summary = f"Current: {temp}°C, {desc.title()}"
                
                if forecast:
                    temps = [f.get("temperature", {}).get("avg") for f in forecast if f.get("temperature", {}).get("avg")]
                    if temps:
                        avg_temp = sum(temps) / len(temps)
                        summary += f". Forecast: {avg_temp:.0f}°C average"
                
                return summary
            
            return "Weather information available"
            
        except Exception:
            return "Check local weather before travel"
    
    def _generate_budget_tips(self, cost_breakdown: Dict[str, float]) -> List[str]:
        """Generate budget optimization tips"""
        tips = []
        
        try:
            total = sum(cost_breakdown.values())
            
            # Accommodation tips
            accommodation = cost_breakdown.get("accommodation", 0)
            if accommodation / total > 0.5:
                tips.append("Consider alternative accommodations to reduce costs")
            
            # Activities tips
            activities = cost_breakdown.get("activities", 0)
            if activities / total > 0.3:
                tips.append("Look for free walking tours and public attractions")
            
            # Meals tips
            meals = cost_breakdown.get("meals", 0)
            if meals / total > 0.25:
                tips.append("Try local markets and street food for authentic, affordable meals")
            
            # General tips
            tips.extend([
                "Book attractions in advance for potential discounts",
                "Use public transportation to save on travel costs"
            ])
            
            return tips[:4]  # Limit to 4 tips
            
        except Exception:
            return ["Plan ahead for better prices", "Look for local deals and discounts"]
    
    def _generate_executive_summary(self, summary: Dict[str, Any], state: TravelPlanState) -> str:
        """Generate a concise executive summary of the trip"""
        try:
            destination = state.destination
            overview = summary.get("overview", {})
            days = overview.get("total_days", 0)
            cost = overview.get("total_cost", 0)
            currency = overview.get("currency", "USD")
            
            # Build summary text
            exec_summary = f"A {days}-day trip to {destination} with an estimated total cost of {currency} {cost:.0f}. "
            
            # Add highlights
            highlights = summary.get("highlights", [])
            if highlights:
                exec_summary += f"Key highlights include {', '.join(highlights[:2])}. "
            
            # Add practical note
            successful_sources = state.get_successful_data_sources()
            if len(successful_sources) >= 2:
                exec_summary += f"Plan includes comprehensive information from {len(successful_sources)} data sources. "
            
            # Add recommendation note
            recommendations = summary.get("recommendations", [])
            if recommendations:
                exec_summary += f"Follow {len(recommendations)} personalized recommendations for the best experience."
            
            return exec_summary
            
        except Exception as e:
            logger.warning(f"Could not generate executive summary: {e}")
            return f"Complete travel plan for {state.destination} with detailed itinerary and cost breakdown."
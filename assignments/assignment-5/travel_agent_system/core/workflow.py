"""
Travel Planner Workflow - Parallel orchestration using LangGraph
"""

from typing import Dict, Any, Optional
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
        return {
            "status": "success",
            "destination": state.destination,
            "travel_plan": state.itinerary_data,
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
"""
LangGraph Node Functions - Individual node implementations for travel planning workflow
"""

from typing import Dict, Any
import logging

from .graph_state import (
    TravelPlanState,
    mark_agent_processing,
    mark_agent_complete,
    mark_agent_error,
    add_error,
    validate_input,
    get_aggregated_data,
    mark_complete
)

logger = logging.getLogger(__name__)


def input_validation_node(state: TravelPlanState) -> TravelPlanState:
    """
    Validate input data and prepare for processing.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state after validation
    """
    logger.info("Starting input validation")
    
    try:
        # Validate input data
        validation_errors = validate_input(state)
        
        if validation_errors:
            logger.error(f"Input validation failed: {validation_errors}")
            new_state = state.copy()
            for error in validation_errors:
                new_state = add_error(new_state, f"Validation: {error}")
            return new_state
        
        # Log successful validation
        logger.info(f"Input validation successful for destination: {state.get('destination')}")
        
        # Return state unchanged if validation passes
        return state
        
    except Exception as e:
        logger.error(f"Input validation node error: {str(e)}")
        return add_error(state, f"Input validation failed: {str(e)}")


def weather_node(state: TravelPlanState) -> TravelPlanState:
    """
    Weather analysis node - placeholder for Stage 2 implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with weather data
    """
    logger.info("Starting weather analysis")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "weather")
        
        # Placeholder implementation - will be replaced in Stage 2
        weather_data = {
            "status": "placeholder",
            "message": "Weather node placeholder - will be implemented in Stage 2",
            "destination": state.get("destination"),
            "current_weather": {
                "temperature": 20,
                "description": "Pleasant",
                "humidity": 60
            },
            "forecast": [
                {"day": "Day 1", "temp": 22, "weather": "Sunny"},
                {"day": "Day 2", "temp": 20, "weather": "Partly Cloudy"},
                {"day": "Day 3", "temp": 18, "weather": "Light Rain"}
            ]
        }
        
        # Mark agent as complete
        new_state = mark_agent_complete(new_state, "weather", weather_data)
        logger.info("Weather analysis completed (placeholder)")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Weather node error: {str(e)}")
        return mark_agent_error(state, "weather", str(e))


def attractions_node(state: TravelPlanState) -> TravelPlanState:
    """
    Attractions discovery node - placeholder for Stage 2 implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with attractions data
    """
    logger.info("Starting attractions discovery")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "attractions")
        
        # Placeholder implementation - will be replaced in Stage 2
        destination = state.get("destination", "Unknown")
        attractions_data = {
            "status": "placeholder",
            "message": "Attractions node placeholder - will be implemented in Stage 2",
            "destination": destination,
            "attractions": [
                {
                    "name": f"{destination} City Center",
                    "category": "landmark",
                    "rating": 8.5,
                    "description": "Historic city center with beautiful architecture"
                },
                {
                    "name": f"{destination} Museum",
                    "category": "museum",
                    "rating": 8.0,
                    "description": "World-class museum with local history"
                },
                {
                    "name": f"{destination} Park",
                    "category": "park",
                    "rating": 7.5,
                    "description": "Beautiful park perfect for walking"
                }
            ],
            "restaurants": [
                {
                    "name": f"Best Restaurant in {destination}",
                    "category": "fine_dining",
                    "rating": 9.0,
                    "price_range": "$$$$"
                }
            ]
        }
        
        # Mark agent as complete
        new_state = mark_agent_complete(new_state, "attractions", attractions_data)
        logger.info("Attractions discovery completed (placeholder)")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Attractions node error: {str(e)}")
        return mark_agent_error(state, "attractions", str(e))


def hotels_node(state: TravelPlanState) -> TravelPlanState:
    """
    Hotel search node - placeholder for Stage 2 implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with hotels data
    """
    logger.info("Starting hotel search")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "hotels")
        
        # Placeholder implementation - will be replaced in Stage 2
        destination = state.get("destination", "Unknown")
        budget = state.get("budget", 1000)
        
        hotels_data = {
            "status": "placeholder",
            "message": "Hotels node placeholder - will be implemented in Stage 2",
            "destination": destination,
            "budget": budget,
            "hotels": [
                {
                    "name": f"Luxury Hotel {destination}",
                    "category": "luxury",
                    "price_per_night": min(200, budget/5),
                    "rating": 9.0,
                    "amenities": ["WiFi", "Pool", "Spa"]
                },
                {
                    "name": f"Business Hotel {destination}",
                    "category": "business",
                    "price_per_night": min(120, budget/8),
                    "rating": 8.0,
                    "amenities": ["WiFi", "Gym", "Conference Room"]
                },
                {
                    "name": f"Budget Inn {destination}",
                    "category": "budget",
                    "price_per_night": min(60, budget/15),
                    "rating": 7.0,
                    "amenities": ["WiFi", "Breakfast"]
                }
            ]
        }
        
        # Mark agent as complete
        new_state = mark_agent_complete(new_state, "hotels", hotels_data)
        logger.info("Hotel search completed (placeholder)")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Hotels node error: {str(e)}")
        return mark_agent_error(state, "hotels", str(e))


def data_aggregation_node(state: TravelPlanState) -> TravelPlanState:
    """
    Aggregate data from parallel nodes and prepare for itinerary creation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with aggregated data
    """
    logger.info("Starting data aggregation")
    
    try:
        # Get aggregated data
        aggregated_data = get_aggregated_data(state)
        
        # Log successful data sources
        successful_sources = []
        if state.get("weather_data"):
            successful_sources.append("weather")
        if state.get("attractions_data"):
            successful_sources.append("attractions")
        if state.get("hotels_data"):
            successful_sources.append("hotels")
        
        logger.info(f"Data aggregation complete. Successful sources: {successful_sources}")
        
        # Add processing stage
        new_state = state.copy()
        current_stages = new_state.get("processing_stages", [])
        new_state["processing_stages"] = current_stages + ["data_aggregation_completed"]
        
        return new_state
        
    except Exception as e:
        logger.error(f"Data aggregation error: {str(e)}")
        return add_error(state, f"Data aggregation failed: {str(e)}")


def itinerary_node(state: TravelPlanState) -> TravelPlanState:
    """
    Itinerary generation node - placeholder for Stage 2 implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with itinerary data
    """
    logger.info("Starting itinerary generation")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "itinerary")
        
        # Get aggregated data
        aggregated_data = get_aggregated_data(new_state)
        
        # Placeholder implementation - will be replaced in Stage 2
        destination = aggregated_data.get("destination", "Unknown")
        budget = aggregated_data.get("budget", 1000)
        
        itinerary_data = {
            "status": "placeholder",
            "message": "Itinerary node placeholder - will be implemented in Stage 2",
            "destination": destination,
            "budget": budget,
            "total_days": 3,
            "daily_itinerary": [
                {
                    "day": 1,
                    "title": f"Arrival in {destination}",
                    "activities": [
                        {"time": "09:00", "activity": "Check-in to hotel", "cost": 0},
                        {"time": "12:00", "activity": "Lunch at local restaurant", "cost": 25},
                        {"time": "14:00", "activity": "City center exploration", "cost": 0},
                        {"time": "19:00", "activity": "Welcome dinner", "cost": 50}
                    ]
                },
                {
                    "day": 2,
                    "title": f"Exploring {destination}",
                    "activities": [
                        {"time": "09:00", "activity": "Museum visit", "cost": 15},
                        {"time": "12:00", "activity": "Lunch in the park", "cost": 20},
                        {"time": "15:00", "activity": "Shopping district", "cost": 100},
                        {"time": "19:00", "activity": "Fine dining experience", "cost": 80}
                    ]
                },
                {
                    "day": 3,
                    "title": "Departure",
                    "activities": [
                        {"time": "09:00", "activity": "Final breakfast", "cost": 15},
                        {"time": "11:00", "activity": "Last-minute shopping", "cost": 50},
                        {"time": "14:00", "activity": "Check-out and departure", "cost": 0}
                    ]
                }
            ],
            "cost_breakdown": {
                "accommodation": budget * 0.4,
                "meals": budget * 0.3,
                "activities": budget * 0.2,
                "transportation": budget * 0.1
            },
            "total_cost": budget
        }
        
        # Mark agent as complete
        new_state = mark_agent_complete(new_state, "itinerary", itinerary_data)
        logger.info("Itinerary generation completed (placeholder)")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Itinerary node error: {str(e)}")
        return mark_agent_error(state, "itinerary", str(e))


def summary_generation_node(state: TravelPlanState) -> TravelPlanState:
    """
    Generate comprehensive trip summary.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state marked as complete
    """
    logger.info("Starting summary generation")
    
    try:
        # Mark workflow as complete
        new_state = mark_complete(state)
        
        # Add summary generation stage
        current_stages = new_state.get("processing_stages", [])
        new_state["processing_stages"] = current_stages + ["summary_generation_completed"]
        
        logger.info("Summary generation completed")
        return new_state
        
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return add_error(state, f"Summary generation failed: {str(e)}")


def error_handling_node(state: TravelPlanState) -> TravelPlanState:
    """
    Handle errors and provide fallback responses.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with error handling
    """
    logger.info("Starting error handling")
    
    try:
        errors = state.get("errors", [])
        logger.warning(f"Handling {len(errors)} errors: {errors}")
        
        # Create error summary  
        new_state = state.copy()
        current_stages = new_state.get("processing_stages", [])
        new_state["processing_stages"] = current_stages + ["error_handling_completed"]
        
        # Mark as complete even with errors
        new_state = mark_complete(new_state)
        
        logger.info("Error handling completed")
        return new_state
        
    except Exception as e:
        logger.error(f"Error handling node failed: {str(e)}")
        return add_error(state, f"Error handling failed: {str(e)}")


# Node validation function for testing
def validate_node_signatures():
    """
    Validate that all node functions have correct signatures.
    
    Returns:
        bool: True if all signatures are valid
    """
    import inspect
    
    nodes = [
        input_validation_node,
        weather_node,
        attractions_node,
        hotels_node,
        data_aggregation_node,
        itinerary_node,
        summary_generation_node,
        error_handling_node
    ]
    
    for node in nodes:
        sig = inspect.signature(node)
        
        # Check parameters
        params = list(sig.parameters.keys())
        if len(params) != 1 or params[0] != "state":
            logger.error(f"Node {node.__name__} has invalid signature: {sig}")
            return False
        
        # Check parameter type annotation
        state_param = sig.parameters["state"]
        if state_param.annotation != TravelPlanState:
            logger.error(f"Node {node.__name__} has incorrect state type annotation")
            return False
        
        # Check return type annotation
        if sig.return_annotation != TravelPlanState:
            logger.error(f"Node {node.__name__} has incorrect return type annotation")
            return False
    
    logger.info("All node signatures are valid")
    return True
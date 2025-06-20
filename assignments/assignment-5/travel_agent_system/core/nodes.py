"""
LangGraph Node Functions - Individual node implementations for travel planning workflow
"""

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
    Enhanced input validation and query parsing for travel requests.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state after validation and enhancement
    """
    logger.info("Starting input validation and enhancement")
    
    try:
        # Check if we have query data that needs parsing FIRST (before validation)
        raw_query = state.get("raw_query")  # May be set from query interface
        if raw_query and not state.get("destination"):
            # Parse the query using LLM + Pydantic approach (LLM-only, no regex fallback)
            from ..utils.llm_query_parser import parse_travel_query
            
            logger.info(f"Parsing raw query with LLM: {raw_query}")
            parsing_result = parse_travel_query(raw_query)
            
            if not parsing_result.success:
                error_msg = parsing_result.error_message
                logger.error(f"LLM query parsing failed: {error_msg}")
                
                # Provide helpful error message for users
                if "LLM parsing service unavailable" in error_msg:
                    helpful_msg = ("Natural language query parsing requires Gemini API. "
                                 "Please either configure GEMINI_API_KEY or provide structured travel data "
                                 "(destination, travel_dates, budget) directly.")
                else:
                    helpful_msg = f"Failed to understand travel query: {error_msg}"
                
                new_state = add_error(state, helpful_msg)
                return new_state
            
            # Convert parsed query to travel request format
            travel_request = parsing_result.query.to_travel_request()
            
            # Update state with parsed data
            new_state = state.copy()
            for key, value in travel_request.items():
                if value is not None:
                    new_state[key] = value
            
            # Re-validate after parsing
            validation_errors = validate_input(new_state)
            if validation_errors:
                logger.error(f"Parsed data validation failed: {validation_errors}")
                for error in validation_errors:
                    new_state = add_error(new_state, f"Parsed validation: {error}")
                return new_state
            
            logger.info(f"Successfully parsed query with LLM for destination: {new_state.get('destination')}")
            return new_state
        
        # If no raw query to parse, do basic validation on existing data
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


# Old string manipulation functions removed - replaced with LLM + Pydantic parsing
# See travel_agent_system/utils/llm_query_parser.py for the new implementation


def weather_node(state: TravelPlanState) -> TravelPlanState:
    """
    Weather analysis node - full WeatherAgent implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with weather data
    """
    logger.info("Starting weather analysis")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "weather")
        
        # Get destination and travel dates from state
        destination = state.get("destination")
        travel_dates = state.get("travel_dates")
        
        if not destination:
            error_msg = "Destination is required for weather analysis"
            logger.error(error_msg)
            return mark_agent_error(state, "weather", error_msg)
        
        # Import and initialize WeatherAgent
        from ..agents.weather_agent import WeatherAgent
        weather_agent = WeatherAgent()
        
        # Perform weather analysis
        weather_analysis = weather_agent.analyze_weather_for_travel(
            destination=destination,
            travel_dates=travel_dates
        )
        
        # Check for errors in analysis
        if "error" in weather_analysis:
            error_msg = weather_analysis["error"]
            logger.error(f"Weather analysis failed: {error_msg}")
            return mark_agent_error(new_state, "weather", error_msg)
        
        # Mark agent as complete with results
        new_state = mark_agent_complete(new_state, "weather", weather_analysis)
        logger.info(f"Weather analysis completed for {destination}")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Weather node error: {str(e)}")
        return mark_agent_error(state, "weather", str(e))


def attractions_node(state: TravelPlanState) -> TravelPlanState:
    """
    Attractions discovery node - full AttractionAgent implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with attractions data
    """
    logger.info("Starting attractions discovery")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "attractions")
        
        # Get destination and budget info from state
        destination = state.get("destination")
        budget = state.get("budget")
        preferences = state.get("preferences", {})
        
        if not destination:
            error_msg = "Destination is required for attractions discovery"
            logger.error(error_msg)
            return mark_agent_error(state, "attractions", error_msg)
        
        # Determine budget level
        if budget:
            if budget <= 500:
                budget_level = "low"
            elif budget <= 1500:
                budget_level = "medium"
            else:
                budget_level = "high"
        else:
            budget_level = "medium"
        
        # Import and initialize AttractionAgent
        from ..agents.attraction_agent import AttractionAgent
        attraction_agent = AttractionAgent()
        
        # Discover attractions
        attractions_discovery = attraction_agent.discover_attractions(
            destination=destination,
            categories=["attractions", "restaurants", "activities", "entertainment"],
            budget_level=budget_level
        )
        
        # Check for errors in discovery
        if "error" in attractions_discovery:
            error_msg = attractions_discovery["error"]
            logger.error(f"Attractions discovery failed: {error_msg}")
            return mark_agent_error(new_state, "attractions", error_msg)
        
        # Mark agent as complete with results
        new_state = mark_agent_complete(new_state, "attractions", attractions_discovery)
        logger.info(f"Attractions discovery completed for {destination}")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Attractions node error: {str(e)}")
        return mark_agent_error(state, "attractions", str(e))


def hotels_node(state: TravelPlanState) -> TravelPlanState:
    """
    Hotel search node - full HotelAgent implementation.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with hotels data
    """
    logger.info("Starting hotel search")
    
    try:
        # Mark agent as processing
        new_state = mark_agent_processing(state, "hotels")
        
        # Get destination, budget, and travel dates from state
        destination = state.get("destination")
        budget = state.get("budget")
        travel_dates = state.get("travel_dates")
        preferences = state.get("preferences", {})
        
        if not destination:
            error_msg = "Destination is required for hotel search"
            logger.error(error_msg)
            return mark_agent_error(state, "hotels", error_msg)
        
        # Prepare budget range if budget is provided
        budget_range = None
        if budget:
            # Assume 40% of total budget goes to accommodation
            accommodation_budget = budget * 0.4
            # Calculate days to get nightly budget
            days = 3  # Default
            if travel_dates and travel_dates.get("start_date") and travel_dates.get("end_date"):
                try:
                    from datetime import datetime
                    start = datetime.fromisoformat(travel_dates["start_date"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(travel_dates["end_date"].replace("Z", "+00:00"))
                    days = max(1, (end - start).days)
                except Exception:
                    pass
            
            max_per_night = accommodation_budget / days
            budget_range = {"min": max_per_night * 0.5, "max": max_per_night}
        
        # Import and initialize HotelAgent
        from ..agents.hotel_agent import HotelAgent
        hotel_agent = HotelAgent()
        
        # Search for hotels
        hotel_search_results = hotel_agent.search_hotels(
            destination=destination,
            budget_range=budget_range,
            travel_dates=travel_dates,
            preferences=preferences
        )
        
        # Check for errors in search
        if "error" in hotel_search_results:
            error_msg = hotel_search_results["error"]
            logger.error(f"Hotel search failed: {error_msg}")
            return mark_agent_error(new_state, "hotels", error_msg)
        
        # Mark agent as complete with results
        new_state = mark_agent_complete(new_state, "hotels", hotel_search_results)
        logger.info(f"Hotel search completed for {destination}")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Hotels node error: {str(e)}")
        return mark_agent_error(state, "hotels", str(e))


def data_aggregation_node(state: TravelPlanState) -> TravelPlanState:
    """
    Execute parallel agents and aggregate their data.
    This node handles the parallel execution of weather, attractions, and hotels agents.
    
    Args:
        state: Current workflow state
        
    Returns:
        TravelPlanState: Updated state with aggregated data from all agents
    """
    logger.info("Starting parallel agent execution and data aggregation")
    
    try:
        # Execute all parallel agents
        new_state = state.copy()
        
        # Execute weather agent
        logger.info("Executing weather agent...")
        new_state = weather_node(new_state)
        
        # Execute attractions agent
        logger.info("Executing attractions agent...")
        new_state = attractions_node(new_state)
        
        # Execute hotels agent  
        logger.info("Executing hotels agent...")
        new_state = hotels_node(new_state)
        
        # Get aggregated data
        aggregated_data = get_aggregated_data(new_state)
        
        # Log successful data sources
        successful_sources = []
        if new_state.get("weather_data"):
            successful_sources.append("weather")
        if new_state.get("attractions_data"):
            successful_sources.append("attractions")
        if new_state.get("hotels_data"):
            successful_sources.append("hotels")
        
        logger.info(f"Data aggregation complete. Successful sources: {successful_sources}")
        
        # Add processing stage
        current_stages = new_state.get("processing_stages", [])
        new_state["processing_stages"] = current_stages + ["data_aggregation_completed"]
        
        return new_state
        
    except Exception as e:
        logger.error(f"Data aggregation error: {str(e)}")
        return add_error(state, f"Data aggregation failed: {str(e)}")


def itinerary_node(state: TravelPlanState) -> TravelPlanState:
    """
    Itinerary generation node - full ItineraryAgent implementation.
    
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
        
        # Validate we have minimum required data
        if not aggregated_data.get("destination"):
            error_msg = "Destination is required for itinerary generation"
            logger.error(error_msg)
            return mark_agent_error(new_state, "itinerary", error_msg)
        
        # Prepare preferences from state
        preferences = aggregated_data.get("preferences", {})
        if aggregated_data.get("budget"):
            preferences["budget"] = aggregated_data["budget"]
        if aggregated_data.get("currency"):
            preferences["currency"] = aggregated_data["currency"]
        if aggregated_data.get("travel_dates"):
            preferences["travel_dates"] = aggregated_data["travel_dates"]
        
        # Import and initialize ItineraryAgent with tools
        from ..agents.itinerary_agent import ItineraryAgent
        from ..tools.cost_calculator import CostCalculator
        from ..tools.currency_converter import CurrencyConverter
        
        cost_calculator = CostCalculator()
        currency_converter = CurrencyConverter()
        itinerary_agent = ItineraryAgent(
            cost_calculator=cost_calculator,
            currency_converter=currency_converter
        )
        
        # Create comprehensive itinerary
        itinerary_result = itinerary_agent.create_itinerary(
            trip_data=aggregated_data,
            preferences=preferences
        )
        
        # Check for errors in itinerary creation
        if "error" in itinerary_result:
            error_msg = itinerary_result["error"]
            logger.error(f"Itinerary generation failed: {error_msg}")
            return mark_agent_error(new_state, "itinerary", error_msg)
        
        # Mark agent as complete with results
        new_state = mark_agent_complete(new_state, "itinerary", itinerary_result)
        logger.info(f"Itinerary generation completed for {aggregated_data.get('destination')}")
        
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
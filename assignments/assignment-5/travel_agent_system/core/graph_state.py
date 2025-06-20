"""
LangGraph State Management - TypedDict state schema for travel planning workflow
"""

from typing import Dict, Any, Optional, List, TypedDict
from typing_extensions import NotRequired
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TravelPlanState(TypedDict):
    """
    LangGraph-compatible state schema for travel planning workflow.
    Uses TypedDict for LangGraph state management with proper annotations.
    """
    
    # Input data
    destination: NotRequired[Optional[str]]
    travel_dates: NotRequired[Optional[Dict[str, str]]]
    budget: NotRequired[Optional[float]]
    currency: NotRequired[Optional[str]]
    preferences: NotRequired[Optional[Dict[str, Any]]]
    raw_query: NotRequired[Optional[str]]  # For LLM parsing of natural language queries
    
    # Agent processing status
    weather_processing: NotRequired[bool]
    attractions_processing: NotRequired[bool]
    hotels_processing: NotRequired[bool]
    itinerary_processing: NotRequired[bool]
    
    # Agent results
    weather_data: NotRequired[Optional[Dict[str, Any]]]
    attractions_data: NotRequired[Optional[Dict[str, Any]]]
    hotels_data: NotRequired[Optional[Dict[str, Any]]]
    itinerary_data: NotRequired[Optional[Dict[str, Any]]]
    
    # Processing metadata
    errors: NotRequired[List[str]]
    start_time: NotRequired[Optional[str]]
    completion_time: NotRequired[Optional[str]]
    processing_stages: NotRequired[List[str]]


def create_initial_state(**kwargs) -> TravelPlanState:
    """
    Create initial state with default values for LangGraph workflow.
    
    Args:
        **kwargs: Initial state values to override defaults
        
    Returns:
        TravelPlanState: Initialized state dictionary
    """
    state: TravelPlanState = {
        # Input data
        "destination": kwargs.get("destination"),
        "travel_dates": kwargs.get("travel_dates"),
        "budget": kwargs.get("budget"),
        "currency": kwargs.get("currency", "USD"),
        "preferences": kwargs.get("preferences", {}),
        "raw_query": kwargs.get("raw_query"),
        
        # Agent processing status
        "weather_processing": False,
        "attractions_processing": False,
        "hotels_processing": False,
        "itinerary_processing": False,
        
        # Agent results
        "weather_data": None,
        "attractions_data": None,
        "hotels_data": None,
        "itinerary_data": None,
        
        # Processing metadata
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "completion_time": None,
        "processing_stages": []
    }
    
    return state


def add_to_list(current: List[str], new: str) -> List[str]:
    """
    State reducer function for adding items to lists.
    Used for errors and processing_stages.
    
    Args:
        current: Current list value
        new: New item to add
        
    Returns:
        List[str]: Updated list with new item
    """
    if current is None:
        return [new]
    return current + [new]


def mark_agent_processing(state: TravelPlanState, agent_name: str) -> TravelPlanState:
    """
    Mark an agent as currently processing.
    
    Args:
        state: Current state
        agent_name: Name of the agent (weather, attractions, hotels, itinerary)
        
    Returns:
        TravelPlanState: Updated state
    """
    new_state = state.copy()
    new_state[f"{agent_name}_processing"] = True
    
    # Add to processing stages
    current_stages = new_state.get("processing_stages", [])
    new_state["processing_stages"] = add_to_list(current_stages, f"{agent_name}_started")
    
    logger.info(f"Agent {agent_name} started processing")
    return new_state


def mark_agent_complete(state: TravelPlanState, agent_name: str, result_data: Dict[str, Any]) -> TravelPlanState:
    """
    Mark an agent as complete with results.
    
    Args:
        state: Current state
        agent_name: Name of the agent
        result_data: Results from the agent
        
    Returns:
        TravelPlanState: Updated state
    """
    new_state = state.copy()
    new_state[f"{agent_name}_processing"] = False
    new_state[f"{agent_name}_data"] = result_data
    
    # Add to processing stages
    current_stages = new_state.get("processing_stages", [])
    new_state["processing_stages"] = add_to_list(current_stages, f"{agent_name}_completed")
    
    logger.info(f"Agent {agent_name} completed processing")
    return new_state


def mark_agent_error(state: TravelPlanState, agent_name: str, error_message: str) -> TravelPlanState:
    """
    Mark an agent as failed with error.
    
    Args:
        state: Current state
        agent_name: Name of the agent
        error_message: Error message
        
    Returns:
        TravelPlanState: Updated state
    """
    new_state = state.copy()
    new_state[f"{agent_name}_processing"] = False
    
    # Add error
    current_errors = new_state.get("errors", [])
    new_state["errors"] = add_to_list(current_errors, f"{agent_name}: {error_message}")
    
    # Add to processing stages
    current_stages = new_state.get("processing_stages", [])
    new_state["processing_stages"] = add_to_list(current_stages, f"{agent_name}_failed")
    
    logger.error(f"Agent {agent_name} failed: {error_message}")
    return new_state


def add_error(state: TravelPlanState, error_message: str) -> TravelPlanState:
    """
    Add a general error message to state.
    
    Args:
        state: Current state
        error_message: Error message to add
        
    Returns:
        TravelPlanState: Updated state
    """
    new_state = state.copy()
    current_errors = new_state.get("errors", [])
    new_state["errors"] = add_to_list(current_errors, error_message)
    
    logger.error(f"State error: {error_message}")
    return new_state


def is_parallel_phase_complete(state: TravelPlanState) -> bool:
    """
    Check if all parallel agents have completed (success or failure).
    
    Args:
        state: Current state
        
    Returns:
        bool: True if all parallel agents are complete
    """
    return (not state.get("weather_processing", False) and 
            not state.get("attractions_processing", False) and 
            not state.get("hotels_processing", False))


def has_sufficient_data_for_itinerary(state: TravelPlanState) -> bool:
    """
    Check if we have enough data to create an itinerary.
    
    Args:
        state: Current state
        
    Returns:
        bool: True if sufficient data is available
    """
    # Need at least destination and one other data source
    if not state.get("destination"):
        return False
    
    data_sources = [
        state.get("weather_data"),
        state.get("attractions_data"), 
        state.get("hotels_data")
    ]
    
    return sum(1 for data in data_sources if data is not None) >= 1


def get_successful_data_sources(state: TravelPlanState) -> List[str]:
    """
    Get list of successfully completed data sources.
    
    Args:
        state: Current state
        
    Returns:
        List[str]: List of successful data sources
    """
    sources = []
    if state.get("weather_data"):
        sources.append("weather")
    if state.get("attractions_data"):
        sources.append("attractions")
    if state.get("hotels_data"):
        sources.append("hotels")
    return sources


def get_failed_data_sources(state: TravelPlanState) -> List[str]:
    """
    Get list of failed data sources.
    
    Args:
        state: Current state
        
    Returns:
        List[str]: List of failed data sources
    """
    failed = []
    errors = state.get("errors", [])
    
    for error in errors:
        if "weather:" in error:
            failed.append("weather")
        elif "attractions:" in error:
            failed.append("attractions")
        elif "hotels:" in error:
            failed.append("hotels")
    
    return list(set(failed))  # Remove duplicates


def get_aggregated_data(state: TravelPlanState) -> Dict[str, Any]:
    """
    Get all collected data in a single dictionary for itinerary creation.
    
    Args:
        state: Current state
        
    Returns:
        Dict[str, Any]: Aggregated data for itinerary creation
    """
    return {
        "destination": state.get("destination"),
        "travel_dates": state.get("travel_dates"),
        "budget": state.get("budget"),
        "currency": state.get("currency"),
        "preferences": state.get("preferences"),
        "weather_data": state.get("weather_data"),
        "attractions_data": state.get("attractions_data"),
        "hotels_data": state.get("hotels_data")
    }


def mark_complete(state: TravelPlanState) -> TravelPlanState:
    """
    Mark the entire workflow as complete.
    
    Args:
        state: Current state
        
    Returns:
        TravelPlanState: Updated state
    """
    new_state = state.copy()
    new_state["completion_time"] = datetime.now().isoformat()
    
    # Add to processing stages
    current_stages = new_state.get("processing_stages", [])
    new_state["processing_stages"] = add_to_list(current_stages, "workflow_completed")
    
    logger.info("Travel planning workflow completed")
    return new_state


def get_processing_summary(state: TravelPlanState) -> Dict[str, Any]:
    """
    Get summary of processing status and results.
    
    Args:
        state: Current state
        
    Returns:
        Dict[str, Any]: Processing summary
    """
    return {
        "destination": state.get("destination"),
        "status": "completed" if state.get("completion_time") else "processing",
        "successful_sources": get_successful_data_sources(state),
        "failed_sources": get_failed_data_sources(state),
        "error_count": len(state.get("errors", [])),
        "processing_time": get_processing_time(state),
        "stages": state.get("processing_stages", [])
    }


def get_processing_time(state: TravelPlanState) -> Optional[float]:
    """
    Calculate processing time in seconds.
    
    Args:
        state: Current state
        
    Returns:
        Optional[float]: Processing time in seconds, None if calculation fails
    """
    start_time = state.get("start_time")
    if not start_time:
        return None
    
    end_time = state.get("completion_time") or datetime.now().isoformat()
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except Exception:
        return None


def validate_input(state: TravelPlanState) -> List[str]:
    """
    Validate input data and return list of validation errors.
    
    Args:
        state: Current state
        
    Returns:
        List[str]: List of validation errors
    """
    validation_errors = []
    
    destination = state.get("destination")
    if not destination or not destination.strip():
        validation_errors.append("Destination is required")
    
    travel_dates = state.get("travel_dates")
    if travel_dates:
        if not isinstance(travel_dates, dict):
            validation_errors.append("Travel dates must be a dictionary")
        else:
            if "start_date" not in travel_dates:
                validation_errors.append("Start date is required in travel dates")
            if "end_date" not in travel_dates:
                validation_errors.append("End date is required in travel dates")
    
    budget = state.get("budget")
    if budget is not None and budget <= 0:
        validation_errors.append("Budget must be positive")
    
    return validation_errors


# State reducers for LangGraph
def error_reducer(current: List[str], new: str) -> List[str]:
    """State reducer for errors list"""
    return add_to_list(current, new)


def processing_stages_reducer(current: List[str], new: str) -> List[str]:
    """State reducer for processing_stages list"""
    return add_to_list(current, new)
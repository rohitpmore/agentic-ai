"""
Travel Plan State Management - State management for parallel workflow
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TravelPlanState:
    """
    State management for travel planning workflow.
    Manages data flow between parallel agents and tracks processing status.
    """
    
    # Input data
    destination: Optional[str] = None
    travel_dates: Optional[Dict[str, str]] = None
    budget: Optional[float] = None
    currency: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    # Agent processing status
    weather_processing: bool = False
    attractions_processing: bool = False
    hotels_processing: bool = False
    itinerary_processing: bool = False
    
    # Agent results
    weather_data: Optional[Dict[str, Any]] = None
    attractions_data: Optional[Dict[str, Any]] = None
    hotels_data: Optional[Dict[str, Any]] = None
    itinerary_data: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    errors: Optional[List[str]] = None
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    processing_stages: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize state after creation"""
        if self.errors is None:
            self.errors = []
        if self.processing_stages is None:
            self.processing_stages = []
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
    
    def mark_agent_processing(self, agent_name: str):
        """Mark an agent as currently processing"""
        setattr(self, f"{agent_name}_processing", True)
        self.processing_stages.append(f"{agent_name}_started")
        logger.info(f"Agent {agent_name} started processing")
    
    def mark_agent_complete(self, agent_name: str, result_data: Dict[str, Any]):
        """Mark an agent as complete with results"""
        setattr(self, f"{agent_name}_processing", False)
        setattr(self, f"{agent_name}_data", result_data)
        self.processing_stages.append(f"{agent_name}_completed")
        logger.info(f"Agent {agent_name} completed processing")
    
    def mark_agent_error(self, agent_name: str, error_message: str):
        """Mark an agent as failed with error"""
        setattr(self, f"{agent_name}_processing", False)
        self.errors.append(f"{agent_name}: {error_message}")
        self.processing_stages.append(f"{agent_name}_failed")
        logger.error(f"Agent {agent_name} failed: {error_message}")
    
    def add_error(self, error_message: str):
        """Add a general error message"""
        self.errors.append(error_message)
        logger.error(f"State error: {error_message}")
    
    def is_parallel_phase_complete(self) -> bool:
        """Check if all parallel agents have completed (success or failure)"""
        return (not self.weather_processing and 
                not self.attractions_processing and 
                not self.hotels_processing)
    
    def has_sufficient_data_for_itinerary(self) -> bool:
        """Check if we have enough data to create an itinerary"""
        # Need at least destination and one other data source
        if not self.destination:
            return False
        
        data_sources = [
            self.weather_data,
            self.attractions_data, 
            self.hotels_data
        ]
        
        return sum(1 for data in data_sources if data is not None) >= 1
    
    def get_successful_data_sources(self) -> List[str]:
        """Get list of successfully completed data sources"""
        sources = []
        if self.weather_data:
            sources.append("weather")
        if self.attractions_data:
            sources.append("attractions")
        if self.hotels_data:
            sources.append("hotels")
        return sources
    
    def get_failed_data_sources(self) -> List[str]:
        """Get list of failed data sources"""
        failed = []
        for error in self.errors:
            if "weather:" in error:
                failed.append("weather")
            elif "attractions:" in error:
                failed.append("attractions")
            elif "hotels:" in error:
                failed.append("hotels")
        return list(set(failed))  # Remove duplicates
    
    def get_aggregated_data(self) -> Dict[str, Any]:
        """Get all collected data in a single dictionary for itinerary creation"""
        return {
            "destination": self.destination,
            "travel_dates": self.travel_dates,
            "budget": self.budget,
            "currency": self.currency,
            "preferences": self.preferences,
            "weather_data": self.weather_data,
            "attractions_data": self.attractions_data,
            "hotels_data": self.hotels_data
        }
    
    def mark_complete(self):
        """Mark the entire workflow as complete"""
        self.completion_time = datetime.now().isoformat()
        self.processing_stages.append("workflow_completed")
        logger.info("Travel planning workflow completed")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing status and results"""
        return {
            "destination": self.destination,
            "status": "completed" if self.completion_time else "processing",
            "successful_sources": self.get_successful_data_sources(),
            "failed_sources": self.get_failed_data_sources(),
            "error_count": len(self.errors),
            "processing_time": self.get_processing_time(),
            "stages": self.processing_stages
        }
    
    def get_processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds"""
        if not self.start_time:
            return None
        
        end_time = self.completion_time or datetime.now().isoformat()
        try:
            start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            return (end - start).total_seconds()
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return asdict(self)
    
    def validate_input(self) -> List[str]:
        """Validate input data and return list of validation errors"""
        validation_errors = []
        
        if not self.destination or not self.destination.strip():
            validation_errors.append("Destination is required")
        
        if self.travel_dates:
            if not isinstance(self.travel_dates, dict):
                validation_errors.append("Travel dates must be a dictionary")
            else:
                if "start_date" not in self.travel_dates:
                    validation_errors.append("Start date is required in travel dates")
                if "end_date" not in self.travel_dates:
                    validation_errors.append("End date is required in travel dates")
        
        if self.budget is not None and self.budget <= 0:
            validation_errors.append("Budget must be positive")
        
        return validation_errors
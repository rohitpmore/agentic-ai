"""
Pydantic models for travel query parsing and validation
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class TravelPace(str, Enum):
    """Travel pace preferences"""
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


class Currency(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"


class TravelPreferences(BaseModel):
    """Travel preferences and constraints"""
    pace: TravelPace = TravelPace.MODERATE
    budget_level: Optional[str] = None  # "low", "medium", "high"
    accommodation_type: Optional[str] = None  # "budget", "mid-range", "luxury"
    food_preferences: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    accessibility_needs: Optional[List[str]] = None


class TravelQuery(BaseModel):
    """
    Structured representation of a travel query parsed from natural language.
    
    This model represents the user's travel requirements extracted from
    natural language input using LLM parsing.
    """
    
    # Core travel parameters
    destination: str = Field(..., description="Travel destination city or country")
    duration_days: Optional[int] = Field(None, ge=1, le=365, description="Trip duration in days")
    
    # Budget information
    budget: Optional[float] = Field(None, ge=0, description="Total budget amount")
    currency: Currency = Currency.USD
    
    # Travel dates
    start_date: Optional[date] = Field(None, description="Trip start date")
    end_date: Optional[date] = Field(None, description="Trip end date")
    
    # Preferences and constraints
    preferences: TravelPreferences = Field(default_factory=TravelPreferences)
    
    # Additional context
    group_size: int = Field(1, ge=1, le=50, description="Number of travelers")
    special_requirements: Optional[List[str]] = None
    
    @field_validator('end_date')
    @classmethod
    def end_date_after_start_date(cls, v, info):
        """Ensure end date is after start date"""
        if v and info.data.get('start_date') and v <= info.data.get('start_date'):
            raise ValueError('End date must be after start date')
        return v
    
    @field_validator('duration_days')
    @classmethod
    def validate_duration_with_dates(cls, v, info):
        """Validate duration matches date range if both are provided"""
        start_date = info.data.get('start_date')
        end_date = info.data.get('end_date')
        
        if v and start_date and end_date:
            calculated_days = (end_date - start_date).days + 1
            if abs(calculated_days - v) > 1:  # Allow 1 day tolerance
                raise ValueError(f'Duration ({v} days) does not match date range ({calculated_days} days)')
        
        return v
    
    @field_validator('destination')
    @classmethod
    def validate_destination(cls, v):
        """Ensure destination is not empty and properly formatted"""
        if not v or not v.strip():
            raise ValueError('Destination cannot be empty')
        return v.strip().title()
    
    def to_travel_request(self) -> Dict[str, Any]:
        """
        Convert to the format expected by the travel planning system.
        
        Returns:
            Dict containing travel request parameters
        """
        travel_request = {
            "destination": self.destination,
            "currency": self.currency.value,
            "preferences": self.preferences.model_dump()
        }
        
        # Add budget if specified
        if self.budget:
            travel_request["budget"] = self.budget
        
        # Add travel dates
        if self.start_date and self.end_date:
            travel_request["travel_dates"] = {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat()
            }
        elif self.duration_days:
            # Generate dates if duration is specified but not specific dates
            from datetime import timedelta
            start = datetime.now().date() + timedelta(days=7)  # Default start in a week
            end = start + timedelta(days=self.duration_days - 1)
            travel_request["travel_dates"] = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat()
            }
        
        # Add group size if > 1
        if self.group_size > 1:
            travel_request["preferences"]["group_size"] = self.group_size
        
        # Add special requirements
        if self.special_requirements:
            travel_request["preferences"]["special_requirements"] = self.special_requirements
        
        return travel_request
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TravelQuery":
        """
        Create TravelQuery from dictionary (typically from LLM output).
        
        Args:
            data: Dictionary containing parsed travel parameters
            
        Returns:
            TravelQuery instance
        """
        # Handle preferences separately
        preferences_data = data.pop("preferences", {})
        # Remove None values to let defaults take effect
        preferences_data = {k: v for k, v in preferences_data.items() if v is not None}
        preferences = TravelPreferences(**preferences_data)
        
        # Handle dates
        if "start_date" in data and isinstance(data["start_date"], str):
            try:
                data["start_date"] = datetime.fromisoformat(data["start_date"]).date()
            except ValueError:
                data.pop("start_date", None)
        
        if "end_date" in data and isinstance(data["end_date"], str):
            try:
                data["end_date"] = datetime.fromisoformat(data["end_date"]).date()
            except ValueError:
                data.pop("end_date", None)
        
        # Handle currency
        if "currency" in data and isinstance(data["currency"], str):
            try:
                data["currency"] = Currency(data["currency"].upper())
            except ValueError:
                data["currency"] = Currency.USD  # Default fallback
        
        return cls(preferences=preferences, **data)


class QueryParsingResult(BaseModel):
    """
    Result of query parsing operation.
    
    Contains either successfully parsed query or error information.
    """
    success: bool
    query: Optional[TravelQuery] = None
    error_message: Optional[str] = None
    raw_input: Optional[str] = None
    
    @classmethod
    def success_result(cls, query: TravelQuery, raw_input: str) -> "QueryParsingResult":
        """Create successful parsing result"""
        return cls(
            success=True,
            query=query,
            raw_input=raw_input
        )
    
    @classmethod
    def error_result(cls, error_message: str, raw_input: str) -> "QueryParsingResult":
        """Create error parsing result"""
        return cls(
            success=False,
            error_message=error_message,
            raw_input=raw_input
        )
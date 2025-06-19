"""
Integration tests for the travel planner workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_agent_system.core.workflow import TravelPlannerWorkflow
from travel_agent_system.core.state import TravelPlanState


class TestTravelPlanState:
    """Test cases for TravelPlanState"""
    
    @pytest.fixture
    def sample_state(self):
        return TravelPlanState(
            destination="Paris",
            travel_dates={"start_date": "2024-01-01", "end_date": "2024-01-03"},
            budget=1500,
            currency="USD",
            preferences={"pace": "moderate"}
        )
    
    def test_init(self, sample_state):
        """Test TravelPlanState initialization"""
        assert sample_state.destination == "Paris"
        assert sample_state.budget == 1500
        assert sample_state.currency == "USD"
        assert sample_state.preferences == {"pace": "moderate"}
        assert sample_state.workflow_complete is False
        assert sample_state.errors == []
    
    def test_validate_input_success(self, sample_state):
        """Test successful input validation"""
        errors = sample_state.validate_input()
        assert errors == []
    
    def test_validate_input_missing_destination(self):
        """Test validation with missing destination"""
        state = TravelPlanState(destination="")
        errors = state.validate_input()
        assert len(errors) > 0
        assert any("destination" in error.lower() for error in errors)
    
    def test_mark_agent_complete(self, sample_state):
        """Test marking agent as complete"""
        test_data = {"result": "success"}
        sample_state.mark_agent_complete("weather", test_data)
        
        assert sample_state.agent_status["weather"] == "completed"
        assert sample_state.weather_data == test_data
    
    def test_mark_agent_error(self, sample_state):
        """Test marking agent as error"""
        sample_state.mark_agent_error("weather", "API failed")
        
        assert sample_state.agent_status["weather"] == "error"
        assert "API failed" in sample_state.errors
    
    def test_has_sufficient_data_for_itinerary(self, sample_state):
        """Test checking sufficient data for itinerary"""
        # Initially insufficient
        assert sample_state.has_sufficient_data_for_itinerary() is False
        
        # Add some data
        sample_state.mark_agent_complete("weather", {"temperature": 20})
        sample_state.mark_agent_complete("attractions", {"attractions": []})
        
        # Still need hotels
        assert sample_state.has_sufficient_data_for_itinerary() is False
        
        # Add hotels
        sample_state.mark_agent_complete("hotels", {"hotel_options": []})
        
        # Now sufficient
        assert sample_state.has_sufficient_data_for_itinerary() is True
    
    def test_get_processing_summary(self, sample_state):
        """Test getting processing summary"""
        sample_state.mark_agent_complete("weather", {"temp": 20})
        sample_state.mark_agent_error("attractions", "Failed")
        
        summary = sample_state.get_processing_summary()
        
        assert summary["completed"] == 1
        assert summary["total"] == 4  # weather, attractions, hotels, itinerary
        assert "attractions" in summary["failed"]
        assert summary["completed_agents"] == ["weather"]


class TestTravelPlannerWorkflow:
    """Test cases for TravelPlannerWorkflow"""
    
    @pytest.fixture
    def workflow(self):
        return TravelPlannerWorkflow()
    
    @pytest.fixture
    def sample_request(self):
        return {
            "destination": "Paris",
            "travel_dates": {"start_date": "2024-01-01", "end_date": "2024-01-03"},
            "budget": 1500,
            "currency": "USD",
            "preferences": {"pace": "moderate"}
        }
    
    def test_init(self, workflow):
        """Test TravelPlannerWorkflow initialization"""
        assert workflow is not None
        assert hasattr(workflow, 'weather_agent')
        assert hasattr(workflow, 'attraction_agent')
        assert hasattr(workflow, 'hotel_agent')
        assert hasattr(workflow, 'itinerary_agent')
        assert workflow.max_workers == 3
    
    @patch('travel_agent_system.core.workflow.ThreadPoolExecutor')
    @patch.object(TravelPlannerWorkflow, '_execute_weather_agent')
    @patch.object(TravelPlannerWorkflow, '_execute_attraction_agent')
    @patch.object(TravelPlannerWorkflow, '_execute_hotel_agent')
    @patch.object(TravelPlannerWorkflow, '_execute_itinerary_agent')
    def test_process_travel_request_success(self, mock_itinerary, mock_hotel, 
                                          mock_attraction, mock_weather, 
                                          mock_executor, workflow, sample_request):
        """Test successful travel request processing"""
        # Mock agent results
        mock_weather.return_value = {"temperature": 20, "forecast": []}
        mock_attraction.return_value = {"attractions": [{"name": "Eiffel Tower"}]}
        mock_hotel.return_value = {"hotel_options": [{"name": "Hotel Paris"}]}
        mock_itinerary.return_value = None
        
        # Mock ThreadPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Mock futures
        mock_futures = [MagicMock(), MagicMock(), MagicMock()]
        mock_executor_instance.submit.side_effect = mock_futures
        
        # Mock as_completed to return futures with results
        with patch('travel_agent_system.core.workflow.as_completed', return_value=mock_futures):
            for i, future in enumerate(mock_futures):
                future.result.return_value = {
                    "weather": {"temperature": 20},
                    "attractions": {"attractions": []},
                    "hotels": {"hotel_options": []}
                }[["weather", "attractions", "hotels"][i]]
        
        result = workflow.process_travel_request(sample_request)
        
        assert result is not None
        assert result["status"] == "success"
        assert result["destination"] == "Paris"
    
    def test_process_travel_request_validation_error(self, workflow):
        """Test travel request with validation errors"""
        invalid_request = {"destination": ""}  # Missing required fields
        
        result = workflow.process_travel_request(invalid_request)
        
        assert result is not None
        assert result["status"] == "error"
        assert len(result["errors"]) > 0
    
    def test_query_simple(self, workflow):
        """Test simple query processing"""
        with patch.object(workflow, 'process_travel_request') as mock_process:
            mock_process.return_value = {"status": "success", "destination": "Paris"}
            
            result = workflow.query("Trip to Paris")
            
            assert result is not None
            assert mock_process.called
            # Verify that process_travel_request was called with extracted destination
            call_args = mock_process.call_args[0][0]
            assert call_args["destination"] == "Paris"
    
    def test_query_no_destination(self, workflow):
        """Test query with no extractable destination"""
        result = workflow.query("I want to travel somewhere")
        
        assert result is not None
        assert result["status"] == "error"
        assert "destination" in result["errors"][0].lower()
    
    def test_extract_destination_from_query(self, workflow):
        """Test destination extraction from natural language"""
        test_cases = [
            ("Trip to Paris", "Paris"),
            ("Visit London for a weekend", "London"),
            ("Plan a trip to New York", "New York"),
            ("I want to go to Tokyo", "Tokyo"),
            ("Vacation in Barcelona", "Barcelona"),
            ("Random text", None)
        ]
        
        for query, expected in test_cases:
            result = workflow._extract_destination_from_query(query)
            assert result == expected
    
    @patch('travel_agent_system.core.workflow.APIClientManager')
    def test_execute_weather_agent_success(self, mock_api_class, workflow):
        """Test successful weather agent execution"""
        mock_api = mock_api_class.return_value
        mock_api.get_current_weather.return_value = {"temperature": 20}
        mock_api.get_weather_forecast.return_value = []
        
        state = TravelPlanState(destination="Paris")
        
        with patch.object(workflow.weather_agent, 'analyze_weather_for_travel') as mock_analyze:
            mock_analyze.return_value = {"temperature": 20, "forecast": []}
            
            result = workflow._execute_weather_agent(state)
            
            assert result is not None
            assert "temperature" in result
            mock_analyze.assert_called_once()
    
    @patch('travel_agent_system.core.workflow.APIClientManager')
    def test_execute_weather_agent_failure(self, mock_api_class, workflow):
        """Test weather agent execution failure"""
        state = TravelPlanState(destination="Paris")
        
        with patch.object(workflow.weather_agent, 'analyze_weather_for_travel') as mock_analyze:
            mock_analyze.side_effect = Exception("API Error")
            
            result = workflow._execute_weather_agent(state)
            
            assert result is not None
            assert "error" in result
            assert "API Error" in result["error"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
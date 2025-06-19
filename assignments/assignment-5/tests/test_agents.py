"""
Unit tests for all agent classes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_agent_system.agents.weather_agent import WeatherAgent
from travel_agent_system.agents.attraction_agent import AttractionAgent
from travel_agent_system.agents.hotel_agent import HotelAgent
from travel_agent_system.agents.itinerary_agent import ItineraryAgent


class TestWeatherAgent:
    """Test cases for WeatherAgent"""
    
    @pytest.fixture
    def weather_agent(self):
        return WeatherAgent()
    
    @pytest.fixture
    def mock_api_manager(self):
        with patch('travel_agent_system.agents.weather_agent.APIClientManager') as mock:
            yield mock.return_value
    
    def test_init(self, weather_agent):
        """Test WeatherAgent initialization"""
        assert weather_agent is not None
        assert hasattr(weather_agent, 'weather_history')
        assert weather_agent.weather_history == []
    
    @patch('travel_agent_system.agents.weather_agent.APIClientManager')
    def test_analyze_weather_success(self, mock_api_class, weather_agent):
        """Test successful weather analysis"""
        # Mock API responses
        mock_api = mock_api_class.return_value
        mock_api.get_current_weather.return_value = {
            'temperature': 22,
            'description': 'Clear sky',
            'humidity': 65
        }
        mock_api.get_weather_forecast.return_value = [
            {'date': '2024-01-01', 'temperature': {'avg': 20}, 'description': 'Sunny'},
            {'date': '2024-01-02', 'temperature': {'avg': 18}, 'description': 'Cloudy'}
        ]
        
        result = weather_agent.analyze_weather_for_travel(
            destination="Paris", 
            travel_dates={"start_date": "2024-01-01", "end_date": "2024-01-03"}
        )
        
        assert result is not None
        assert 'current_weather' in result
        assert 'forecast' in result
        assert 'travel_recommendations' in result
        assert 'packing_suggestions' in result
        assert len(weather_agent.weather_history) == 1
    
    def test_analyze_weather_no_destination(self, weather_agent):
        """Test weather analysis with missing destination"""
        result = weather_agent.analyze_weather_for_travel(destination="")
        
        assert result is not None
        assert 'error' in result
        assert 'required' in result['error'].lower()
    
    @patch('travel_agent_system.agents.weather_agent.APIClientManager')
    def test_analyze_weather_api_failure(self, mock_api_class, weather_agent):
        """Test weather analysis with API failure"""
        mock_api = mock_api_class.return_value
        mock_api.get_current_weather.side_effect = Exception("API Error")
        
        result = weather_agent.analyze_weather_for_travel(destination="Paris")
        
        assert result is not None
        assert 'error' in result
        assert 'api error' in result['error'].lower()


class TestAttractionAgent:
    """Test cases for AttractionAgent"""
    
    @pytest.fixture
    def attraction_agent(self):
        return AttractionAgent()
    
    def test_init(self, attraction_agent):
        """Test AttractionAgent initialization"""
        assert attraction_agent is not None
        assert hasattr(attraction_agent, 'search_history')
        assert attraction_agent.search_history == []
    
    @patch('travel_agent_system.agents.attraction_agent.APIClientManager')
    def test_discover_attractions_success(self, mock_api_class, attraction_agent):
        """Test successful attraction discovery"""
        mock_api = mock_api_class.return_value
        mock_api.search_places.return_value = [
            {'name': 'Eiffel Tower', 'category': 'attractions', 'rating': 9.5},
            {'name': 'Louvre Museum', 'category': 'attractions', 'rating': 9.0}
        ]
        
        result = attraction_agent.discover_attractions(
            destination="Paris",
            categories=["attractions"],
            budget_level="medium"
        )
        
        assert result is not None
        assert 'attractions' in result
        assert 'recommendations' in result
        assert len(result['attractions']) > 0
        assert len(attraction_agent.search_history) == 1
    
    def test_discover_attractions_no_destination(self, attraction_agent):
        """Test attraction discovery with missing destination"""
        result = attraction_agent.discover_attractions(destination="")
        
        assert result is not None
        assert 'error' in result
        assert 'required' in result['error'].lower()
    
    @patch('travel_agent_system.agents.attraction_agent.APIClientManager')
    def test_discover_attractions_api_failure(self, mock_api_class, attraction_agent):
        """Test attraction discovery with API failure"""
        mock_api = mock_api_class.return_value
        mock_api.search_places.side_effect = Exception("API Error")
        
        result = attraction_agent.discover_attractions(
            destination="Paris",
            categories=["attractions"]
        )
        
        assert result is not None
        assert 'error' in result


class TestHotelAgent:
    """Test cases for HotelAgent"""
    
    @pytest.fixture
    def hotel_agent(self):
        return HotelAgent()
    
    def test_init(self, hotel_agent):
        """Test HotelAgent initialization"""
        assert hotel_agent is not None
        assert hasattr(hotel_agent, 'search_history')
        assert hotel_agent.search_history == []
    
    def test_search_hotels_success(self, hotel_agent):
        """Test successful hotel search"""
        result = hotel_agent.search_hotels(
            destination="Paris",
            budget_range={"min": 100, "max": 200},
            travel_dates={"start_date": "2024-01-01", "end_date": "2024-01-03"}
        )
        
        assert result is not None
        assert 'hotel_options' in result
        assert 'cost_estimates' in result
        assert 'recommendations' in result
        assert len(result['hotel_options']) > 0
        assert len(hotel_agent.search_history) == 1
    
    def test_search_hotels_no_destination(self, hotel_agent):
        """Test hotel search with missing destination"""
        result = hotel_agent.search_hotels(destination="")
        
        assert result is not None
        assert 'error' in result
        assert 'required' in result['error'].lower()
    
    def test_search_hotels_invalid_budget(self, hotel_agent):
        """Test hotel search with invalid budget range"""
        result = hotel_agent.search_hotels(
            destination="Paris",
            budget_range={"min": 200, "max": 100}  # Invalid: min > max
        )
        
        assert result is not None
        # Should still work with fallback behavior


class TestItineraryAgent:
    """Test cases for ItineraryAgent"""
    
    @pytest.fixture
    def itinerary_agent(self):
        return ItineraryAgent()
    
    @pytest.fixture
    def sample_trip_data(self):
        return {
            'destination': 'Paris',
            'travel_dates': {
                'start_date': '2024-01-01T00:00:00Z',
                'end_date': '2024-01-03T00:00:00Z'
            },
            'weather_data': {
                'current_weather': {'temperature': 20, 'description': 'Clear'},
                'forecast': [
                    {'date': '2024-01-01', 'temperature': {'avg': 22}, 'description': 'Sunny'},
                    {'date': '2024-01-02', 'temperature': {'avg': 18}, 'description': 'Cloudy'}
                ]
            },
            'attractions_data': {
                'attractions': [
                    {'name': 'Eiffel Tower', 'category': 'attractions', 'rating': 9.5}
                ],
                'restaurants': [
                    {'name': 'Le Bernardin', 'category': 'restaurants', 'price_level': 'high'}
                ]
            },
            'hotels_data': {
                'hotel_options': [
                    {'name': 'Hotel Paris', 'price_per_night': 150, 'rating': 8.5}
                ]
            }
        }
    
    def test_init(self, itinerary_agent):
        """Test ItineraryAgent initialization"""
        assert itinerary_agent is not None
        assert hasattr(itinerary_agent, 'cost_calculator')
        assert hasattr(itinerary_agent, 'currency_converter')
        assert hasattr(itinerary_agent, 'itinerary_history')
        assert itinerary_agent.itinerary_history == []
    
    def test_create_itinerary_success(self, itinerary_agent, sample_trip_data):
        """Test successful itinerary creation"""
        result = itinerary_agent.create_itinerary(
            trip_data=sample_trip_data,
            preferences={'pace': 'moderate'}
        )
        
        assert result is not None
        assert 'destination' in result
        assert 'daily_plans' in result
        assert 'cost_breakdown' in result
        assert 'total_cost' in result
        assert 'recommendations' in result
        assert len(result['daily_plans']) > 0
        assert len(itinerary_agent.itinerary_history) == 1
    
    def test_create_itinerary_no_destination(self, itinerary_agent):
        """Test itinerary creation with missing destination"""
        result = itinerary_agent.create_itinerary(trip_data={})
        
        assert result is not None
        assert 'error' in result
        assert 'required' in result['error'].lower()
    
    def test_calculate_trip_days(self, itinerary_agent):
        """Test trip days calculation"""
        # Test with valid dates
        travel_dates = {
            'start_date': '2024-01-01T00:00:00Z',
            'end_date': '2024-01-03T00:00:00Z'
        }
        days = itinerary_agent._calculate_trip_days(travel_dates)
        assert days == 3
        
        # Test with no dates
        days = itinerary_agent._calculate_trip_days(None)
        assert days == 3  # Default
        
        # Test with invalid dates
        travel_dates = {
            'start_date': 'invalid',
            'end_date': 'invalid'
        }
        days = itinerary_agent._calculate_trip_days(travel_dates)
        assert days == 3  # Default fallback
    
    def test_get_itinerary_history(self, itinerary_agent, sample_trip_data):
        """Test itinerary history retrieval"""
        # Create an itinerary to populate history
        itinerary_agent.create_itinerary(sample_trip_data)
        
        history = itinerary_agent.get_itinerary_history()
        assert len(history) == 1
        assert history[0]['success'] is True
        assert history[0]['destination'] == 'Paris'
    
    def test_clear_history(self, itinerary_agent, sample_trip_data):
        """Test clearing itinerary history"""
        # Create an itinerary to populate history
        itinerary_agent.create_itinerary(sample_trip_data)
        assert len(itinerary_agent.itinerary_history) == 1
        
        # Clear history
        itinerary_agent.clear_history()
        assert len(itinerary_agent.itinerary_history) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
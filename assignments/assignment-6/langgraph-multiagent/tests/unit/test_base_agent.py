import pytest
from unittest.mock import Mock
from src.agents.base_agent import BaseAgent
from datetime import datetime

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""
    
    def process(self, state):
        return Mock()
    
    def get_required_fields(self):
        return ["test_field"]

class TestBaseAgent:
    """Test suite for BaseAgent class"""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization"""
        agent = ConcreteAgent("test_agent")
        assert agent.name == "test_agent"
        assert agent.model is None
        assert agent.metrics["total_calls"] == 0
        assert agent.metrics["successful_calls"] == 0
        assert agent.metrics["failed_calls"] == 0
        assert agent.metrics["total_processing_time"] == 0
    
    def test_validate_input_success(self):
        """Test successful input validation"""
        agent = ConcreteAgent("test_agent")
        state = {"test_field": "value"}
        assert agent.validate_input(state) is True
    
    def test_validate_input_failure(self):
        """Test failed input validation"""
        agent = ConcreteAgent("test_agent")
        state = {"wrong_field": "value"}
        assert agent.validate_input(state) is False
    
    def test_record_metrics(self):
        """Test metrics recording"""
        agent = ConcreteAgent("test_agent")
        start_time = datetime.now()
        
        agent.record_metrics(start_time, success=True)
        
        assert agent.metrics["total_calls"] == 1
        assert agent.metrics["successful_calls"] == 1
        assert agent.metrics["failed_calls"] == 0
        assert agent.metrics["total_processing_time"] > 0
    
    def test_get_metrics(self):
        """Test metrics retrieval"""
        agent = ConcreteAgent("test_agent")
        start_time = datetime.now()
        
        agent.record_metrics(start_time, success=True)
        metrics = agent.get_metrics()
        
        assert "average_processing_time" in metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] == 1.0
    
    def test_handle_error(self):
        """Test error handling"""
        agent = ConcreteAgent("test_agent")
        error = Exception("Test error")
        state = {"test": "value"}
        
        command = agent.handle_error(error, state)
        
        assert command.goto == "error_handler"
        assert "error" in command.update
        assert agent.metrics["failed_calls"] == 1
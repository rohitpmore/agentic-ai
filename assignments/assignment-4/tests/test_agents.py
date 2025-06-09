"""
Unit tests for individual agent classes
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_system.agents.supervisor import SupervisorAgent
from multi_agent_system.agents.llm_agent import LLMAgent
from multi_agent_system.agents.validation_agent import ValidationAgent
from multi_agent_system.core.state import AgentState


class TestSupervisorAgent(unittest.TestCase):
    """Test cases for the SupervisorAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.supervisor = SupervisorAgent(self.mock_model)
    
    @patch('multi_agent_system.agents.supervisor.get_topic_parser')
    def test_classify_query_usa_economy(self, mock_parser):
        """Test classification of USA Economy questions."""
        # Mock parser response
        mock_result = Mock()
        mock_result.Topic = "USA Economy"
        mock_result.Reasoning = "This is about US economic structure"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        self.supervisor.chain = mock_chain
        
        result = self.supervisor.classify_query("What is the US GDP structure?")
        
        self.assertEqual(result["topic"], "USA Economy")
        self.assertIn("economic", result["reasoning"])
    
    def test_route_decision_rag(self):
        """Test routing decision for RAG agent."""
        state = {"messages": ["USA Economy"]}
        decision = self.supervisor.route_decision(state)
        self.assertEqual(decision, "Call RAG")
    
    def test_route_decision_llm(self):
        """Test routing decision for LLM agent."""
        state = {"messages": ["General Knowledge"]}
        decision = self.supervisor.route_decision(state)
        self.assertEqual(decision, "Call LLM")
    
    def test_route_decision_web_crawler(self):
        """Test routing decision for Web Crawler agent."""
        state = {"messages": ["Real-time/Current Events"]}
        decision = self.supervisor.route_decision(state)
        self.assertEqual(decision, "Call Web Crawler")
    
    def test_route_decision_fallback(self):
        """Test fallback routing for unknown classifications."""
        state = {"messages": ["Unknown Category"]}
        decision = self.supervisor.route_decision(state)
        self.assertEqual(decision, "Call LLM")


class TestLLMAgent(unittest.TestCase):
    """Test cases for the LLMAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.llm_agent = LLMAgent(self.mock_model)
    
    def test_answer_question_success(self):
        """Test successful question answering."""
        # Mock model response
        mock_response = Mock()
        mock_response.content = "This is a test answer"
        self.mock_model.invoke.return_value = mock_response
        
        result = self.llm_agent.answer_question("What is 2+2?")
        
        self.assertEqual(result, "This is a test answer")
        self.mock_model.invoke.assert_called_once()
    
    def test_answer_question_error_handling(self):
        """Test error handling in question answering."""
        # Mock model to raise an exception
        self.mock_model.invoke.side_effect = Exception("API Error")
        
        result = self.llm_agent.answer_question("What is 2+2?")
        
        self.assertIn("technical issue", result)
        self.assertIn("LLM processing failed", result)
    
    def test_process_state(self):
        """Test processing of agent state."""
        # Mock model response
        mock_response = Mock()
        mock_response.content = "Test response"
        self.mock_model.invoke.return_value = mock_response
        
        state = {"messages": ["Test question"]}
        result = self.llm_agent.process(state)
        
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0], "Test response")
    
    def test_generate_explanation(self):
        """Test explanation generation with different detail levels."""
        mock_response = Mock()
        mock_response.content = "Detailed explanation"
        self.mock_model.invoke.return_value = mock_response
        
        result = self.llm_agent.generate_explanation("quantum physics", "detailed")
        
        self.assertEqual(result, "Detailed explanation")


class TestValidationAgent(unittest.TestCase):
    """Test cases for the ValidationAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.validation_agent = ValidationAgent(self.mock_model)
    
    def test_validate_response_pass(self):
        """Test successful response validation."""
        # Mock model to return PASS
        mock_response = Mock()
        mock_response.content = "PASS - The response is relevant and complete"
        self.mock_model.invoke.return_value = mock_response
        
        is_valid, reason = self.validation_agent.validate_response(
            "What is 2+2?", "2+2 equals 4"
        )
        
        self.assertTrue(is_valid)
        self.assertIn("PASS", reason)
    
    def test_validate_response_fail(self):
        """Test failed response validation."""
        # Mock model to return FAIL
        mock_response = Mock()
        mock_response.content = "FAIL - The response is not relevant"
        self.mock_model.invoke.return_value = mock_response
        
        is_valid, reason = self.validation_agent.validate_response(
            "What is 2+2?", "The sky is blue"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("FAIL", reason)
    
    def test_validate_response_error_fallback(self):
        """Test fallback validation when LLM fails."""
        # Mock model to raise exception
        self.mock_model.invoke.side_effect = Exception("API Error")
        
        is_valid, reason = self.validation_agent.validate_response(
            "What is 2+2?", "This is a sufficiently long response for fallback validation"
        )
        
        self.assertTrue(is_valid)  # Should pass fallback length check
        self.assertIn("Fallback validation", reason)
    
    def test_feedback_router_pass(self):
        """Test feedback router with passing validation."""
        state = {"messages": ["question", "response", "VALIDATION_PASS"]}
        decision = self.validation_agent.feedback_router(state)
        self.assertEqual(decision, "END")
    
    def test_feedback_router_retry(self):
        """Test feedback router with failed validation."""
        state = {"messages": ["question", "response", "VALIDATION_FAIL"]}
        decision = self.validation_agent.feedback_router(state)
        self.assertEqual(decision, "RETRY")
    
    def test_feedback_router_max_retries(self):
        """Test feedback router with maximum retries reached."""
        self.validation_agent.retry_count = self.validation_agent.max_retries
        state = {"messages": ["question", "response", "VALIDATION_FAIL"]}
        decision = self.validation_agent.feedback_router(state)
        self.assertEqual(decision, "END")
    
    def test_validate_query_format_valid(self):
        """Test query format validation with valid queries."""
        is_valid, message = self.validation_agent.validate_query_format("What is 2+2?")
        self.assertTrue(is_valid)
        self.assertEqual(message, "Query format is valid")
    
    def test_validate_query_format_empty(self):
        """Test query format validation with empty query."""
        is_valid, message = self.validation_agent.validate_query_format("")
        self.assertFalse(is_valid)
        self.assertIn("empty", message)
    
    def test_validate_query_format_too_short(self):
        """Test query format validation with too short query."""
        is_valid, message = self.validation_agent.validate_query_format("Hi")
        self.assertFalse(is_valid)
        self.assertIn("too short", message)
    
    def test_validate_query_format_too_long(self):
        """Test query format validation with too long query."""
        long_query = "x" * 1001
        is_valid, message = self.validation_agent.validate_query_format(long_query)
        self.assertFalse(is_valid)
        self.assertIn("too long", message)
    
    def test_retry_count_management(self):
        """Test retry count management."""
        self.assertEqual(self.validation_agent.get_retry_count(), 0)
        
        # Simulate a retry
        self.validation_agent.retry_count = 1
        self.assertEqual(self.validation_agent.get_retry_count(), 1)
        
        # Reset retry count
        self.validation_agent.reset_retry_count()
        self.assertEqual(self.validation_agent.get_retry_count(), 0)


if __name__ == '__main__':
    unittest.main() 
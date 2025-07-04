"""
Complete Integration Test Suite
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock


class TestFullWorkflowIntegration:
    """Test complete workflow integration"""
    
    @pytest.mark.asyncio
    async def test_research_to_reporting_handoff(self, test_settings):
        """Test handoff from research team to reporting team"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock external dependencies
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup mocks
            mock_search.return_value = [{"title": "Test Paper", "abstract": "Test", "url": "http://test.com", "relevance_score": 0.8}]
            mock_response = Mock()
            mock_response.content = "KEY FINDINGS:\n- Test finding\n\nSUMMARY:\nTest summary"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/document.pdf"
            
            # Initialize workflow
            workflow = MultiAgentWorkflow(test_settings)
            
            # Run workflow
            result = await workflow.run_workflow("Test research integration")
            
            # Verify handoff occurred
            assert "final_output" in result
            assert result.get("system_metrics", {}).get("success") is True
    
    @pytest.mark.asyncio
    async def test_parallel_research_execution(self, test_settings):
        """Test parallel execution of medical and financial research"""
        
        from src.utils.state_management import ParallelExecutionManager
        
        manager = ParallelExecutionManager()
        
        # Mock researchers
        medical_researcher = AsyncMock()
        medical_researcher.aprocess.return_value = Mock(
            update={"medical_findings": {"key_findings": ["Medical finding"]}}
        )
        
        financial_researcher = AsyncMock()
        financial_researcher.aprocess.return_value = Mock(
            update={"financial_findings": {"key_findings": ["Financial finding"]}}
        )
        
        research_state = {
            "research_topic": "Test parallel execution",
            "research_status": "in_progress"
        }
        
        # Execute in parallel
        result = await manager.execute_research_parallel(
            medical_researcher,
            financial_researcher,
            research_state
        )
        
        # Verify both executed
        medical_researcher.aprocess.assert_called_once()
        financial_researcher.aprocess.assert_called_once()
        
        # Verify results combined
        assert "medical_findings" in result
        assert "financial_findings" in result
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, test_settings):
        """Test error recovery across the workflow"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock API failure then success
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search:
            # First call fails, second succeeds
            mock_search.side_effect = [
                Exception("API Error"),
                [{"title": "Recovery Paper", "abstract": "Test", "url": "http://test.com", "relevance_score": 0.8}]
            ]
            
            with patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = "Recovered response"
                mock_invoke.return_value = mock_response
                
                workflow = MultiAgentWorkflow(test_settings)
                
                # This should handle the error and potentially retry
                result = await workflow.run_workflow("Test error recovery")
                
                # Should either succeed with recovery or fail gracefully
                assert "error" in result or "final_output" in result


class TestStateManagementIntegration:
    """Test state management across components"""
    
    def test_nested_state_updates(self):
        """Test nested state updates in supervisor state"""
        
        from src.utils.state_management import supervisor_state_reducer
        
        current_state = {
            "research_state": {
                "research_status": "pending",
                "medical_findings": {"key_findings": ["Finding 1"]}
            },
            "reporting_state": {
                "report_status": "pending"
            },
            "messages": ["Initial message"]
        }
        
        update = {
            "research_state": {
                "research_status": "completed",
                "medical_findings": {"key_findings": ["Finding 2"]}
            },
            "messages": ["Update message"]
        }
        
        result = supervisor_state_reducer(current_state, update)
        
        # Verify nested updates
        assert result["research_state"]["research_status"] == "completed"
        assert len(result["research_state"]["medical_findings"]["key_findings"]) == 2
        assert len(result["messages"]) == 2
    
    def test_concurrent_state_updates(self):
        """Test handling of concurrent state updates"""
        
        from src.utils.state_management import research_state_reducer
        
        # Simulate concurrent updates to the same state
        base_state = {
            "medical_findings": {"key_findings": ["Base finding"]},
            "financial_findings": {"key_findings": ["Base financial"]},
            "messages": ["Base message"]
        }
        
        # Two updates that might happen concurrently
        update1 = {
            "medical_findings": {"key_findings": ["Medical update 1"]},
            "messages": ["Message 1"]
        }
        
        update2 = {
            "financial_findings": {"key_findings": ["Financial update 1"]},
            "messages": ["Message 2"]
        }
        
        # Apply updates sequentially (simulating resolution)
        intermediate = research_state_reducer(base_state, update1)
        final = research_state_reducer(intermediate, update2)
        
        # Verify all updates preserved
        assert len(final["medical_findings"]["key_findings"]) == 2
        assert len(final["financial_findings"]["key_findings"]) == 2
        assert len(final["messages"]) == 3


class TestAPIIntegrationMocks:
    """Test API integrations with comprehensive mocking"""
    
    @pytest.mark.asyncio
    async def test_arxiv_api_integration(self, test_settings):
        """Test arXiv API integration with various response scenarios"""
        
        from src.tools.arxiv_tool import ArxivTool
        
        tool = ArxivTool()
        
        # Mock successful response
        with patch('arxiv.Search') as mock_search:
            mock_result = Mock()
            mock_result.title = "Test Paper"
            mock_result.authors = [Mock(name="Test Author")]
            mock_result.summary = "Test abstract"
            mock_result.entry_id = "http://arxiv.org/abs/test"
            
            mock_search.return_value.results.return_value = [mock_result]
            
            results = tool.search_papers("test query", "cs.AI", 5)
            
            assert len(results) == 1
            assert results[0]["title"] == "Test Paper"
    
    @pytest.mark.asyncio
    async def test_openai_api_integration(self, test_settings):
        """Test OpenAI API integration with error scenarios"""
        
        from langchain_openai import ChatOpenAI
        
        # Test successful call
        with patch.object(ChatOpenAI, 'invoke') as mock_invoke:
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_invoke.return_value = mock_response
            
            model = ChatOpenAI(model="gpt-4")
            response = model.invoke("Test prompt")
            
            assert response.content == "Test response"
        
        # Test API error handling
        with patch.object(ChatOpenAI, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("API Rate Limit")
            
            model = ChatOpenAI(model="gpt-4")
            
            with pytest.raises(Exception) as exc_info:
                model.invoke("Test prompt")
            
            assert "API Rate Limit" in str(exc_info.value)


class TestAgentCommunication:
    """Test agent-to-agent communication"""
    
    @pytest.mark.asyncio
    async def test_supervisor_to_researcher_handoff(self, test_settings):
        """Test supervisor handing off to researchers"""
        
        from src.agents.research.research_supervisor import ResearchTeamSupervisor
        from src.agents.research.medical_researcher import MedicalResearcher
        
        supervisor = ResearchTeamSupervisor(test_settings)
        researcher = MedicalResearcher(test_settings)
        
        # Mock handoff data
        handoff_data = {
            "research_topic": "Test medical research",
            "research_status": "pending"
        }
        
        # Test handoff creation
        from src.utils.handoff import HandoffProtocol
        handoff = HandoffProtocol.create_handoff(
            destination="medical_researcher",
            payload=handoff_data
        )
        
        assert handoff.goto == "medical_researcher"
        assert handoff.update["handoff_data"]["research_topic"] == "Test medical research"
    
    @pytest.mark.asyncio
    async def test_research_to_reporting_communication(self, test_settings):
        """Test communication from research to reporting team"""
        
        from src.agents.research.research_supervisor import ResearchTeamSupervisor
        from src.agents.reporting.reporting_supervisor import ReportingTeamSupervisor
        
        research_supervisor = ResearchTeamSupervisor(test_settings)
        reporting_supervisor = ReportingTeamSupervisor(test_settings)
        
        # Simulate completed research state
        research_state = {
            "research_status": "completed",
            "medical_findings": {"key_findings": ["Medical finding 1"]},
            "financial_findings": {"key_findings": ["Financial finding 1"]}
        }
        
        # Test handoff to reporting
        from src.utils.handoff import HandoffProtocol
        handoff = HandoffProtocol.create_handoff(
            destination="reporting",
            payload=research_state
        )
        
        assert handoff.goto == "reporting"
        assert "medical_findings" in handoff.update["handoff_data"]
        assert "financial_findings" in handoff.update["handoff_data"]


class TestWorkflowOrchestration:
    """Test LangGraph workflow orchestration"""
    
    @pytest.mark.asyncio
    async def test_workflow_graph_compilation(self, test_settings):
        """Test that the workflow graph compiles correctly"""
        
        from src.main import MultiAgentWorkflow
        
        workflow = MultiAgentWorkflow(test_settings)
        
        # Get the compiled graph
        compiled_graph = workflow.get_graph()
        
        # Verify graph structure
        assert compiled_graph is not None
        
        # Test graph visualization
        visualization = workflow.get_graph_visualization()
        assert isinstance(visualization, str)
        assert len(visualization) > 0
    
    @pytest.mark.asyncio
    async def test_conditional_routing(self, test_settings):
        """Test conditional routing in the workflow"""
        
        from src.agents.supervisor import MainSupervisor
        
        supervisor = MainSupervisor(test_settings)
        
        # Test different routing scenarios
        # Research pending
        route1 = supervisor._determine_routing(
            {"research_status": "pending"},
            {"report_status": "pending"}
        )
        assert route1 == "research"
        
        # Research complete, reporting pending
        route2 = supervisor._determine_routing(
            {"research_status": "completed"},
            {"report_status": "pending"}
        )
        assert route2 == "reporting"
        
        # Both complete
        route3 = supervisor._determine_routing(
            {"research_status": "completed"},
            {"report_status": "completed"}
        )
        assert route3 == "end"


class TestDataFlow:
    """Test data flow through the system"""
    
    def test_research_data_aggregation(self):
        """Test aggregation of research data"""
        
        from src.utils.state_management import research_state_reducer
        
        # Initial state
        state = {
            "medical_findings": {},
            "financial_findings": {},
            "messages": []
        }
        
        # Medical research update
        medical_update = {
            "medical_findings": {
                "key_findings": ["Medical finding 1", "Medical finding 2"],
                "research_complete": True
            },
            "messages": ["Medical research completed"]
        }
        
        state = research_state_reducer(state, medical_update)
        
        # Financial research update
        financial_update = {
            "financial_findings": {
                "key_findings": ["Financial finding 1"],
                "research_complete": True
            },
            "messages": ["Financial research completed"]
        }
        
        final_state = research_state_reducer(state, financial_update)
        
        # Verify aggregation
        assert len(final_state["medical_findings"]["key_findings"]) == 2
        assert len(final_state["financial_findings"]["key_findings"]) == 1
        assert len(final_state["messages"]) == 2
        assert final_state["medical_findings"]["research_complete"] is True
        assert final_state["financial_findings"]["research_complete"] is True
    
    def test_reporting_data_flow(self):
        """Test data flow through reporting pipeline"""
        
        from src.utils.state_management import reporting_state_reducer
        
        # Initial reporting state
        state = {
            "research_data": {},
            "report_status": "pending",
            "document_path": "",
            "summary": ""
        }
        
        # Document creation update
        doc_update = {
            "document_path": "/test/document.pdf",
            "report_status": "document_created"
        }
        
        state = reporting_state_reducer(state, doc_update)
        
        # Summary generation update
        summary_update = {
            "summary": "Executive summary of research findings",
            "report_status": "completed"
        }
        
        final_state = reporting_state_reducer(state, summary_update)
        
        # Verify flow
        assert final_state["document_path"] == "/test/document.pdf"
        assert final_state["summary"] == "Executive summary of research findings"
        assert final_state["report_status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__])
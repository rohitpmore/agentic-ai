import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.main import MultiAgentWorkflow
from config.settings import Settings

class TestWorkflowIntegration:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4",
            researcher_model="gpt-4",
            reporter_model="gpt-3.5-turbo",
            max_retries=3,
            output_directory="./test_outputs"
        )
    
    @pytest.fixture
    def workflow(self, settings):
        with patch('langchain_openai.ChatOpenAI'):
            return MultiAgentWorkflow(settings)
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, workflow):
        """Test complete workflow from start to finish"""
        
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(return_value=[
                {
                    "title": "Test Paper",
                    "authors": ["Test Author"],
                    "abstract": "Test abstract",
                    "url": "http://test.com",
                    "relevance_score": 0.8
                }
            ])
        ), patch.multiple(
            'langchain_openai.ChatOpenAI',
            invoke=Mock(return_value=Mock(content="Test LLM response"))
        ), patch.multiple(
            'src.tools.document_tools.DocumentTool',
            create_document=Mock(return_value="/test/document.pdf")
        ):
            
            task_description = "Research AI applications in healthcare and finance"
            
            result = await workflow.run_workflow(task_description)
            
            # Verify workflow completion
            assert "final_output" in result
            assert result["system_metrics"]["success"] is True
            
            # Verify research completion
            research_state = result.get("research_state", {})
            assert research_state.get("research_status") == "completed"
            assert "medical_findings" in research_state
            assert "financial_findings" in research_state
            
            # Verify reporting completion
            reporting_state = result.get("reporting_state", {})
            assert reporting_state.get("report_status") == "completed"
            assert "document_path" in reporting_state
            assert "summary" in reporting_state
    
    @pytest.mark.asyncio
    async def test_workflow_streaming(self, workflow):
        """Test streaming workflow execution"""
        
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(return_value=[])
        ), patch.multiple(
            'langchain_openai.ChatOpenAI',
            invoke=Mock(return_value=Mock(content="Streaming test"))
        ):
            
            task_description = "Test streaming workflow"
            chunks = []
            
            async for chunk in workflow.run_workflow_streaming(task_description):
                chunks.append(chunk)
                if len(chunks) >= 5:  # Limit for test
                    break
            
            assert len(chunks) > 0
            assert any("messages" in chunk for chunk in chunks if isinstance(chunk, dict))
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, workflow):
        """Test error handling throughout the workflow"""
        
        # Mock API failure
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(side_effect=Exception("API Error"))
        ):
            
            task_description = "Test error handling"
            
            result = await workflow.run_workflow(task_description)
            
            # Should handle error gracefully
            assert "error" in result or result.get("system_metrics", {}).get("success") is False
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, workflow):
        """Test state transitions between teams"""
        
        with patch.multiple(
            'src.agents.supervisor.MainSupervisor.process',
            return_value=Mock(
                goto="research_team",
                update={"current_team": "research"}
            )
        ):
            
            # Test initial routing to research team
            initial_state = {
                "task_description": "Test state transitions",
                "current_team": "research"
            }
            
            # Verify routing logic works
            routing_result = workflow._route_from_main_supervisor(initial_state)
            assert routing_result == "research_team"
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, workflow):
        """Test parallel execution within teams"""
        
        from src.utils.state_management import ParallelExecutionManager
        
        parallel_manager = ParallelExecutionManager()
        
        # Mock researchers
        medical_researcher = AsyncMock(return_value=Mock(
            update={"medical_findings": {"key_findings": ["Medical finding"]}}
        ))
        
        financial_researcher = AsyncMock(return_value=Mock(
            update={"financial_findings": {"key_findings": ["Financial finding"]}}
        ))
        
        research_state = {
            "research_topic": "Test parallel execution",
            "research_status": "in_progress"
        }
        
        result = await parallel_manager.execute_research_parallel(
            medical_researcher,
            financial_researcher,
            research_state
        )
        
        # Verify both researchers were called
        medical_researcher.assert_called_once()
        financial_researcher.assert_called_once()
        
        # Verify results are combined
        assert "medical_findings" in result
        assert "financial_findings" in result
    
    def test_graph_visualization(self, workflow):
        """Test graph visualization functionality"""
        
        # Test Mermaid diagram generation
        mermaid_diagram = workflow.get_graph_visualization()
        assert isinstance(mermaid_diagram, str)
        assert len(mermaid_diagram) > 0
        
        # Test graph image save (may fail in test environment)
        try:
            image_path = workflow.save_graph_image("test_graph.png")
            assert isinstance(image_path, str)
        except Exception:
            # Graph image save may fail in test environment
            pass

class TestStateReducers:
    def test_research_state_reducer(self):
        """Test research state reduction logic"""
        
        from src.utils.state_management import research_state_reducer
        
        current_state = {
            "medical_findings": {"key_findings": ["Finding 1"]},
            "financial_findings": {"key_findings": ["Finding 2"]},
            "messages": ["Message 1"]
        }
        
        update = {
            "medical_findings": {"key_findings": ["Finding 3"]},
            "messages": ["Message 2"]
        }
        
        result = research_state_reducer(current_state, update)
        
        # Verify findings are accumulated
        assert len(result["medical_findings"]["key_findings"]) == 2
        assert "Finding 1" in result["medical_findings"]["key_findings"]
        assert "Finding 3" in result["medical_findings"]["key_findings"]
        
        # Verify messages are added
        assert len(result["messages"]) == 2
    
    def test_supervisor_state_reducer(self):
        """Test supervisor state reduction logic"""
        
        from src.utils.state_management import supervisor_state_reducer
        
        current_state = {
            "research_state": {"research_status": "pending"},
            "reporting_state": {"report_status": "pending"},
            "system_metrics": {"start_time": "2023-01-01T00:00:00"}
        }
        
        update = {
            "research_state": {"research_status": "completed"},
            "system_metrics": {"end_time": "2023-01-01T01:00:00"}
        }
        
        result = supervisor_state_reducer(current_state, update)
        
        # Verify nested state updates
        assert result["research_state"]["research_status"] == "completed"
        assert result["reporting_state"]["report_status"] == "pending"
        
        # Verify metrics are merged
        assert "start_time" in result["system_metrics"]
        assert "end_time" in result["system_metrics"]

class TestGraphStructure:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4"
        )
    
    def test_main_graph_structure(self, settings):
        """Test main graph has correct structure"""
        
        with patch('langchain_openai.ChatOpenAI'):
            workflow = MultiAgentWorkflow(settings)
            graph = workflow.graph
            
            # Verify nodes exist
            nodes = list(graph.get_graph().nodes())
            expected_nodes = [
                "main_supervisor",
                "research_team",
                "reporting_team",
                "input_validator",
                "error_handler"
            ]
            
            for node in expected_nodes:
                assert node in nodes
    
    def test_research_team_graph_structure(self, settings):
        """Test research team subgraph structure"""
        
        with patch('langchain_openai.ChatOpenAI'):
            workflow = MultiAgentWorkflow(settings)
            research_graph = workflow._build_research_team_graph()
            
            # Verify research team nodes
            nodes = list(research_graph.get_graph().nodes())
            expected_nodes = [
                "research_supervisor",
                "medical_researcher",
                "financial_researcher",
                "research_validator"
            ]
            
            for node in expected_nodes:
                assert node in nodes
    
    def test_reporting_team_graph_structure(self, settings):
        """Test reporting team subgraph structure"""
        
        with patch('langchain_openai.ChatOpenAI'):
            workflow = MultiAgentWorkflow(settings)
            reporting_graph = workflow._build_reporting_team_graph()
            
            # Verify reporting team nodes
            nodes = list(reporting_graph.get_graph().nodes())
            expected_nodes = [
                "reporting_supervisor",
                "document_creator",
                "summarizer",
                "reporting_validator"
            ]
            
            for node in expected_nodes:
                assert node in nodes
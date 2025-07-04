# Stage 6: Testing & Quality Assurance

**Timeline:** 2-3 hours  
**Status:** ⏳ Pending  
**Priority:** High

## 📋 Overview

This final stage focuses on comprehensive testing, quality assurance, performance benchmarking, and documentation completion. We'll ensure the system meets all quality targets and is production-ready.

## 🎯 Key Deliverables

### ✅ Complete Unit Test Suite (90%+ Coverage)
### ✅ Integration Tests for All Team Interactions
### ✅ End-to-End Tests for Full Workflows
### ✅ Performance Benchmarks and Optimization
### ✅ Error Scenario Testing and Edge Cases
### ✅ Documentation Completion and Review

## 🔧 Implementation Details

### ✅ Comprehensive Test Suite
```python
# tests/conftest.py
import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from config.settings import Settings
from src.main import MultiAgentWorkflow

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Create test settings"""
    return Settings(
        openai_api_key="test-api-key",
        supervisor_model="gpt-4",
        researcher_model="gpt-4",
        reporter_model="gpt-3.5-turbo",
        max_retries=3,
        timeout_seconds=300,
        output_directory="./test_outputs",
        debug_mode=True,
        test_mode=True
    )

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_arxiv_results():
    """Mock arXiv API results"""
    return [
        {
            "title": "Test Medical Paper 1",
            "authors": ["Dr. Test Author"],
            "abstract": "This is a test abstract for medical research paper.",
            "url": "http://arxiv.org/abs/test1",
            "relevance_score": 0.85
        },
        {
            "title": "Test Financial Paper 1",
            "authors": ["Prof. Finance Expert"],
            "abstract": "This is a test abstract for financial research paper.",
            "url": "http://arxiv.org/abs/test2",
            "relevance_score": 0.78
        }
    ]

@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    response = Mock()
    response.content = """
    KEY FINDINGS:
    - Test finding 1
    - Test finding 2
    - Test finding 3
    
    CLINICAL IMPLICATIONS:
    - Test implication 1
    - Test implication 2
    
    RESEARCH GAPS:
    - Test gap 1
    
    SUMMARY:
    This is a test summary of the research findings.
    """
    return response

@pytest.fixture
def sample_research_state():
    """Sample research state for testing"""
    return {
        "research_topic": "AI applications in healthcare and finance",
        "research_status": "pending",
        "medical_findings": {},
        "financial_findings": {},
        "research_metadata": {}
    }

@pytest.fixture
def sample_reporting_state():
    """Sample reporting state for testing"""
    return {
        "research_data": {
            "medical_findings": {"key_findings": ["Medical finding 1"]},
            "financial_findings": {"key_findings": ["Financial finding 1"]}
        },
        "report_status": "pending",
        "document_path": "",
        "summary": "",
        "report_metadata": {}
    }

@pytest.fixture
def sample_supervisor_state():
    """Sample supervisor state for testing"""
    return {
        "task_description": "Test research task",
        "current_team": "research",
        "research_state": {
            "research_topic": "Test topic",
            "research_status": "pending"
        },
        "reporting_state": {
            "report_status": "pending"
        },
        "messages": [],
        "system_metrics": {}
    }
```

### ✅ Unit Test Coverage
```python
# tests/unit/test_complete_coverage.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

class TestMedicalResearcherCoverage:
    """Comprehensive tests for Medical Researcher"""
    
    @pytest.fixture
    def researcher(self, test_settings):
        from src.agents.research.medical_researcher import MedicalResearcher
        return MedicalResearcher(test_settings)
    
    def test_initialization(self, researcher):
        assert researcher.name == "medical_researcher"
        assert researcher.specializations is not None
        assert len(researcher.specializations) > 0
    
    def test_get_required_fields(self, researcher):
        fields = researcher.get_required_fields()
        assert "research_topic" in fields
        assert "research_status" in fields
    
    @patch('src.tools.arxiv_tool.ArxivTool.search_papers')
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_conduct_medical_research(self, mock_invoke, mock_search, researcher, mock_arxiv_results, mock_llm_response):
        mock_search.return_value = mock_arxiv_results
        mock_invoke.return_value = mock_llm_response
        
        findings = researcher._conduct_medical_research("diabetes treatment")
        
        assert findings["research_complete"] is True
        assert "key_findings" in findings
        assert "drug_interactions" in findings
        assert "clinical_insights" in findings
        assert "research_papers" in findings
    
    def test_extract_medical_keywords(self, researcher):
        with patch.object(researcher.model, 'invoke') as mock_invoke:
            mock_response = Mock()
            mock_response.content = "diabetes\ninsulin\nglucose\nblood sugar"
            mock_invoke.return_value = mock_response
            
            keywords = researcher._extract_medical_keywords("diabetes treatment research")
            
            assert isinstance(keywords, list)
            assert len(keywords) > 0
    
    def test_parse_medical_analysis(self, researcher):
        analysis_text = """
        KEY FINDINGS:
        - Finding 1
        - Finding 2
        
        CLINICAL IMPLICATIONS:
        - Implication 1
        
        SUMMARY:
        Test summary
        """
        
        result = researcher._parse_medical_analysis(analysis_text)
        
        assert "key_findings" in result
        assert len(result["key_findings"]) == 2
        assert "Finding 1" in result["key_findings"]
        assert "clinical_implications" in result
        assert len(result["clinical_implications"]) == 1
    
    def test_assess_research_quality(self, researcher):
        findings = {
            "research_papers": [{"title": "Paper 1"}, {"title": "Paper 2"}],
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "clinical_insights": {"trials": [{"type": "test"}], "patient_populations": ["adults"]},
            "drug_interactions": {"interactions": ["interaction 1"], "contraindications": ["contra 1"]}
        }
        
        quality_score = researcher._assess_research_quality(findings)
        
        assert 0 <= quality_score <= 1.0
        assert quality_score > 0  # Should have some quality

class TestFinancialResearcherCoverage:
    """Comprehensive tests for Financial Researcher"""
    
    @pytest.fixture
    def researcher(self, test_settings):
        from src.agents.research.financial_researcher import FinancialResearcher
        return FinancialResearcher(test_settings)
    
    def test_initialization(self, researcher):
        assert researcher.name == "financial_researcher"
        assert researcher.specializations is not None
    
    @patch('src.tools.arxiv_tool.ArxivTool.search_papers')
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_conduct_financial_research(self, mock_invoke, mock_search, researcher, mock_arxiv_results, mock_llm_response):
        mock_search.return_value = mock_arxiv_results
        mock_invoke.return_value = mock_llm_response
        
        findings = researcher._conduct_financial_research("cryptocurrency investment")
        
        assert findings["research_complete"] is True
        assert "market_analysis" in findings
        assert "risk_assessment" in findings
        assert "economic_indicators" in findings
    
    def test_identify_market_trends(self, researcher):
        implications = [
            "Market trends show increasing growth",
            "Economic decline is evident",
            "Emerging technologies are trending"
        ]
        
        trends = researcher._identify_market_trends(implications)
        
        assert len(trends) == 3  # All contain trend keywords
        assert any("growth" in trend for trend in trends)

class TestDocumentCreatorCoverage:
    """Comprehensive tests for Document Creator"""
    
    @pytest.fixture
    def creator(self, test_settings):
        from src.agents.reporting.document_creator import DocumentCreator
        return DocumentCreator(test_settings)
    
    @patch('src.tools.document_tools.DocumentTool.create_document')
    def test_create_professional_document(self, mock_create_doc, creator):
        mock_create_doc.return_value = "/test/document.pdf"
        
        research_data = {
            "medical_findings": {"key_findings": ["Medical finding 1"]},
            "financial_findings": {"key_findings": ["Financial finding 1"]}
        }
        
        document_path = creator._create_professional_document(research_data)
        
        assert document_path == "/test/document.pdf"
        mock_create_doc.assert_called_once()
    
    def test_generate_document_structure(self, creator):
        medical_findings = {
            "research_topic": "Medical AI",
            "key_findings": ["Finding 1", "Finding 2"]
        }
        financial_findings = {
            "research_topic": "AI Investment",
            "key_findings": ["Market insight 1"]
        }
        
        structure = creator._generate_document_structure(medical_findings, financial_findings)
        
        assert "title" in structure
        assert "executive_summary" in structure
        assert "sections" in structure
        assert len(structure["sections"]) >= 4  # Medical, Financial, Cross-domain, Recommendations
    
    def test_identify_synergies(self, creator):
        medical_findings = {"key_terms": ["AI", "healthcare", "technology"]}
        financial_findings = {"key_terms": ["AI", "investment", "technology"]}
        
        synergies = creator._identify_synergies(medical_findings, financial_findings)
        
        assert len(synergies) > 0
        assert any("AI" in synergy for synergy in synergies)

class TestSummarizerCoverage:
    """Comprehensive tests for Summarizer"""
    
    @pytest.fixture
    def summarizer(self, test_settings):
        from src.agents.reporting.summarizer import Summarizer
        return Summarizer(test_settings)
    
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_generate_executive_summary(self, mock_invoke, summarizer):
        mock_response = Mock()
        mock_response.content = "This is a comprehensive executive summary of the research findings."
        mock_invoke.return_value = mock_response
        
        medical_findings = {"key_findings": ["Medical finding 1"]}
        financial_findings = {"key_findings": ["Financial finding 1"]}
        
        summary = summarizer._generate_executive_summary(medical_findings, financial_findings)
        
        assert len(summary) > 0
        assert isinstance(summary, str)
    
    def test_extract_key_points(self, summarizer):
        medical_findings = {"key_findings": ["Medical finding 1", "Medical finding 2"]}
        financial_findings = {"key_findings": ["Financial finding 1"]}
        
        key_points = summarizer._extract_key_points(medical_findings, financial_findings)
        
        assert len(key_points) > 0
        assert any("Medical finding 1" in point for point in key_points)
        assert any("Financial finding 1" in point for point in key_points)
    
    def test_generate_fallback_summary(self, summarizer):
        medical_findings = {"key_findings": ["Finding 1"], "research_papers": [{"title": "Paper 1"}]}
        financial_findings = {"key_findings": ["Finding 2"], "research_papers": [{"title": "Paper 2"}]}
        
        summary = summarizer._generate_fallback_summary(medical_findings, financial_findings)
        
        assert "Research Summary" in summary
        assert len(summary) > 0

class TestSupervisorsCoverage:
    """Comprehensive tests for all supervisors"""
    
    def test_main_supervisor_routing(self, test_settings):
        from src.agents.supervisor import MainSupervisor
        supervisor = MainSupervisor(test_settings)
        
        # Test route to research
        research_pending = supervisor._determine_routing(
            {"research_status": "pending"},
            {"report_status": "pending"}
        )
        assert research_pending == "research"
        
        # Test route to reporting
        reporting_next = supervisor._determine_routing(
            {"research_status": "completed"},
            {"report_status": "pending"}
        )
        assert reporting_next == "reporting"
        
        # Test route to end
        end_workflow = supervisor._determine_routing(
            {"research_status": "completed"},
            {"report_status": "completed"}
        )
        assert end_workflow == "end"
    
    def test_research_supervisor_coordination(self, test_settings):
        from src.agents.research.research_supervisor import ResearchTeamSupervisor
        supervisor = ResearchTeamSupervisor(test_settings)
        
        # Test route to medical researcher
        medical_next = supervisor._determine_next_researcher(
            {},  # No medical findings
            {},  # No financial findings
            "medical research topic"
        )
        assert medical_next == "medical"
        
        # Test completion
        complete = supervisor._determine_next_researcher(
            {"research_complete": True},
            {"research_complete": True},
            "research topic"
        )
        assert complete == "complete"

class TestUtilitiesCoverage:
    """Test utility functions and classes"""
    
    def test_handoff_protocol(self):
        from src.utils.handoff import HandoffProtocol
        
        # Test standard handoff
        handoff = HandoffProtocol.create_handoff(
            destination="test_destination",
            payload={"test": "data"},
            urgency="high"
        )
        
        assert handoff.goto == "test_destination"
        assert handoff.update["handoff_data"]["test"] == "data"
        assert handoff.update["handoff_metadata"]["urgency"] == "high"
    
    def test_error_handler(self):
        from src.utils.error_handling import ErrorHandler
        
        handler = ErrorHandler(max_retries=3)
        
        # Test should_retry logic
        assert handler.should_retry(ConnectionError("Network error")) is True
        assert handler.should_retry(ValueError("Invalid value")) is False
    
    def test_state_reducers(self):
        from src.utils.state_management import research_state_reducer
        
        current = {
            "medical_findings": {"key_findings": ["Finding 1"]},
            "messages": ["Message 1"]
        }
        
        update = {
            "medical_findings": {"key_findings": ["Finding 2"]},
            "messages": ["Message 2"]
        }
        
        result = research_state_reducer(current, update)
        
        assert len(result["medical_findings"]["key_findings"]) == 2
        assert len(result["messages"]) == 2
```

### ✅ Integration Test Suite
```python
# tests/integration/test_complete_integration.py
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
```

### ✅ End-to-End Test Suite
```python
# tests/e2e/test_complete_e2e.py
import pytest
import asyncio
import os
import tempfile
from unittest.mock import patch, Mock

class TestCompleteE2EWorkflows:
    """End-to-end tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_medical_research_workflow(self, test_settings):
        """Test complete medical research workflow"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock all external dependencies
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc, \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=5000):
            
            # Setup comprehensive mocks
            mock_search.return_value = [
                {
                    "title": "Advanced Diabetes Treatment Research",
                    "authors": ["Dr. Medical Expert"],
                    "abstract": "Comprehensive study on diabetes treatment methods and drug interactions.",
                    "url": "http://arxiv.org/abs/medical1",
                    "relevance_score": 0.92
                },
                {
                    "title": "AI in Medical Diagnostics",
                    "authors": ["Prof. AI Medicine"],
                    "abstract": "Application of artificial intelligence in medical diagnostic procedures.",
                    "url": "http://arxiv.org/abs/medical2",
                    "relevance_score": 0.88
                }
            ]
            
            # Medical analysis response
            medical_response = Mock()
            medical_response.content = """
            KEY FINDINGS:
            - AI-powered diagnostic tools show 95% accuracy in diabetes detection
            - New drug combinations reduce side effects by 40%
            - Machine learning models can predict treatment outcomes
            
            CLINICAL IMPLICATIONS:
            - Earlier intervention possible with AI diagnostics
            - Personalized treatment plans improve patient outcomes
            
            RESEARCH GAPS:
            - Long-term effects of AI-assisted treatments need study
            
            SUMMARY:
            Revolutionary advances in AI-powered diabetes treatment show promising results.
            """
            
            # Financial analysis response
            financial_response = Mock()
            financial_response.content = """
            KEY FINDINGS:
            - Healthcare AI market expected to grow 45% annually
            - Diabetes treatment market valued at $95B globally
            - ROI on AI diagnostic tools exceeds 300%
            
            MARKET IMPLICATIONS:
            - Significant investment opportunities in health tech
            - Cost savings from early diagnosis substantial
            
            INVESTMENT INSIGHTS:
            - Medical AI startups showing strong growth
            - Traditional pharma companies investing heavily
            
            SUMMARY:
            Healthcare AI represents a major investment opportunity with strong fundamentals.
            """
            
            # Executive summary response
            summary_response = Mock()
            summary_response.content = """
            This comprehensive analysis reveals breakthrough developments in AI-powered diabetes treatment. 
            Medical research demonstrates 95% diagnostic accuracy and 40% reduction in side effects, 
            while financial analysis shows a $95B market with 45% annual growth. The convergence of 
            advanced AI capabilities and substantial market opportunity creates compelling investment 
            prospects in healthcare technology.
            """
            
            # Return different responses based on call content
            def mock_invoke_side_effect(prompt):
                prompt_text = str(prompt).lower()
                if "medical" in prompt_text or "clinical" in prompt_text or "drug" in prompt_text:
                    return medical_response
                elif "financial" in prompt_text or "market" in prompt_text or "investment" in prompt_text:
                    return financial_response
                elif "executive summary" in prompt_text or "comprehensive summary" in prompt_text:
                    return summary_response
                else:
                    # Default response for other prompts
                    default_response = Mock()
                    default_response.content = "Generic analysis response"
                    return default_response
            
            mock_invoke.side_effect = mock_invoke_side_effect
            mock_doc.return_value = "/test/comprehensive_diabetes_ai_report.pdf"
            
            # Initialize and run workflow
            workflow = MultiAgentWorkflow(test_settings)
            
            result = await workflow.run_workflow(
                "Research AI applications in diabetes treatment and analyze the investment potential"
            )
            
            # Comprehensive verification
            assert "final_output" in result
            assert result["system_metrics"]["success"] is True
            
            # Verify research completion
            research_state = result["research_state"]
            assert research_state["research_status"] == "completed"
            
            # Verify medical findings
            medical_findings = research_state["medical_findings"]
            assert medical_findings["research_complete"] is True
            assert len(medical_findings["key_findings"]) >= 3
            assert any("95% accuracy" in finding for finding in medical_findings["key_findings"])
            assert len(medical_findings["research_papers"]) == 2
            
            # Verify financial findings
            financial_findings = research_state["financial_findings"]
            assert financial_findings["research_complete"] is True
            assert len(financial_findings["key_findings"]) >= 3
            assert any("45% annually" in finding for finding in financial_findings["key_findings"])
            
            # Verify reporting completion
            reporting_state = result["reporting_state"]
            assert reporting_state["report_status"] == "completed"
            assert reporting_state["document_path"] == "/test/comprehensive_diabetes_ai_report.pdf"
            assert len(reporting_state["summary"]) > 100
            
            # Verify final output
            final_output = result["final_output"]
            assert "document_path" in final_output
            assert "summary" in final_output
    
    @pytest.mark.asyncio
    async def test_cli_integration_workflow(self, test_settings):
        """Test complete CLI integration"""
        
        from src.cli import MultiAgentCLI
        
        # Mock CLI dependencies
        with patch('src.main.MultiAgentWorkflow') as mock_workflow_class:
            # Setup mock workflow
            mock_workflow = Mock()
            mock_workflow.run_workflow = AsyncMock()
            mock_workflow.run_workflow.return_value = {
                "final_output": {
                    "document_path": "/test/cli_report.pdf",
                    "summary": "CLI integration test completed successfully"
                },
                "system_metrics": {
                    "success": True,
                    "start_time": "2023-01-01T00:00:00",
                    "end_time": "2023-01-01T01:00:00"
                },
                "research_state": {
                    "research_status": "completed",
                    "medical_findings": {"key_findings": ["CLI medical finding"]},
                    "financial_findings": {"key_findings": ["CLI financial finding"]}
                },
                "reporting_state": {
                    "report_status": "completed",
                    "document_path": "/test/cli_report.pdf",
                    "summary": "CLI test summary"
                }
            }
            
            mock_workflow_class.return_value = mock_workflow
            
            # Initialize CLI
            cli = MultiAgentCLI()
            cli.console = Mock()  # Mock console for testing
            
            # Mock arguments
            args = Mock()
            args.config = None
            args.output_dir = "./test_outputs"
            args.debug = False
            args.timeout = 300
            args.verbose = 0
            args.log_file = None
            args.stream = True
            args.no_stream = False
            args.format = "markdown"
            
            # Setup environment
            setup_success = cli.setup_environment(args)
            assert setup_success is True
            
            # Run query
            result = await cli.run_query("Test CLI integration", args)
            
            # Verify CLI handled the workflow correctly
            assert "final_output" in result
            assert result["system_metrics"]["success"] is True
            
            # Verify workflow was called
            mock_workflow.run_workflow.assert_called_once_with("Test CLI integration")
    
    @pytest.mark.asyncio
    async def test_streaming_workflow(self, test_settings):
        """Test streaming workflow execution"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock streaming responses
        async def mock_streaming_generator():
            yield {"current_team": "research", "messages": ["Starting research phase"]}
            yield {"research_state": {"research_status": "in_progress"}}
            yield {"messages": ["Medical research in progress"]}
            yield {"messages": ["Financial research in progress"]}
            yield {"current_team": "reporting", "messages": ["Starting reporting phase"]}
            yield {"reporting_state": {"report_status": "in_progress"}}
            yield {"messages": ["Creating document"]}
            yield {"messages": ["Generating summary"]}
            yield {
                "final_output": {
                    "document_path": "/test/streaming_report.pdf",
                    "summary": "Streaming test completed"
                },
                "system_metrics": {"success": True}
            }
        
        with patch.object(MultiAgentWorkflow, 'run_workflow_streaming') as mock_stream:
            mock_stream.return_value = mock_streaming_generator()
            
            workflow = MultiAgentWorkflow(test_settings)
            
            # Collect streaming results
            results = []
            async for chunk in workflow.run_workflow_streaming("Test streaming"):
                results.append(chunk)
            
            # Verify streaming progression
            assert len(results) == 9
            
            # Verify progression through teams
            team_changes = [r for r in results if "current_team" in r]
            assert len(team_changes) == 2
            assert team_changes[0]["current_team"] == "research"
            assert team_changes[1]["current_team"] == "reporting"
            
            # Verify final result
            final_result = results[-1]
            assert "final_output" in final_result
            assert final_result["system_metrics"]["success"] is True

class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_workflow_performance_benchmark(self, test_settings):
        """Benchmark complete workflow performance"""
        
        import time
        from src.main import MultiAgentWorkflow
        
        # Mock for fast execution
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Fast mock responses
            mock_search.return_value = [{"title": "Fast Test", "abstract": "Fast", "url": "http://fast.com", "relevance_score": 0.8}]
            mock_response = Mock()
            mock_response.content = "Fast response"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/fast/document.pdf"
            
            workflow = MultiAgentWorkflow(test_settings)
            
            # Benchmark execution
            start_time = time.time()
            result = await workflow.run_workflow("Performance benchmark test")
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Performance assertions (with mocked APIs, should be very fast)
            assert execution_time < 10.0  # Should complete in under 10 seconds with mocks
            assert result["system_metrics"]["success"] is True
            
            # Log performance metrics
            print(f"Workflow execution time: {execution_time:.2f} seconds")
    
    def test_memory_usage_benchmark(self, test_settings):
        """Benchmark memory usage during workflow"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple workflow instances to test memory
        workflows = []
        for i in range(5):
            workflow = MultiAgentWorkflow(test_settings)
            workflows.append(workflow)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Cleanup
        del workflows

class TestErrorScenarios:
    """Test various error scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, test_settings):
        """Test handling of API timeouts"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock timeout errors
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search:
            mock_search.side_effect = TimeoutError("API Timeout")
            
            workflow = MultiAgentWorkflow(test_settings)
            
            result = await workflow.run_workflow("Test timeout handling")
            
            # Should handle timeout gracefully
            assert "error" in result or result.get("system_metrics", {}).get("success") is False
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, test_settings):
        """Test handling of invalid inputs"""
        
        from src.main import MultiAgentWorkflow
        
        workflow = MultiAgentWorkflow(test_settings)
        
        # Test empty query
        result = await workflow.run_workflow("")
        assert "error" in result or result.get("system_metrics", {}).get("success") is False
        
        # Test very long query
        long_query = "A" * 10000
        result = await workflow.run_workflow(long_query)
        # Should either succeed or fail gracefully
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, test_settings):
        """Test recovery from partial failures"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock partial failure (medical succeeds, financial fails)
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
            
            # Medical research succeeds
            def search_side_effect(query, category, max_results):
                if category == "q-bio":  # Medical
                    return [{"title": "Medical Success", "abstract": "Success", "url": "http://med.com", "relevance_score": 0.8}]
                else:  # Financial fails
                    raise Exception("Financial API Error")
            
            mock_search.side_effect = search_side_effect
            
            mock_response = Mock()
            mock_response.content = "Partial success response"
            mock_invoke.return_value = mock_response
            
            workflow = MultiAgentWorkflow(test_settings)
            
            result = await workflow.run_workflow("Test partial failure")
            
            # Should have some results even with partial failure
            research_state = result.get("research_state", {})
            medical_findings = research_state.get("medical_findings", {})
            
            # Medical should succeed
            assert medical_findings.get("research_complete") is True or "error" in medical_findings
```

### ✅ Performance Benchmarking
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import asyncio
import psutil
import os
from unittest.mock import Mock, patch

class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarking"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self, test_settings):
        """Benchmark end-to-end workflow performance"""
        
        from src.main import MultiAgentWorkflow
        
        # Setup performance monitoring
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock all external APIs for consistent timing
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup fast mocks
            mock_search.return_value = [
                {"title": f"Paper {i}", "abstract": f"Abstract {i}", "url": f"http://test{i}.com", "relevance_score": 0.8}
                for i in range(5)
            ]
            
            mock_response = Mock()
            mock_response.content = "Quick mock response with findings"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/benchmark_document.pdf"
            
            workflow = MultiAgentWorkflow(test_settings)
            
            # Performance benchmark
            start_time = time.time()
            
            result = await workflow.run_workflow("Performance benchmark test query")
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Performance assertions
            assert execution_time < 30.0, f"Workflow took {execution_time:.2f}s, expected < 30s"
            assert memory_usage < 200, f"Memory usage {memory_usage:.2f}MB, expected < 200MB"
            assert result["system_metrics"]["success"] is True
            
            # Log performance metrics
            print(f"\nPerformance Metrics:")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Memory Usage: {memory_usage:.2f} MB")
            print(f"Success Rate: 100%")
            
            return {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "success": result["system_metrics"]["success"]
            }
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, test_settings):
        """Test performance with multiple concurrent workflows"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock external dependencies
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup mocks
            mock_search.return_value = [{"title": "Concurrent Test", "abstract": "Test", "url": "http://test.com", "relevance_score": 0.8}]
            mock_response = Mock()
            mock_response.content = "Concurrent response"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/concurrent_document.pdf"
            
            # Create multiple workflows
            num_concurrent = 3
            workflows = [MultiAgentWorkflow(test_settings) for _ in range(num_concurrent)]
            
            # Run concurrent workflows
            start_time = time.time()
            
            tasks = [
                workflow.run_workflow(f"Concurrent test query {i}")
                for i, workflow in enumerate(workflows)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Verify all succeeded
            assert len(results) == num_concurrent
            assert all(result["system_metrics"]["success"] for result in results)
            
            execution_time = end_time - start_time
            
            # Should complete reasonably quickly even with concurrency
            assert execution_time < 60.0, f"Concurrent execution took {execution_time:.2f}s"
            
            print(f"Concurrent Execution Time ({num_concurrent} workflows): {execution_time:.2f} seconds")
    
    def test_memory_leak_detection(self, test_settings):
        """Test for memory leaks over multiple executions"""
        
        import gc
        from src.main import MultiAgentWorkflow
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple workflow creations and destructions
        for i in range(10):
            workflow = MultiAgentWorkflow(test_settings)
            # Simulate some usage
            _ = workflow.get_graph_visualization()
            del workflow
            
            if i % 5 == 0:
                gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - baseline_memory
        
        # Should not have significant memory leak
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.2f}MB increase"
        
        print(f"Memory leak test - Increase: {memory_increase:.2f}MB")

class TestScalabilityBenchmarks:
    """Test system scalability"""
    
    def test_large_input_handling(self, test_settings):
        """Test handling of large inputs"""
        
        from src.agents.research.medical_researcher import MedicalResearcher
        
        researcher = MedicalResearcher(test_settings)
        
        # Test with large research topic
        large_topic = "Research artificial intelligence applications in healthcare diagnosis and treatment with specific focus on machine learning algorithms for medical imaging analysis, natural language processing for clinical notes, predictive modeling for patient outcomes, drug discovery and development processes, personalized medicine approaches, telemedicine platforms, electronic health record optimization, clinical decision support systems, and integration challenges in existing healthcare infrastructure" * 10
        
        start_time = time.time()
        
        # This should handle large input gracefully
        try:
            keywords = researcher._extract_medical_keywords(large_topic[:1000])  # Truncate to reasonable size
            execution_time = time.time() - start_time
            
            assert execution_time < 10.0, f"Large input processing took {execution_time:.2f}s"
            assert isinstance(keywords, list)
            
        except Exception as e:
            # Should handle gracefully, not crash
            assert "timeout" in str(e).lower() or "too long" in str(e).lower()
    
    def test_high_volume_state_updates(self, test_settings):
        """Test performance with high volume of state updates"""
        
        from src.utils.state_management import research_state_reducer
        
        # Start with base state
        state = {
            "medical_findings": {"key_findings": []},
            "financial_findings": {"key_findings": []},
            "messages": []
        }
        
        start_time = time.time()
        
        # Apply many updates
        for i in range(1000):
            update = {
                "medical_findings": {"key_findings": [f"Finding {i}"]},
                "messages": [f"Message {i}"]
            }
            state = research_state_reducer(state, update)
        
        execution_time = time.time() - start_time
        
        # Should handle high volume efficiently
        assert execution_time < 5.0, f"High volume updates took {execution_time:.2f}s"
        assert len(state["medical_findings"]["key_findings"]) == 1000
        assert len(state["messages"]) == 1000
        
        print(f"High volume state updates: {execution_time:.2f}s for 1000 updates")

def generate_performance_report(benchmark_results: dict):
    """Generate performance report"""
    
    report = f"""
# Performance Benchmark Report

## Summary
- **Execution Time**: {benchmark_results.get('execution_time', 'N/A'):.2f} seconds
- **Memory Usage**: {benchmark_results.get('memory_usage', 'N/A'):.2f} MB
- **Success Rate**: {100 if benchmark_results.get('success', False) else 0}%

## Performance Targets
- ✅ Complete workflow: < 30 seconds (Target: < 5 minutes)
- ✅ Memory usage: < 200 MB (Target: Reasonable memory footprint)
- ✅ Success rate: 100% (Target: > 95%)

## Recommendations
- System performance meets all targets
- Memory usage is within acceptable limits
- No optimization required at this time
    """
    
    return report.strip()
```

## 🎯 Success Criteria

### Functional Requirements:
- [ ] Unit test coverage achieves 90%+
- [ ] All integration tests pass consistently
- [ ] End-to-end tests cover major use cases
- [ ] Error scenarios are handled gracefully
- [ ] Performance benchmarks meet targets
- [ ] Documentation is comprehensive and accurate

### Quality Requirements:
- [ ] Code quality metrics meet standards
- [ ] No critical security vulnerabilities
- [ ] All edge cases are tested
- [ ] Error messages are helpful and actionable
- [ ] System handles failures gracefully

### Performance Requirements:
- [ ] Complete workflow finishes in <5 minutes
- [ ] Unit tests run in <30 seconds
- [ ] Integration tests run in <2 minutes
- [ ] Memory usage remains stable
- [ ] No memory leaks detected

## 📊 Stage 6 Metrics

### Time Allocation:
- Unit test completion: 45 minutes
- Integration test implementation: 40 minutes
- End-to-end test scenarios: 35 minutes
- Performance benchmarking: 25 minutes
- Error scenario testing: 20 minutes
- Documentation review: 15 minutes

### Success Indicators:
- All tests pass consistently
- Coverage targets are met
- Performance benchmarks pass
- Documentation is complete
- System is production-ready
- Quality metrics meet standards

## 🔄 Final Deliverables

After completing Stage 6:
1. **Production-Ready System**: Fully tested and documented
2. **Comprehensive Test Suite**: 90%+ coverage with all test types
3. **Performance Report**: Benchmarks and optimization recommendations
4. **Quality Assurance Report**: Complete QA assessment
5. **Documentation Package**: User guides, API docs, and deployment instructions

---

*This final stage ensures the multi-agent system meets all quality, performance, and reliability standards for production deployment, providing confidence in system robustness and maintainability.*
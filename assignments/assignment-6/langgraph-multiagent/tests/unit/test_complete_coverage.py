"""
Comprehensive Unit Test Coverage
"""

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
        assert quality_score > 0


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
        
        assert len(trends) == 3
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
        assert len(structure["sections"]) >= 4
    
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
            {},
            {},
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


class TestErrorHandlingCoverage:
    """Test error handling across all components"""
    
    def test_medical_researcher_error_handling(self, test_settings):
        from src.agents.research.medical_researcher import MedicalResearcher
        researcher = MedicalResearcher(test_settings)
        
        # Test API failure handling
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search:
            mock_search.side_effect = Exception("API Error")
            
            findings = researcher._conduct_medical_research("test query")
            
            # Should handle error gracefully
            assert "error" in findings or findings.get("research_complete") is False
    
    def test_financial_researcher_error_handling(self, test_settings):
        from src.agents.research.financial_researcher import FinancialResearcher
        researcher = FinancialResearcher(test_settings)
        
        # Test invalid input handling
        findings = researcher._conduct_financial_research("")
        
        # Should handle empty input gracefully
        assert "error" in findings or findings.get("research_complete") is False
    
    def test_document_creator_error_handling(self, test_settings):
        from src.agents.reporting.document_creator import DocumentCreator
        creator = DocumentCreator(test_settings)
        
        # Test empty research data
        result = creator._create_professional_document({})
        
        # Should handle gracefully
        assert result is None or "error" in str(result)
    
    def test_summarizer_error_handling(self, test_settings):
        from src.agents.reporting.summarizer import Summarizer
        summarizer = Summarizer(test_settings)
        
        # Test fallback summary generation
        summary = summarizer._generate_fallback_summary({}, {})
        
        # Should provide fallback
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestConfigurationCoverage:
    """Test configuration management"""
    
    def test_settings_validation(self):
        from config.settings import Settings
        
        # Test valid settings
        settings = Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4"
        )
        assert settings.openai_api_key == "test-key"
        assert settings.supervisor_model == "gpt-4"
    
    def test_settings_defaults(self):
        from config.settings import Settings
        
        settings = Settings(openai_api_key="test-key")
        
        # Should have reasonable defaults
        assert settings.max_retries > 0
        assert settings.timeout_seconds > 0
        assert settings.output_directory is not None


class TestStateSchemaCoverage:
    """Test state schema validation"""
    
    def test_research_state_schema(self):
        from src.state.schemas import ResearchState
        
        # Test valid state
        state = ResearchState(
            research_topic="test topic",
            research_status="pending",
            medical_findings={},
            financial_findings={},
            research_metadata={}
        )
        
        assert state.research_topic == "test topic"
        assert state.research_status == "pending"
    
    def test_supervisor_state_schema(self):
        from src.state.schemas import SupervisorState
        
        # Test valid state
        state = SupervisorState(
            task_description="test task",
            current_team="research",
            research_state={
                "research_topic": "test",
                "research_status": "pending",
                "medical_findings": {},
                "financial_findings": {},
                "research_metadata": {}
            },
            reporting_state={
                "research_data": {},
                "report_status": "pending",
                "document_path": "",
                "summary": "",
                "report_metadata": {}
            },
            messages=[],
            system_metrics={}
        )
        
        assert state.task_description == "test task"
        assert state.current_team == "research"


if __name__ == "__main__":
    pytest.main([__file__])
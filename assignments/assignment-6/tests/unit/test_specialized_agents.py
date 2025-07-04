import pytest
from unittest.mock import Mock, patch, MagicMock
from config.settings import Settings


class TestMedicalResearcher:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            researcher_model="gpt-4",
            max_retries=3
        )
    
    @patch('src.agents.research.medical_researcher.ChatOpenAI')
    @patch('src.agents.research.medical_researcher.ArxivTool')
    @patch('src.agents.research.medical_researcher.ErrorHandler')
    def test_medical_research_process(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_arxiv_instance = MagicMock()
        mock_arxiv_tool.return_value = mock_arxiv_instance
        mock_arxiv_instance.search_papers.return_value = [
            {
                "title": "Medical AI Research",
                "authors": ["Author 1"],
                "abstract": "Abstract content",
                "url": "http://example.com",
                "relevance_score": 0.8
            }
        ]
        
        mock_error_handler_instance = MagicMock()
        mock_error_handler.return_value = mock_error_handler_instance
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        KEY FINDINGS:
        - Finding 1
        - Finding 2
        
        CLINICAL IMPLICATIONS:
        - Implication 1
        
        RESEARCH GAPS:
        - Gap 1
        
        SUMMARY:
        Test summary
        """
        mock_openai_instance.invoke.return_value = mock_response
        
        # Import and create researcher
        from src.agents.research.medical_researcher import MedicalResearcher
        researcher = MedicalResearcher(settings)
        
        state = {
            "research_topic": "AI in medical diagnostics",
            "research_status": "pending"
        }
        
        result = researcher.process(state)
        
        assert result.goto == "research_supervisor"
        assert "medical_findings" in result.update
        assert result.update["medical_findings"]["research_complete"] is True

    @patch('src.agents.research.medical_researcher.ChatOpenAI')
    @patch('src.agents.research.medical_researcher.ArxivTool')
    @patch('src.agents.research.medical_researcher.ErrorHandler')
    def test_medical_researcher_required_fields(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        from src.agents.research.medical_researcher import MedicalResearcher
        researcher = MedicalResearcher(settings)
        fields = researcher.get_required_fields()
        assert "research_topic" in fields
        assert "research_status" in fields

    @patch('src.agents.research.medical_researcher.ChatOpenAI')
    @patch('src.agents.research.medical_researcher.ArxivTool')
    @patch('src.agents.research.medical_researcher.ErrorHandler')
    def test_medical_researcher_specializations(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        from src.agents.research.medical_researcher import MedicalResearcher
        researcher = MedicalResearcher(settings)
        assert "drug_interactions" in researcher.specializations
        assert "clinical_trials" in researcher.specializations
        assert "pharmaceutical_development" in researcher.specializations


class TestFinancialResearcher:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            researcher_model="gpt-4",
            max_retries=3
        )
    
    @patch('src.agents.research.financial_researcher.ChatOpenAI')
    @patch('src.agents.research.financial_researcher.ArxivTool')
    @patch('src.agents.research.financial_researcher.ErrorHandler')
    def test_financial_research_process(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_arxiv_instance = MagicMock()
        mock_arxiv_tool.return_value = mock_arxiv_instance
        mock_arxiv_instance.search_papers.return_value = [
            {
                "title": "Financial AI Research",
                "authors": ["Author 1"],
                "abstract": "Financial abstract",
                "url": "http://example.com",
                "relevance_score": 0.9
            }
        ]
        
        mock_error_handler_instance = MagicMock()
        mock_error_handler.return_value = mock_error_handler_instance
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        KEY FINDINGS:
        - Market finding 1
        - Market finding 2
        
        MARKET IMPLICATIONS:
        - Implication 1
        
        INVESTMENT INSIGHTS:
        - Insight 1
        
        SUMMARY:
        Financial summary
        """
        mock_openai_instance.invoke.return_value = mock_response
        
        # Import and create researcher
        from src.agents.research.financial_researcher import FinancialResearcher
        researcher = FinancialResearcher(settings)
        
        state = {
            "research_topic": "AI in financial markets",
            "research_status": "pending"
        }
        
        result = researcher.process(state)
        
        assert result.goto == "research_supervisor"
        assert "financial_findings" in result.update
        assert result.update["financial_findings"]["research_complete"] is True

    @patch('src.agents.research.financial_researcher.ChatOpenAI')
    @patch('src.agents.research.financial_researcher.ArxivTool')
    @patch('src.agents.research.financial_researcher.ErrorHandler')
    def test_financial_researcher_required_fields(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        from src.agents.research.financial_researcher import FinancialResearcher
        researcher = FinancialResearcher(settings)
        fields = researcher.get_required_fields()
        assert "research_topic" in fields
        assert "research_status" in fields

    @patch('src.agents.research.financial_researcher.ChatOpenAI')
    @patch('src.agents.research.financial_researcher.ArxivTool')
    @patch('src.agents.research.financial_researcher.ErrorHandler')
    def test_financial_researcher_specializations(self, mock_error_handler, mock_arxiv_tool, mock_openai, settings):
        from src.agents.research.financial_researcher import FinancialResearcher
        researcher = FinancialResearcher(settings)
        assert "market_analysis" in researcher.specializations
        assert "investment_strategies" in researcher.specializations
        assert "risk_assessment" in researcher.specializations


class TestDocumentCreator:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            reporter_model="gpt-3.5-turbo",
            output_directory="./test_outputs"
        )
    
    @patch('src.agents.reporting.document_creator.ChatOpenAI')
    @patch('src.agents.reporting.document_creator.DocumentTool')
    @patch('src.agents.reporting.document_creator.ErrorHandler')
    def test_document_creation(self, mock_error_handler, mock_document_tool, mock_openai, settings):
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock LLM responses for all the enhanced content generation
        mock_response = Mock()
        mock_response.content = "Enhanced Professional Research Report"
        mock_openai_instance.invoke.return_value = mock_response
        
        mock_document_tool_instance = MagicMock()
        mock_document_tool.return_value = mock_document_tool_instance
        mock_document_tool_instance.create_document.return_value = "/path/to/document.pdf"
        
        mock_error_handler_instance = MagicMock()
        mock_error_handler.return_value = mock_error_handler_instance
        
        # Import and create document creator
        from src.agents.reporting.document_creator import DocumentCreator
        creator = DocumentCreator(settings)
        
        state = {
            "research_data": {
                "medical_findings": {"key_findings": ["Finding 1"], "research_topic": "Medical AI"},
                "financial_findings": {"key_findings": ["Finding 2"], "research_topic": "AI Investment"}
            },
            "report_status": "pending"
        }
        
        result = creator.process(state)
        
        assert result.goto == "reporting_supervisor"
        assert "document_path" in result.update
        assert result.update["document_path"] == "/path/to/document.pdf"
        
        # Verify LLM was called for content enhancement
        assert mock_openai_instance.invoke.call_count >= 1

    @patch('src.agents.reporting.document_creator.ChatOpenAI')
    @patch('src.agents.reporting.document_creator.DocumentTool')
    @patch('src.agents.reporting.document_creator.ErrorHandler')
    def test_document_creator_required_fields(self, mock_error_handler, mock_document_tool, mock_openai, settings):
        from src.agents.reporting.document_creator import DocumentCreator
        creator = DocumentCreator(settings)
        fields = creator.get_required_fields()
        assert "research_data" in fields
        assert "report_status" in fields

    @patch('src.agents.reporting.document_creator.ChatOpenAI')
    @patch('src.agents.reporting.document_creator.DocumentTool')
    @patch('src.agents.reporting.document_creator.ErrorHandler')
    def test_document_structure_generation(self, mock_error_handler, mock_document_tool, mock_openai, settings):
        # Mock LLM responses
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_response = Mock()
        mock_response.content = "Enhanced Professional Report Title"
        mock_openai_instance.invoke.return_value = mock_response
        
        from src.agents.reporting.document_creator import DocumentCreator
        creator = DocumentCreator(settings)
        
        medical_findings = {"research_topic": "Medical Topic", "key_findings": ["Finding 1"]}
        financial_findings = {"research_topic": "Financial Topic", "key_findings": ["Finding 2"]}
        
        structure = creator._generate_document_structure(medical_findings, financial_findings)
        
        assert "title" in structure
        assert "executive_summary" in structure
        assert "sections" in structure
        assert len(structure["sections"]) == 4
        
        # Verify LLM was called for title and summary generation
        assert mock_openai_instance.invoke.call_count >= 2


class TestSummarizer:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            reporter_model="gpt-3.5-turbo"
        )
    
    @patch('src.agents.reporting.summarizer.ChatOpenAI')
    @patch('src.agents.reporting.summarizer.ErrorHandler')
    def test_summary_creation(self, mock_error_handler, mock_openai, settings):
        # Setup mocks
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        mock_error_handler_instance = MagicMock()
        mock_error_handler.return_value = mock_error_handler_instance
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a comprehensive summary of the research findings."
        mock_openai_instance.invoke.return_value = mock_response
        
        # Import and create summarizer
        from src.agents.reporting.summarizer import Summarizer
        summarizer = Summarizer(settings)
        
        state = {
            "research_data": {
                "medical_findings": {"key_findings": ["Medical finding 1", "Medical finding 2"]},
                "financial_findings": {"key_findings": ["Financial finding 1", "Financial finding 2"]}
            },
            "report_status": "pending"
        }
        
        result = summarizer.process(state)
        
        assert result.goto == "reporting_supervisor"
        assert "summary" in result.update
        assert len(result.update["summary"]) > 0

    @patch('src.agents.reporting.summarizer.ChatOpenAI')
    @patch('src.agents.reporting.summarizer.ErrorHandler')
    def test_summarizer_required_fields(self, mock_error_handler, mock_openai, settings):
        from src.agents.reporting.summarizer import Summarizer
        summarizer = Summarizer(settings)
        fields = summarizer.get_required_fields()
        assert "research_data" in fields
        assert "report_status" in fields

    @patch('src.agents.reporting.summarizer.ChatOpenAI')
    @patch('src.agents.reporting.summarizer.ErrorHandler')
    def test_key_points_extraction(self, mock_error_handler, mock_openai, settings):
        from src.agents.reporting.summarizer import Summarizer
        summarizer = Summarizer(settings)
        
        medical_findings = {"key_findings": ["Medical finding 1", "Medical finding 2"]}
        financial_findings = {"key_findings": ["Financial finding 1", "Financial finding 2"]}
        
        key_points = summarizer._extract_key_points(medical_findings, financial_findings)
        
        assert len(key_points) > 0
        assert any("Medical research identified" in point for point in key_points)
        assert any("Financial analysis revealed" in point for point in key_points)

    @patch('src.agents.reporting.summarizer.ChatOpenAI')
    @patch('src.agents.reporting.summarizer.ErrorHandler')
    def test_fallback_summary_generation(self, mock_error_handler, mock_openai, settings):
        from src.agents.reporting.summarizer import Summarizer
        summarizer = Summarizer(settings)
        
        medical_findings = {"key_findings": ["Finding 1"]}
        financial_findings = {"key_findings": ["Finding 2"]}
        
        summary = summarizer._generate_fallback_summary(medical_findings, financial_findings)
        
        assert "Research Summary" in summary
        assert "Medical research identified" in summary
        assert "Financial analysis revealed" in summary


class TestTools:
    """Test the tools independently"""
    
    def test_arxiv_tool_structure(self):
        from src.tools.arxiv_tool import ArxivTool
        tool = ArxivTool()
        assert hasattr(tool, 'search_papers')
        assert hasattr(tool, '_calculate_relevance')
    
    def test_document_tool_structure(self):
        # Mock settings for testing
        class MockSettings:
            def __init__(self):
                self.output_directory = "./test_outputs"
                
            def ensure_output_directory(self):
                import os
                os.makedirs(self.output_directory, exist_ok=True)
        
        from src.tools.document_tools import DocumentTool
        settings = MockSettings()
        tool = DocumentTool(settings)
        assert hasattr(tool, 'create_document')
        assert hasattr(tool, '_generate_filename')
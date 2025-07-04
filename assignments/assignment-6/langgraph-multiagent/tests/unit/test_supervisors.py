import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.supervisor import MainSupervisor
from src.agents.research.research_supervisor import ResearchTeamSupervisor
from src.agents.reporting.reporting_supervisor import ReportingTeamSupervisor
from config.settings import Settings


class TestMainSupervisor:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4",
            max_retries=3
        )
    
    @pytest.fixture
    @patch('src.agents.supervisor.ChatOpenAI')
    def supervisor(self, mock_openai, settings):
        mock_openai.return_value = MagicMock()
        return MainSupervisor(settings)
    
    def test_initialization(self, supervisor):
        assert supervisor.name == "main_supervisor"
        assert supervisor.settings is not None
        
    def test_route_to_research_team(self, supervisor):
        state = {
            "task_description": "Research AI in healthcare",
            "research_state": {"research_status": "pending"},
            "reporting_state": {"report_status": "pending"}
        }
        
        result = supervisor.process(state)
        assert result.goto == "research_team"
        assert result.update["current_team"] == "research"
        
    def test_route_to_reporting_team(self, supervisor):
        state = {
            "task_description": "Create report",
            "research_state": {"research_status": "completed"},
            "reporting_state": {"report_status": "pending"}
        }
        
        result = supervisor.process(state)
        assert result.goto == "reporting_team"
        assert result.update["current_team"] == "reporting"
        
    def test_route_to_end(self, supervisor):
        state = {
            "task_description": "Complete workflow",
            "research_state": {"research_status": "completed"},
            "reporting_state": {"report_status": "completed", "document_path": "test.pdf", "summary": "Test summary"}
        }
        
        result = supervisor.process(state)
        assert result.goto == "end"
        assert "final_output" in result.update


class TestResearchTeamSupervisor:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4"
        )
    
    @pytest.fixture
    @patch('src.agents.research.research_supervisor.ChatOpenAI')
    def supervisor(self, mock_openai, settings):
        mock_openai.return_value = MagicMock()
        return ResearchTeamSupervisor(settings)
    
    @patch('src.agents.research.research_supervisor.ChatOpenAI')
    def test_route_to_medical_researcher(self, mock_openai, supervisor):
        mock_response = Mock()
        mock_response.content = "- Next Researcher: medical\n- Priority: high\n- Reasoning: Medical AI applications require specialized medical domain knowledge"
        supervisor.model.invoke.return_value = mock_response
        
        state = {
            "research_topic": "medical AI applications",
            "research_status": "pending",
            "medical_findings": {},
            "financial_findings": {}
        }
        
        result = supervisor.process(state)
        assert result.goto == "medical_researcher"
        assert result.update["current_researcher"] == "medical"
        
    @patch('src.agents.research.research_supervisor.ChatOpenAI')
    def test_route_to_financial_researcher(self, mock_openai, supervisor):
        mock_response = Mock()
        mock_response.content = "- Next Researcher: financial\n- Priority: high\n- Reasoning: AI in finance requires specialized financial domain expertise"
        supervisor.model.invoke.return_value = mock_response
        
        state = {
            "research_topic": "AI in finance",
            "research_status": "pending",
            "medical_findings": {"research_complete": True},
            "financial_findings": {}
        }
        
        result = supervisor.process(state)
        assert result.goto == "financial_researcher"
        assert result.update["current_researcher"] == "financial"
        
    def test_complete_research(self, supervisor):
        state = {
            "research_topic": "AI applications",
            "research_status": "in_progress",
            "medical_findings": {"research_complete": True, "key_findings": ["finding1"]},
            "financial_findings": {"research_complete": True, "key_findings": ["finding2"]}
        }
        
        result = supervisor.process(state)
        assert result.goto == "main_supervisor"
        assert result.update["research_status"] == "completed"


class TestReportingTeamSupervisor:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4"
        )
    
    @pytest.fixture
    @patch('src.agents.reporting.reporting_supervisor.ChatOpenAI')
    def supervisor(self, mock_openai, settings):
        mock_openai.return_value = MagicMock()
        return ReportingTeamSupervisor(settings)
    
    def test_route_to_document_creator(self, supervisor):
        state = {
            "research_data": {"medical_findings": {}, "financial_findings": {}},
            "report_status": "pending",
            "document_path": "",
            "summary": ""
        }
        
        result = supervisor.process(state)
        assert result.goto == "document_creator"
        assert result.update["current_reporter"] == "document_creator"
        
    def test_route_to_summarizer(self, supervisor):
        state = {
            "research_data": {"medical_findings": {}, "financial_findings": {}},
            "report_status": "in_progress",
            "document_path": "/path/to/document.pdf",
            "summary": ""
        }
        
        result = supervisor.process(state)
        assert result.goto == "summarizer"
        assert result.update["current_reporter"] == "summarizer"
        
    def test_complete_reporting(self, supervisor):
        state = {
            "research_data": {"medical_findings": {}, "financial_findings": {}},
            "report_status": "in_progress",
            "document_path": "/path/to/document.pdf",
            "summary": "This is a comprehensive summary of the research findings."
        }
        
        result = supervisor.process(state)
        assert result.goto == "main_supervisor"
        assert result.update["report_status"] == "completed"
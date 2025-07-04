from typing import Dict, Any, Literal, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.state.schemas import ReportingState
from src.utils.handoff import HandoffProtocol
from config.settings import Settings


class ReportingTeamSupervisor(BaseAgent):
    """Supervisor for the reporting team coordinating document creation and summarization"""
    
    def __init__(self, settings: Settings):
        super().__init__("reporting_team_supervisor")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.supervisor_model,
            temperature=0,
            timeout=settings.timeout_seconds
        )
        self.handoff_protocol = HandoffProtocol()
        
    def get_required_fields(self) -> List[str]:
        return ["research_data", "report_status"]
        
    def process(self, state: ReportingState) -> Command[Literal["document_creator", "summarizer", "main_supervisor"]]:
        """Process reporting state and coordinate between document creator and summarizer"""
        
        start_time = datetime.now()
        
        try:
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid reporting state"),
                    state
                )
            
            research_data = state.get("research_data", {})
            document_path = state.get("document_path", "")
            summary = state.get("summary", "")
            
            self.logger.info(f"Coordinating reporting for research data")
            
            next_action = self._determine_next_reporter(
                document_path,
                summary,
                research_data
            )
            
            command = self._create_reporting_command(
                next_action,
                research_data,
                document_path,
                summary
            )
            
            self.record_metrics(start_time, success=True)
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _determine_next_reporter(
        self,
        document_path: str,
        summary: str,
        research_data: Dict[str, Any]
    ) -> Literal["document", "summary", "complete"]:
        """Determine which reporter should work next"""
        
        document_complete = bool(document_path and len(document_path) > 0)
        summary_complete = bool(summary and len(summary) > 0)
        
        if document_complete and summary_complete:
            return "complete"
        
        if not document_complete:
            return "document"
        
        if not summary_complete:
            return "summary"
        
        return "document"
    
    def _create_reporting_command(
        self,
        next_action: Literal["document", "summary", "complete"],
        research_data: Dict[str, Any],
        document_path: str,
        summary: str
    ) -> Command:
        """Create command for next reporting action"""
        
        if next_action == "document":
            return Command(
                goto="document_creator",
                update={
                    "report_status": "in_progress",
                    "current_reporter": "document_creator",
                    "report_metadata": {
                        "assigned_timestamp": datetime.now().isoformat(),
                        "task_type": "document_creation",
                        "data_complexity": self._assess_data_complexity(research_data)
                    }
                }
            )
        
        elif next_action == "summary":
            return Command(
                goto="summarizer",
                update={
                    "report_status": "in_progress",
                    "current_reporter": "summarizer",
                    "report_metadata": {
                        "assigned_timestamp": datetime.now().isoformat(),
                        "task_type": "summarization",
                        "source_document": document_path
                    }
                }
            )
        
        else:
            return Command(
                goto="main_supervisor",
                update={
                    "report_status": "completed",
                    "completion_timestamp": datetime.now().isoformat(),
                    "final_deliverables": {
                        "document_path": document_path,
                        "summary": summary,
                        "quality_metrics": self._calculate_quality_metrics(
                            document_path,
                            summary,
                            research_data
                        )
                    }
                },
                graph=Command.PARENT
            )
    
    def _assess_data_complexity(self, research_data: Dict[str, Any]) -> str:
        """Assess complexity of research data for document creation"""
        
        medical_findings = research_data.get("medical_findings", {})
        financial_findings = research_data.get("financial_findings", {})
        
        medical_points = len(medical_findings.get("key_findings", []))
        financial_points = len(financial_findings.get("key_findings", []))
        total_points = medical_points + financial_points
        
        if total_points > 20:
            return "high"
        elif total_points > 10:
            return "medium"
        else:
            return "low"
    
    def _calculate_quality_metrics(
        self,
        document_path: str,
        summary: str,
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for completed reporting"""
        
        return {
            "document_generated": bool(document_path),
            "summary_generated": bool(summary),
            "summary_length": len(summary) if summary else 0,
            "data_coverage": self._calculate_data_coverage(research_data),
            "completion_score": self._calculate_completion_score(
                document_path,
                summary,
                research_data
            ),
            "processing_time": self.get_metrics().get("total_processing_time", 0)
        }
    
    def _calculate_data_coverage(self, research_data: Dict[str, Any]) -> float:
        """Calculate how well the research data is covered"""
        
        medical_findings = research_data.get("medical_findings", {})
        financial_findings = research_data.get("financial_findings", {})
        
        medical_coverage = 1.0 if medical_findings.get("research_complete") else 0.5
        financial_coverage = 1.0 if financial_findings.get("research_complete") else 0.5
        
        return (medical_coverage + financial_coverage) / 2.0
    
    def _calculate_completion_score(
        self,
        document_path: str,
        summary: str,
        research_data: Dict[str, Any]
    ) -> float:
        """Calculate overall completion score"""
        
        score = 0.0
        
        if document_path:
            score += 0.5
        
        if summary and len(summary) > 100:
            score += 0.3
        
        data_coverage = self._calculate_data_coverage(research_data)
        score += 0.2 * data_coverage
        
        return score
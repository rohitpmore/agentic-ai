from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from datetime import datetime

class ResearchState(MessagesState):
    """State for research team operations"""
    research_topic: str
    medical_findings: Dict[str, Any]
    financial_findings: Dict[str, Any]
    research_status: Literal["pending", "in_progress", "completed", "failed"]
    research_metadata: Dict[str, Any]
    
class ReportingState(MessagesState):
    """State for reporting team operations"""
    research_data: Dict[str, Any]
    document_path: str
    summary: str
    report_status: Literal["pending", "in_progress", "completed", "failed"]
    report_metadata: Dict[str, Any]
    
class SupervisorState(MessagesState):
    """Main supervisor state combining all team states"""
    current_team: Literal["research", "reporting", "end"]
    research_state: ResearchState
    reporting_state: ReportingState
    task_description: str
    final_output: Dict[str, Any]
    error_state: Optional[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    
class AgentMetrics(TypedDict):
    """Metrics for individual agents"""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime]
    tokens_used: int
    success: bool
    error_message: Optional[str]
    processing_time: float
    
class SystemMetrics(TypedDict):
    """System-wide metrics"""
    total_agents: int
    active_agents: int
    total_processing_time: float
    successful_handoffs: int
    failed_handoffs: int
    documents_generated: int
    api_calls_made: int
# Stage 2: Core Agent Development

**Timeline:** 4-5 hours  
**Status:** ⏳ Pending  
**Priority:** High

## 📋 Overview

This stage focuses on implementing the core supervisor agents that orchestrate the multi-agent workflow. We'll build the Main Supervisor, Research Team Supervisor, and Reporting Team Supervisor with LLM-based intelligent routing, Command-based communication, and comprehensive error handling.

## 🎯 Key Deliverables

### ✅ Main Supervisor Agent
### ✅ Research Team Supervisor  
### ✅ Reporting Team Supervisor
### ✅ Command-based Routing Logic
### ✅ Error Handling and Retry Mechanisms
### ✅ Unit Tests for All Supervisors

## 🔧 Implementation Details

### ✅ Main Supervisor Agent
```python
# src/agents/supervisor.py
from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import logging

from src.agents.base_agent import BaseAgent
from src.state.schemas import SupervisorState
from src.utils.handoff import HandoffProtocol
from config.settings import Settings

class MainSupervisor(BaseAgent):
    """Main supervisor agent that orchestrates between research and reporting teams"""
    
    def __init__(self, settings: Settings):
        super().__init__("main_supervisor")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.supervisor_model,
            temperature=0,
            timeout=settings.timeout_seconds
        )
        self.handoff_protocol = HandoffProtocol()
        
    def get_required_fields(self) -> List[str]:
        return ["task_description", "research_state", "reporting_state"]
        
    def process(self, state: SupervisorState) -> Command[Literal["research_team", "reporting_team", "end"]]:
        """Process supervisor state and route to appropriate team"""
        
        start_time = datetime.now()
        
        try:
            # Validate input state
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid input state"),
                    state
                )
            
            # Extract state information
            task_description = state.get("task_description", "")
            research_state = state.get("research_state", {})
            reporting_state = state.get("reporting_state", {})
            
            # Log current state
            self.logger.info(f"Processing task: {task_description}")
            
            # Determine routing based on state
            routing_decision = self._determine_routing(research_state, reporting_state)
            
            # Create appropriate command
            command = self._create_routing_command(
                routing_decision,
                task_description,
                research_state,
                reporting_state
            )
            
            # Record successful processing
            self.record_metrics(start_time, success=True)
            
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _determine_routing(
        self,
        research_state: Dict[str, Any],
        reporting_state: Dict[str, Any]
    ) -> Literal["research", "reporting", "end"]:
        """Determine which team to route to based on current state"""
        
        research_status = research_state.get("research_status", "pending")
        reporting_status = reporting_state.get("report_status", "pending")
        
        # Route to research team if not completed
        if research_status != "completed":
            return "research"
        
        # Route to reporting team if research is done but reporting is not
        elif reporting_status != "completed":
            return "reporting"
        
        # End workflow if both teams are completed
        else:
            return "end"
    
    def _create_routing_command(
        self,
        destination: Literal["research", "reporting", "end"],
        task_description: str,
        research_state: Dict[str, Any],
        reporting_state: Dict[str, Any]
    ) -> Command:
        """Create routing command based on destination"""
        
        if destination == "research":
            return Command(
                goto="research_team",
                update={
                    "current_team": "research",
                    "messages": [f"Routing to research team for: {task_description}"],
                    "research_state": {
                        **research_state,
                        "research_status": "in_progress",
                        "assigned_timestamp": datetime.now().isoformat()
                    }
                }
            )
        
        elif destination == "reporting":
            return Command(
                goto="reporting_team",
                update={
                    "current_team": "reporting",
                    "messages": [f"Routing to reporting team for: {task_description}"],
                    "reporting_state": {
                        **reporting_state,
                        "report_status": "in_progress",
                        "assigned_timestamp": datetime.now().isoformat()
                    },
                    "research_data": {
                        "medical_findings": research_state.get("medical_findings", {}),
                        "financial_findings": research_state.get("financial_findings", {}),
                        "research_metadata": research_state.get("research_metadata", {})
                    }
                }
            )
        
        else:  # destination == "end"
            return Command(
                goto="end",
                update={
                    "current_team": "end",
                    "final_output": {
                        "document_path": reporting_state.get("document_path", ""),
                        "summary": reporting_state.get("summary", ""),
                        "completion_timestamp": datetime.now().isoformat(),
                        "system_metrics": self.get_metrics()
                    }
                }
            )
    
    def _analyze_with_llm(self, state: SupervisorState) -> Dict[str, Any]:
        """Use LLM to analyze complex routing decisions"""
        
        prompt = f"""
        Analyze the following task and determine the optimal routing strategy:
        
        Task: {state.get('task_description', '')}
        Research Status: {state.get('research_state', {}).get('research_status', 'pending')}
        Reporting Status: {state.get('reporting_state', {}).get('report_status', 'pending')}
        
        Provide analysis in the following format:
        - Next Step: [research/reporting/end]
        - Reasoning: [explanation]
        - Priority: [high/medium/low]
        """
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_llm_response(response.content)
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {"next_step": "research", "reasoning": "default fallback", "priority": "medium"}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for routing decisions"""
        
        lines = response.strip().split('\n')
        analysis = {}
        
        for line in lines:
            if line.startswith('- Next Step:'):
                analysis['next_step'] = line.split(':')[1].strip()
            elif line.startswith('- Reasoning:'):
                analysis['reasoning'] = line.split(':')[1].strip()
            elif line.startswith('- Priority:'):
                analysis['priority'] = line.split(':')[1].strip()
        
        return analysis
```

### ✅ Research Team Supervisor
```python
# src/agents/research/research_supervisor.py
from typing import Dict, Any, Literal, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.state.schemas import ResearchState
from src.utils.handoff import HandoffProtocol
from config.settings import Settings

class ResearchTeamSupervisor(BaseAgent):
    """Supervisor for the research team with LLM-based intelligent routing between medical and financial researchers"""
    
    def __init__(self, settings: Settings):
        super().__init__("research_team_supervisor")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.supervisor_model,
            temperature=0,
            timeout=settings.timeout_seconds
        )
        self.handoff_protocol = HandoffProtocol()
        
    def get_required_fields(self) -> List[str]:
        return ["research_topic", "research_status"]
        
    def process(self, state: ResearchState) -> Command[Literal["medical_researcher", "financial_researcher", "main_supervisor"]]:
        """Process research state and coordinate between researchers"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid research state"),
                    state
                )
            
            # Extract research information
            research_topic = state.get("research_topic", "")
            medical_findings = state.get("medical_findings", {})
            financial_findings = state.get("financial_findings", {})
            research_status = state.get("research_status", "pending")
            
            self.logger.info(f"Coordinating research for: {research_topic}")
            
            # Determine next action
            next_action = self._determine_next_researcher(
                medical_findings,
                financial_findings,
                research_topic
            )
            
            # Create command
            command = self._create_research_command(
                next_action,
                research_topic,
                medical_findings,
                financial_findings
            )
            
            # Record metrics
            self.record_metrics(start_time, success=True)
            
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _determine_next_researcher(
        self,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any],
        research_topic: str
    ) -> Literal["medical", "financial", "complete"]:
        """Determine which researcher should work next using LLM-based decision making"""
        
        medical_complete = bool(medical_findings.get("research_complete", False))
        financial_complete = bool(financial_findings.get("research_complete", False))
        
        # Both completed - return to main supervisor
        if medical_complete and financial_complete:
            return "complete"
        
        # Use LLM to determine optimal researcher assignment
        decision = self._analyze_research_assignment(
            research_topic,
            medical_findings,
            financial_findings,
            medical_complete,
            financial_complete
        )
        
        return decision.get("next_researcher", "medical")
    
    def _create_research_command(
        self,
        next_action: Literal["medical", "financial", "complete"],
        research_topic: str,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any]
    ) -> Command:
        """Create command for next research action"""
        
        if next_action == "medical":
            return Command(
                goto="medical_researcher",
                update={
                    "research_status": "in_progress",
                    "current_researcher": "medical",
                    "research_metadata": {
                        "assigned_timestamp": datetime.now().isoformat(),
                        "priority": "high" if "medical" in research_topic.lower() else "medium"
                    }
                }
            )
        
        elif next_action == "financial":
            return Command(
                goto="financial_researcher",
                update={
                    "research_status": "in_progress",
                    "current_researcher": "financial",
                    "research_metadata": {
                        "assigned_timestamp": datetime.now().isoformat(),
                        "priority": "high" if "financial" in research_topic.lower() else "medium"
                    }
                }
            )
        
        else:  # complete
            return Command(
                goto="main_supervisor",
                update={
                    "research_status": "completed",
                    "completion_timestamp": datetime.now().isoformat(),
                    "combined_findings": {
                        "medical": medical_findings,
                        "financial": financial_findings,
                        "research_summary": self._create_research_summary(
                            medical_findings,
                            financial_findings
                        )
                    }
                },
                graph=Command.PARENT  # Return to parent graph
            )
    
    def _analyze_research_assignment(
        self,
        research_topic: str,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any],
        medical_complete: bool,
        financial_complete: bool
    ) -> Dict[str, Any]:
        """Use LLM to analyze and determine optimal researcher assignment"""
        
        prompt = f"""
        You are a research coordination supervisor. Analyze the following research task and determine which specialized researcher should work next.

        Research Topic: {research_topic}
        
        Medical Research Status:
        - Complete: {medical_complete}
        - Findings: {len(medical_findings.get('key_findings', []))} key findings
        - Sources: {len(medical_findings.get('sources', []))} sources
        
        Financial Research Status:
        - Complete: {financial_complete}
        - Findings: {len(financial_findings.get('key_findings', []))} key findings
        - Sources: {len(financial_findings.get('sources', []))} sources
        
        Instructions:
        1. Analyze the research topic to determine domain relevance
        2. Consider current completion status of each researcher
        3. Evaluate interdependencies between medical and financial aspects
        4. Prioritize based on research efficiency and logical flow
        
        Respond with:
        - Next Researcher: [medical|financial]
        - Priority: [high|medium|low]
        - Reasoning: [detailed explanation]
        - Dependencies: [any dependencies on other research]
        
        Be concise and decisive in your analysis.
        """
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_research_assignment_response(response.content)
        except Exception as e:
            self.logger.error(f"LLM research assignment analysis failed: {e}")
            # Fallback to simple completion-based logic
            if not medical_complete:
                return {"next_researcher": "medical", "priority": "medium", "reasoning": "fallback to medical"}
            elif not financial_complete:
                return {"next_researcher": "financial", "priority": "medium", "reasoning": "fallback to financial"}
            else:
                return {"next_researcher": "medical", "priority": "low", "reasoning": "fallback default"}
    
    def _parse_research_assignment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for research assignment decisions"""
        
        lines = response.strip().split('\n')
        analysis = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('- Next Researcher:'):
                researcher = line.split(':')[1].strip().lower()
                analysis['next_researcher'] = researcher if researcher in ['medical', 'financial'] else 'medical'
            elif line.startswith('- Priority:'):
                analysis['priority'] = line.split(':')[1].strip().lower()
            elif line.startswith('- Reasoning:'):
                analysis['reasoning'] = line.split(':')[1].strip()
            elif line.startswith('- Dependencies:'):
                analysis['dependencies'] = line.split(':')[1].strip()
        
        # Ensure we have a valid next_researcher
        if 'next_researcher' not in analysis:
            analysis['next_researcher'] = 'medical'
        
        return analysis
    
    def _create_research_summary(
        self,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive research summary"""
        
        return {
            "medical_key_points": medical_findings.get("key_findings", []),
            "financial_key_points": financial_findings.get("key_findings", []),
            "cross_domain_insights": self._identify_cross_domain_insights(
                medical_findings,
                financial_findings
            ),
            "research_quality_score": self._calculate_quality_score(
                medical_findings,
                financial_findings
            ),
            "completion_metrics": {
                "medical_sources": len(medical_findings.get("sources", [])),
                "financial_sources": len(financial_findings.get("sources", [])),
                "total_processing_time": self.get_metrics().get("total_processing_time", 0)
            }
        }
    
    def _identify_cross_domain_insights(
        self,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any]
    ) -> List[str]:
        """Identify insights that span both medical and financial domains"""
        
        insights = []
        
        # Extract key terms from both domains
        medical_terms = set(medical_findings.get("key_terms", []))
        financial_terms = set(financial_findings.get("key_terms", []))
        
        # Find overlapping concepts
        overlap = medical_terms.intersection(financial_terms)
        
        if overlap:
            insights.append(f"Common themes identified: {', '.join(overlap)}")
        
        # Look for investment implications in medical findings
        if medical_findings.get("market_impact"):
            insights.append("Medical findings have identified market implications")
        
        # Look for health implications in financial findings
        if financial_findings.get("health_sector_analysis"):
            insights.append("Financial analysis includes health sector considerations")
        
        return insights
    
    def _calculate_quality_score(
        self,
        medical_findings: Dict[str, Any],
        financial_findings: Dict[str, Any]
    ) -> float:
        """Calculate overall research quality score"""
        
        medical_score = medical_findings.get("quality_score", 0)
        financial_score = financial_findings.get("quality_score", 0)
        
        # Weight scores based on completeness
        medical_weight = 1.0 if medical_findings.get("research_complete") else 0.5
        financial_weight = 1.0 if financial_findings.get("research_complete") else 0.5
        
        total_weight = medical_weight + financial_weight
        
        if total_weight == 0:
            return 0.0
        
        return (medical_score * medical_weight + financial_score * financial_weight) / total_weight
```

### ✅ Reporting Team Supervisor
```python
# src/agents/reporting/reporting_supervisor.py
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
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid reporting state"),
                    state
                )
            
            # Extract reporting information
            research_data = state.get("research_data", {})
            document_path = state.get("document_path", "")
            summary = state.get("summary", "")
            report_status = state.get("report_status", "pending")
            
            self.logger.info(f"Coordinating reporting for research data")
            
            # Determine next action
            next_action = self._determine_next_reporter(
                document_path,
                summary,
                research_data
            )
            
            # Create command
            command = self._create_reporting_command(
                next_action,
                research_data,
                document_path,
                summary
            )
            
            # Record metrics
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
        
        # Both completed - return to main supervisor
        if document_complete and summary_complete:
            return "complete"
        
        # Prefer document creation first, then summary
        if not document_complete:
            return "document"
        
        if not summary_complete:
            return "summary"
        
        # Fallback to document creation
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
        
        else:  # complete
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
                graph=Command.PARENT  # Return to parent graph
            )
    
    def _assess_data_complexity(self, research_data: Dict[str, Any]) -> str:
        """Assess complexity of research data for document creation"""
        
        medical_findings = research_data.get("medical_findings", {})
        financial_findings = research_data.get("financial_findings", {})
        
        # Count data points
        medical_points = len(medical_findings.get("key_findings", []))
        financial_points = len(financial_findings.get("key_findings", []))
        total_points = medical_points + financial_points
        
        # Assess complexity
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
        
        # Calculate coverage based on available data
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
        
        # Document completion (50% weight)
        if document_path:
            score += 0.5
        
        # Summary completion (30% weight)
        if summary and len(summary) > 100:
            score += 0.3
        
        # Data utilization (20% weight)
        data_coverage = self._calculate_data_coverage(research_data)
        score += 0.2 * data_coverage
        
        return score
```

### ✅ Error Handling and Retry Logic
```python
# src/utils/error_handling.py
from typing import Dict, Any, Optional, Callable
from langgraph.types import Command
from datetime import datetime, timedelta
import asyncio
import logging
from functools import wraps

class ErrorHandler:
    """Centralized error handling for the multi-agent system"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger("error_handler")
        
    def with_retry(self, func: Callable) -> Callable:
        """Decorator for adding retry logic to functions"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        self.logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
            
            raise last_exception
        
        return wrapper
    
    def create_error_command(
        self,
        error: Exception,
        source_agent: str,
        state: Dict[str, Any],
        recovery_action: str = "retry"
    ) -> Command:
        """Create error command for failed operations"""
        
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "source_agent": source_agent,
            "timestamp": datetime.now().isoformat(),
            "recovery_action": recovery_action,
            "state_snapshot": state
        }
        
        return Command(
            goto="error_handler",
            update={"error_state": error_data}
        )
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if error should trigger retry"""
        
        # Network errors - retry
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        # API rate limits - retry
        if "rate limit" in str(error).lower():
            return True
        
        # Temporary API errors - retry
        if "500" in str(error) or "502" in str(error) or "503" in str(error):
            return True
        
        # Configuration errors - don't retry
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        # Default to retry
        return True
```

## 🧪 Testing Implementation

### ✅ Unit Tests for Supervisors
```python
# tests/unit/test_supervisors.py
import pytest
from unittest.mock import Mock, patch
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
    def supervisor(self, settings):
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
    def supervisor(self, settings):
        return ResearchTeamSupervisor(settings)
    
    @patch('src.agents.research.research_supervisor.ChatOpenAI')
    def test_route_to_medical_researcher(self, mock_openai, supervisor):
        # Mock LLM response for medical researcher assignment
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
        # Mock LLM response for financial researcher assignment
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
    def supervisor(self, settings):
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
```

## 🎯 Success Criteria

### Functional Requirements:
- [ ] Main Supervisor routes correctly based on state
- [ ] Research Team Supervisor uses LLM-based intelligent routing for researcher assignment
- [ ] Reporting Team Supervisor coordinates document creation and summarization
- [ ] LLM decision-making provides accurate and contextual routing decisions
- [ ] Command objects are properly structured
- [ ] Error handling works for all failure scenarios including LLM failures
- [ ] Retry mechanisms function with exponential backoff
- [ ] Fallback logic handles LLM unavailability gracefully

### Quality Requirements:
- [ ] Unit tests achieve >90% coverage
- [ ] All supervisors handle edge cases gracefully
- [ ] Error messages are informative and actionable
- [ ] State transitions are atomic and consistent
- [ ] Performance metrics are properly recorded

### Performance Requirements:
- [ ] Supervisor decisions complete in <2 seconds
- [ ] State validation executes in <100ms
- [ ] Error handling adds minimal overhead
- [ ] Memory usage remains stable

## 📊 Stage 2 Metrics

### Time Allocation:
- Main Supervisor implementation: 90 minutes
- Research Team Supervisor: 75 minutes
- Reporting Team Supervisor: 75 minutes
- Error handling and retry logic: 60 minutes
- Unit tests: 90 minutes
- Integration testing: 30 minutes

### Success Indicators:
- All supervisor classes instantiate correctly
- Routing logic passes all test cases
- Error handling prevents system crashes
- State transitions maintain consistency
- Performance metrics are within targets

## 🔄 Next Steps

After completing Stage 2, proceed to:
1. **Stage 3**: Specialized Agent Implementation
2. Implement medical and financial researchers
3. Create document creator and summarizer
4. Add API integrations and tools

---

*This stage establishes the core coordination logic that orchestrates the sophisticated multi-agent workflow, ensuring reliable and efficient team management.*
from typing import Dict, Any, Literal, List
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
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid input state"),
                    state
                )
            
            task_description = state.get("task_description", "")
            research_state = state.get("research_state", {})
            reporting_state = state.get("reporting_state", {})
            
            self.logger.info(f"Processing task: {task_description}")
            
            routing_decision = self._determine_routing(research_state, reporting_state)
            
            command = self._create_routing_command(
                routing_decision,
                task_description,
                research_state,
                reporting_state
            )
            
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
        
        if research_status != "completed":
            return "research"
        elif reporting_status != "completed":
            return "reporting"
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
        
        else:
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
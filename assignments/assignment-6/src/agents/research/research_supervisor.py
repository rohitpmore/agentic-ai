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
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid research state"),
                    state
                )
            
            research_topic = state.get("research_topic", "")
            medical_findings = state.get("medical_findings", {})
            financial_findings = state.get("financial_findings", {})
            
            self.logger.info(f"Coordinating research for: {research_topic}")
            
            next_action = self._determine_next_researcher(
                medical_findings,
                financial_findings,
                research_topic
            )
            
            command = self._create_research_command(
                next_action,
                research_topic,
                medical_findings,
                financial_findings
            )
            
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
        
        if medical_complete and financial_complete:
            return "complete"
        
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
        
        else:
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
                graph=Command.PARENT
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
        
        medical_terms = set(medical_findings.get("key_terms", []))
        financial_terms = set(financial_findings.get("key_terms", []))
        
        overlap = medical_terms.intersection(financial_terms)
        
        if overlap:
            insights.append(f"Common themes identified: {', '.join(overlap)}")
        
        if medical_findings.get("market_impact"):
            insights.append("Medical findings have identified market implications")
        
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
        
        medical_weight = 1.0 if medical_findings.get("research_complete") else 0.5
        financial_weight = 1.0 if financial_findings.get("research_complete") else 0.5
        
        total_weight = medical_weight + financial_weight
        
        if total_weight == 0:
            return 0.0
        
        return (medical_score * medical_weight + financial_score * financial_weight) / total_weight
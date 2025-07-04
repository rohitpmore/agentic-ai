#!/usr/bin/env python3
"""
LangGraph Multi-Agent Hierarchical Workflow System - Main Implementation

Stage 4: LangGraph Integration & Workflow Assembly
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ValidationNode
from datetime import datetime
import logging
import os
import uuid

from src.agents.supervisor import MainSupervisor
from src.agents.research.research_supervisor import ResearchTeamSupervisor
from src.agents.research.medical_researcher import MedicalResearcher
from src.agents.research.financial_researcher import FinancialResearcher
from src.agents.reporting.reporting_supervisor import ReportingTeamSupervisor
from src.agents.reporting.document_creator import DocumentCreator
from src.agents.reporting.summarizer import Summarizer
from src.state.schemas import SupervisorState, ResearchState, ReportingState
from config.settings import Settings


class MultiAgentWorkflow:
    """Main workflow orchestrator for the multi-agent system"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("workflow")
        
        # Initialize agents
        self.main_supervisor = MainSupervisor(settings)
        
        # Build complete graph
        self.graph = self._build_main_graph()
        
    def _build_main_graph(self) -> StateGraph:
        """Build the main hierarchical graph"""
        
        # Create main graph builder
        builder = StateGraph(SupervisorState)
        
        # Add main supervisor
        builder.add_node("main_supervisor", self.main_supervisor)
        
        # Add team subgraphs
        builder.add_node("research_team", self._build_research_team_graph())
        builder.add_node("reporting_team", self._build_reporting_team_graph())
        
        # Add validation and error handling nodes
        builder.add_node("input_validator", self._create_input_validator())
        builder.add_node("error_handler", self._create_error_handler())
        
        # Define entry point
        builder.add_edge(START, "input_validator")
        
        # Main workflow edges
        builder.add_edge("input_validator", "main_supervisor")
        builder.add_edge("research_team", "main_supervisor")
        builder.add_edge("reporting_team", "main_supervisor")
        
        # Error handling edges
        builder.add_edge("error_handler", "main_supervisor")
        
        # Conditional edges from main supervisor
        builder.add_conditional_edges(
            "main_supervisor",
            self._route_from_main_supervisor,
            {
                "research_team": "research_team",
                "reporting_team": "reporting_team",
                "end": END,
                "error": "error_handler"
            }
        )
        
        # Add memory for persistence
        memory = MemorySaver()
        
        # Compile graph with configuration
        return builder.compile(
            checkpointer=memory,
            debug=self.settings.debug_mode
        )
    
    def _build_research_team_graph(self) -> StateGraph:
        """Build research team subgraph"""
        
        # Create research team builder
        builder = StateGraph(ResearchState)
        
        # Initialize research team agents
        research_supervisor = ResearchTeamSupervisor(self.settings)
        medical_researcher = MedicalResearcher(self.settings)
        financial_researcher = FinancialResearcher(self.settings)
        
        # Add nodes
        builder.add_node("research_supervisor", research_supervisor)
        builder.add_node("medical_researcher", medical_researcher)
        builder.add_node("financial_researcher", financial_researcher)
        
        # Add research validation
        builder.add_node("research_validator", self._create_research_validator())
        
        # Define entry point
        builder.add_edge(START, "research_supervisor")
        
        # Research workflow edges
        builder.add_edge("medical_researcher", "research_validator")
        builder.add_edge("financial_researcher", "research_validator")
        builder.add_edge("research_validator", "research_supervisor")
        
        # Conditional edges from research supervisor
        builder.add_conditional_edges(
            "research_supervisor",
            self._route_from_research_supervisor,
            {
                "medical_researcher": "medical_researcher",
                "financial_researcher": "financial_researcher",
                "main_supervisor": END  # Return to parent graph
            }
        )
        
        return builder.compile(debug=self.settings.debug_mode)
    
    def _build_reporting_team_graph(self) -> StateGraph:
        """Build reporting team subgraph"""
        
        # Create reporting team builder
        builder = StateGraph(ReportingState)
        
        # Initialize reporting team agents
        reporting_supervisor = ReportingTeamSupervisor(self.settings)
        document_creator = DocumentCreator(self.settings)
        summarizer = Summarizer(self.settings)
        
        # Add nodes
        builder.add_node("reporting_supervisor", reporting_supervisor)
        builder.add_node("document_creator", document_creator)
        builder.add_node("summarizer", summarizer)
        
        # Add reporting validation
        builder.add_node("reporting_validator", self._create_reporting_validator())
        
        # Define entry point
        builder.add_edge(START, "reporting_supervisor")
        
        # Reporting workflow edges
        builder.add_edge("document_creator", "reporting_validator")
        builder.add_edge("summarizer", "reporting_validator")
        builder.add_edge("reporting_validator", "reporting_supervisor")
        
        # Conditional edges from reporting supervisor
        builder.add_conditional_edges(
            "reporting_supervisor",
            self._route_from_reporting_supervisor,
            {
                "document_creator": "document_creator",
                "summarizer": "summarizer",
                "main_supervisor": END  # Return to parent graph
            }
        )
        
        return builder.compile(debug=self.settings.debug_mode)
    
    def _route_from_main_supervisor(self, state: SupervisorState) -> str:
        """Route from main supervisor based on state"""
        
        try:
            # Check for errors
            if state.get("error_state"):
                return "error"
            
            current_team = state.get("current_team", "research")
            
            # Route based on team assignment
            if current_team == "research":
                return "research_team"
            elif current_team == "reporting":
                return "reporting_team"
            elif current_team == "end":
                return "end"
            else:
                # Default to research team
                return "research_team"
                
        except Exception as e:
            self.logger.error(f"Routing error from main supervisor: {e}")
            return "error"
    
    def _route_from_research_supervisor(self, state: ResearchState) -> str:
        """Route from research supervisor based on state"""
        
        try:
            current_researcher = state.get("current_researcher", "")
            research_status = state.get("research_status", "pending")
            
            # If research is completed, return to main supervisor
            if research_status == "completed":
                return "main_supervisor"
            
            # Route to specific researcher
            if current_researcher == "medical":
                return "medical_researcher"
            elif current_researcher == "financial":
                return "financial_researcher"
            else:
                # Default to medical researcher
                return "medical_researcher"
                
        except Exception as e:
            self.logger.error(f"Routing error from research supervisor: {e}")
            return "main_supervisor"
    
    def _route_from_reporting_supervisor(self, state: ReportingState) -> str:
        """Route from reporting supervisor based on state"""
        
        try:
            current_reporter = state.get("current_reporter", "")
            report_status = state.get("report_status", "pending")
            
            # If reporting is completed, return to main supervisor
            if report_status == "completed":
                return "main_supervisor"
            
            # Route to specific reporter
            if current_reporter == "document_creator":
                return "document_creator"
            elif current_reporter == "summarizer":
                return "summarizer"
            else:
                # Default to document creator
                return "document_creator"
                
        except Exception as e:
            self.logger.error(f"Routing error from reporting supervisor: {e}")
            return "main_supervisor"
    
    def _create_input_validator(self) -> ValidationNode:
        """Create input validation node"""
        
        def validate_input(state: SupervisorState) -> SupervisorState:
            """Validate initial input state"""
            
            # Check required fields
            required_fields = ["task_description"]
            missing_fields = [field for field in required_fields if not state.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Initialize state if needed
            if "research_state" not in state:
                state["research_state"] = {
                    "research_topic": state["task_description"],
                    "research_status": "pending",
                    "medical_findings": {},
                    "financial_findings": {},
                    "research_metadata": {}
                }
            
            if "reporting_state" not in state:
                state["reporting_state"] = {
                    "report_status": "pending",
                    "research_data": {},
                    "document_path": "",
                    "summary": "",
                    "report_metadata": {}
                }
            
            # Initialize current team
            if "current_team" not in state:
                state["current_team"] = "research"
            
            self.logger.info(f"Input validated for task: {state['task_description']}")
            
            return state
        
        return ValidationNode(validate_input)
    
    def _create_research_validator(self) -> ValidationNode:
        """Create research validation node"""
        
        def validate_research(state: ResearchState) -> ResearchState:
            """Validate research findings"""
            
            medical_findings = state.get("medical_findings", {})
            financial_findings = state.get("financial_findings", {})
            
            # Validate medical findings
            if medical_findings:
                if not medical_findings.get("key_findings"):
                    self.logger.warning("Medical findings missing key findings")
                    
                if not medical_findings.get("research_papers"):
                    self.logger.warning("Medical findings missing research papers")
            
            # Validate financial findings
            if financial_findings:
                if not financial_findings.get("key_findings"):
                    self.logger.warning("Financial findings missing key findings")
                    
                if not financial_findings.get("research_papers"):
                    self.logger.warning("Financial findings missing research papers")
            
            # Update research metadata
            state["research_metadata"] = {
                **state.get("research_metadata", {}),
                "validation_timestamp": datetime.now().isoformat(),
                "medical_quality": medical_findings.get("quality_score", 0),
                "financial_quality": financial_findings.get("quality_score", 0)
            }
            
            self.logger.info("Research findings validated")
            
            return state
        
        return ValidationNode(validate_research)
    
    def _create_reporting_validator(self) -> ValidationNode:
        """Create reporting validation node"""
        
        def validate_reporting(state: ReportingState) -> ReportingState:
            """Validate reporting outputs"""
            
            document_path = state.get("document_path", "")
            summary = state.get("summary", "")
            
            # Validate document
            if document_path:
                if not os.path.exists(document_path):
                    self.logger.error(f"Document not found: {document_path}")
                    state["document_path"] = ""
                else:
                    # Check file size
                    file_size = os.path.getsize(document_path)
                    if file_size < 1000:  # Less than 1KB
                        self.logger.warning(f"Document suspiciously small: {file_size} bytes")
            
            # Validate summary
            if summary:
                if len(summary) < 100:
                    self.logger.warning(f"Summary suspiciously short: {len(summary)} characters")
            
            # Update reporting metadata
            state["report_metadata"] = {
                **state.get("report_metadata", {}),
                "validation_timestamp": datetime.now().isoformat(),
                "document_valid": bool(document_path and os.path.exists(document_path)),
                "summary_valid": bool(summary and len(summary) >= 100)
            }
            
            self.logger.info("Reporting outputs validated")
            
            return state
        
        return ValidationNode(validate_reporting)
    
    def _create_error_handler(self) -> ValidationNode:
        """Create error handling node"""
        
        def handle_error(state: SupervisorState) -> SupervisorState:
            """Handle system errors"""
            
            error_state = state.get("error_state", {})
            
            if error_state:
                error_type = error_state.get("error_type", "Unknown")
                error_message = error_state.get("error_message", "No message")
                source_agent = error_state.get("source_agent", "Unknown")
                
                self.logger.error(f"Error from {source_agent}: {error_type} - {error_message}")
                
                # Attempt recovery based on error type
                if "API" in error_type or "rate limit" in error_message.lower():
                    # API error - add delay and retry
                    state["retry_delay"] = 5
                    state["current_team"] = "research"  # Restart from research
                elif "validation" in error_type.lower():
                    # Validation error - reset to previous state
                    state["current_team"] = "research"
                else:
                    # Unknown error - end workflow
                    state["current_team"] = "end"
                    state["final_output"] = {
                        "error": "Workflow terminated due to unrecoverable error",
                        "error_details": error_state
                    }
                
                # Clear error state after handling
                state["error_state"] = None
            
            return state
        
        return ValidationNode(handle_error)
    
    async def run_workflow(
        self,
        task_description: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the complete workflow"""
        
        # Prepare initial state
        initial_state = {
            "task_description": task_description,
            "messages": [f"Starting multi-agent research workflow: {task_description}"],
            "system_metrics": {
                "start_time": datetime.now().isoformat(),
                "workflow_id": str(uuid.uuid4())
            }
        }
        
        # Default configuration
        default_config = {
            "recursion_limit": self.settings.recursion_limit,
            "callbacks": []
        }
        
        if config:
            default_config.update(config)
        
        try:
            # Run workflow
            self.logger.info(f"Starting workflow: {task_description}")
            
            result = await self.graph.ainvoke(
                initial_state,
                config=default_config
            )
            
            # Add completion metrics
            result["system_metrics"]["end_time"] = datetime.now().isoformat()
            result["system_metrics"]["success"] = True
            
            self.logger.info("Workflow completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            
            return {
                "error": str(e),
                "system_metrics": {
                    "start_time": initial_state["system_metrics"]["start_time"],
                    "end_time": datetime.now().isoformat(),
                    "success": False
                }
            }
    
    async def run_workflow_streaming(
        self,
        task_description: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Run workflow with streaming updates"""
        
        # Prepare initial state
        initial_state = {
            "task_description": task_description,
            "messages": [f"Starting streaming workflow: {task_description}"],
            "system_metrics": {
                "start_time": datetime.now().isoformat(),
                "workflow_id": str(uuid.uuid4())
            }
        }
        
        # Default configuration
        default_config = {
            "recursion_limit": self.settings.recursion_limit,
            "callbacks": []
        }
        
        if config:
            default_config.update(config)
        
        try:
            self.logger.info(f"Starting streaming workflow: {task_description}")
            
            async for chunk in self.graph.astream(
                initial_state,
                config=default_config
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Streaming workflow failed: {e}")
            yield {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_graph_visualization(self) -> str:
        """Get Mermaid diagram of the workflow"""
        
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            self.logger.error(f"Graph visualization failed: {e}")
            return "Graph visualization not available"
    
    def save_graph_image(self, filename: str = "workflow_graph.png") -> str:
        """Save graph visualization as image"""
        
        try:
            output_path = os.path.join(self.settings.output_directory, filename)
            self.graph.get_graph().draw_mermaid_png(output_file_path=output_path)
            return output_path
        except Exception as e:
            self.logger.error(f"Graph image save failed: {e}")
            return ""


def main():
    """Main entry point for the multi-agent system"""
    from src.utils.logging_config import setup_logging
    
    # Setup logging
    setup_logging(level="INFO", format_type="standard")
    
    print("🚀 LangGraph Multi-Agent Hierarchical Workflow System")
    print("📋 Stage 4: LangGraph Integration & Workflow Assembly - READY")
    print("")
    print("✅ Components Implemented:")
    print("  - MultiAgentWorkflow orchestrator")
    print("  - Hierarchical team subgraphs")
    print("  - State transition logic")
    print("  - Validation and error handling nodes")
    print("  - Graph visualization capabilities")
    print("")
    print("🔄 Next: Run integration tests and proceed to Stage 5")


if __name__ == "__main__":
    main()
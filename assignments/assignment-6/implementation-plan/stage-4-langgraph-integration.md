# Stage 4: LangGraph Integration & Workflow Assembly

**Timeline:** 3-4 hours  
**Status:** ⏳ Pending  
**Priority:** High

## 📋 Overview

This stage focuses on assembling all components into a cohesive LangGraph StateGraph workflow. We'll build hierarchical team subgraphs, implement state transition logic, enable parallel execution, and create comprehensive integration tests.

## 🎯 Key Deliverables

### ✅ Complete LangGraph StateGraph Implementation
### ✅ Hierarchical Team Subgraphs
### ✅ State Transition Logic
### ✅ Parallel Execution Capabilities
### ✅ Integration Tests for Full Workflow
### ✅ Graph Visualization and Debugging

## 🔧 Implementation Details

### ✅ Main Graph Assembly
```python
# src/main.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ValidationNode
import logging

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
```

### ✅ State Reducers and Parallel Execution
```python
# src/utils/state_management.py
from typing import Dict, Any, List, Union, Callable
from langgraph.graph import add_messages
from datetime import datetime
import logging

logger = logging.getLogger("state_management")

def research_state_reducer(
    current: Dict[str, Any],
    update: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer for research state updates"""
    
    # Handle messages
    if "messages" in update:
        current_messages = current.get("messages", [])
        new_messages = update["messages"]
        current["messages"] = add_messages(current_messages, new_messages)
        del update["messages"]
    
    # Handle medical findings accumulation
    if "medical_findings" in update:
        current_medical = current.get("medical_findings", {})
        new_medical = update["medical_findings"]
        
        # Merge findings intelligently
        merged_medical = {**current_medical, **new_medical}
        
        # Accumulate key findings
        if "key_findings" in current_medical and "key_findings" in new_medical:
            merged_medical["key_findings"] = list(set(
                current_medical["key_findings"] + new_medical["key_findings"]
            ))
        
        current["medical_findings"] = merged_medical
        del update["medical_findings"]
    
    # Handle financial findings accumulation
    if "financial_findings" in update:
        current_financial = current.get("financial_findings", {})
        new_financial = update["financial_findings"]
        
        # Merge findings intelligently
        merged_financial = {**current_financial, **new_financial}
        
        # Accumulate key findings
        if "key_findings" in current_financial and "key_findings" in new_financial:
            merged_financial["key_findings"] = list(set(
                current_financial["key_findings"] + new_financial["key_findings"]
            ))
        
        current["financial_findings"] = merged_financial
        del update["financial_findings"]
    
    # Handle research metadata
    if "research_metadata" in update:
        current_metadata = current.get("research_metadata", {})
        new_metadata = update["research_metadata"]
        current["research_metadata"] = {**current_metadata, **new_metadata}
        del update["research_metadata"]
    
    # Apply remaining updates
    current.update(update)
    
    return current

def reporting_state_reducer(
    current: Dict[str, Any],
    update: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer for reporting state updates"""
    
    # Handle messages
    if "messages" in update:
        current_messages = current.get("messages", [])
        new_messages = update["messages"]
        current["messages"] = add_messages(current_messages, new_messages)
        del update["messages"]
    
    # Handle research data updates
    if "research_data" in update:
        current_data = current.get("research_data", {})
        new_data = update["research_data"]
        current["research_data"] = {**current_data, **new_data}
        del update["research_data"]
    
    # Handle report metadata
    if "report_metadata" in update:
        current_metadata = current.get("report_metadata", {})
        new_metadata = update["report_metadata"]
        current["report_metadata"] = {**current_metadata, **new_metadata}
        del update["report_metadata"]
    
    # Apply remaining updates
    current.update(update)
    
    return current

def supervisor_state_reducer(
    current: Dict[str, Any],
    update: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer for supervisor state updates"""
    
    # Handle messages
    if "messages" in update:
        current_messages = current.get("messages", [])
        new_messages = update["messages"]
        current["messages"] = add_messages(current_messages, new_messages)
        del update["messages"]
    
    # Handle nested state updates
    if "research_state" in update:
        current_research = current.get("research_state", {})
        new_research = update["research_state"]
        current["research_state"] = research_state_reducer(current_research, new_research)
        del update["research_state"]
    
    if "reporting_state" in update:
        current_reporting = current.get("reporting_state", {})
        new_reporting = update["reporting_state"]
        current["reporting_state"] = reporting_state_reducer(current_reporting, new_reporting)
        del update["reporting_state"]
    
    # Handle system metrics
    if "system_metrics" in update:
        current_metrics = current.get("system_metrics", {})
        new_metrics = update["system_metrics"]
        current["system_metrics"] = {**current_metrics, **new_metrics}
        del update["system_metrics"]
    
    # Apply remaining updates
    current.update(update)
    
    return current

class ParallelExecutionManager:
    """Manager for parallel execution within teams"""
    
    def __init__(self):
        self.logger = logging.getLogger("parallel_execution")
    
    async def execute_research_parallel(
        self,
        medical_researcher: Callable,
        financial_researcher: Callable,
        research_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute medical and financial research in parallel"""
        
        import asyncio
        
        # Prepare separate states for each researcher
        medical_state = {
            **research_state,
            "researcher_type": "medical"
        }
        
        financial_state = {
            **research_state,
            "researcher_type": "financial"
        }
        
        try:
            # Execute in parallel
            medical_task = asyncio.create_task(
                medical_researcher.aprocess(medical_state)
            )
            
            financial_task = asyncio.create_task(
                financial_researcher.aprocess(financial_state)
            )
            
            # Wait for both to complete
            medical_result, financial_result = await asyncio.gather(
                medical_task,
                financial_task,
                return_exceptions=True
            )
            
            # Handle results
            combined_state = research_state.copy()
            
            if isinstance(medical_result, Exception):
                self.logger.error(f"Medical research failed: {medical_result}")
                combined_state["medical_findings"] = {"error": str(medical_result)}
            else:
                combined_state.update(medical_result.update)
            
            if isinstance(financial_result, Exception):
                self.logger.error(f"Financial research failed: {financial_result}")
                combined_state["financial_findings"] = {"error": str(financial_result)}
            else:
                combined_state.update(financial_result.update)
            
            return combined_state
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return research_state
    
    async def execute_reporting_parallel(
        self,
        document_creator: Callable,
        summarizer: Callable,
        reporting_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute document creation and summarization in parallel"""
        
        import asyncio
        
        # Prepare separate states
        doc_state = {
            **reporting_state,
            "task_type": "document_creation"
        }
        
        summary_state = {
            **reporting_state,
            "task_type": "summarization"
        }
        
        try:
            # Execute in parallel
            doc_task = asyncio.create_task(
                document_creator.aprocess(doc_state)
            )
            
            summary_task = asyncio.create_task(
                summarizer.aprocess(summary_state)
            )
            
            # Wait for both to complete
            doc_result, summary_result = await asyncio.gather(
                doc_task,
                summary_task,
                return_exceptions=True
            )
            
            # Handle results
            combined_state = reporting_state.copy()
            
            if isinstance(doc_result, Exception):
                self.logger.error(f"Document creation failed: {doc_result}")
                combined_state["document_path"] = ""
            else:
                combined_state.update(doc_result.update)
            
            if isinstance(summary_result, Exception):
                self.logger.error(f"Summarization failed: {summary_result}")
                combined_state["summary"] = ""
            else:
                combined_state.update(summary_result.update)
            
            return combined_state
            
        except Exception as e:
            self.logger.error(f"Parallel reporting execution failed: {e}")
            return reporting_state
```

### ✅ Integration Testing Framework
```python
# tests/integration/test_workflow_integration.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.main import MultiAgentWorkflow
from config.settings import Settings

class TestWorkflowIntegration:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4",
            researcher_model="gpt-4",
            reporter_model="gpt-3.5-turbo",
            max_retries=3,
            output_directory="./test_outputs"
        )
    
    @pytest.fixture
    def workflow(self, settings):
        return MultiAgentWorkflow(settings)
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, workflow):
        """Test complete workflow from start to finish"""
        
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(return_value=[
                {
                    "title": "Test Paper",
                    "authors": ["Test Author"],
                    "abstract": "Test abstract",
                    "url": "http://test.com",
                    "relevance_score": 0.8
                }
            ])
        ), patch.multiple(
            'langchain_openai.ChatOpenAI',
            invoke=Mock(return_value=Mock(content="Test LLM response"))
        ), patch.multiple(
            'src.tools.document_tools.DocumentTool',
            create_document=Mock(return_value="/test/document.pdf")
        ):
            
            task_description = "Research AI applications in healthcare and finance"
            
            result = await workflow.run_workflow(task_description)
            
            # Verify workflow completion
            assert "final_output" in result
            assert result["system_metrics"]["success"] is True
            
            # Verify research completion
            research_state = result.get("research_state", {})
            assert research_state.get("research_status") == "completed"
            assert "medical_findings" in research_state
            assert "financial_findings" in research_state
            
            # Verify reporting completion
            reporting_state = result.get("reporting_state", {})
            assert reporting_state.get("report_status") == "completed"
            assert "document_path" in reporting_state
            assert "summary" in reporting_state
    
    @pytest.mark.asyncio
    async def test_workflow_streaming(self, workflow):
        """Test streaming workflow execution"""
        
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(return_value=[])
        ), patch.multiple(
            'langchain_openai.ChatOpenAI',
            invoke=Mock(return_value=Mock(content="Streaming test"))
        ):
            
            task_description = "Test streaming workflow"
            chunks = []
            
            async for chunk in workflow.run_workflow_streaming(task_description):
                chunks.append(chunk)
                if len(chunks) >= 5:  # Limit for test
                    break
            
            assert len(chunks) > 0
            assert any("messages" in chunk for chunk in chunks if isinstance(chunk, dict))
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, workflow):
        """Test error handling throughout the workflow"""
        
        # Mock API failure
        with patch.multiple(
            'src.tools.arxiv_tool.ArxivTool',
            search_papers=Mock(side_effect=Exception("API Error"))
        ):
            
            task_description = "Test error handling"
            
            result = await workflow.run_workflow(task_description)
            
            # Should handle error gracefully
            assert "error" in result or result.get("system_metrics", {}).get("success") is False
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, workflow):
        """Test state transitions between teams"""
        
        with patch.multiple(
            'src.agents.supervisor.MainSupervisor.process',
            return_value=Mock(
                goto="research_team",
                update={"current_team": "research"}
            )
        ):
            
            # Test initial routing to research team
            initial_state = {
                "task_description": "Test state transitions",
                "current_team": "research"
            }
            
            # Verify routing logic works
            routing_result = workflow._route_from_main_supervisor(initial_state)
            assert routing_result == "research_team"
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, workflow):
        """Test parallel execution within teams"""
        
        from src.utils.state_management import ParallelExecutionManager
        
        parallel_manager = ParallelExecutionManager()
        
        # Mock researchers
        medical_researcher = AsyncMock(return_value=Mock(
            update={"medical_findings": {"key_findings": ["Medical finding"]}}
        ))
        
        financial_researcher = AsyncMock(return_value=Mock(
            update={"financial_findings": {"key_findings": ["Financial finding"]}}
        ))
        
        research_state = {
            "research_topic": "Test parallel execution",
            "research_status": "in_progress"
        }
        
        result = await parallel_manager.execute_research_parallel(
            medical_researcher,
            financial_researcher,
            research_state
        )
        
        # Verify both researchers were called
        medical_researcher.assert_called_once()
        financial_researcher.assert_called_once()
        
        # Verify results are combined
        assert "medical_findings" in result
        assert "financial_findings" in result
    
    def test_graph_visualization(self, workflow):
        """Test graph visualization functionality"""
        
        # Test Mermaid diagram generation
        mermaid_diagram = workflow.get_graph_visualization()
        assert isinstance(mermaid_diagram, str)
        assert len(mermaid_diagram) > 0
        
        # Test graph image save (may fail in test environment)
        try:
            image_path = workflow.save_graph_image("test_graph.png")
            assert isinstance(image_path, str)
        except Exception:
            # Graph image save may fail in test environment
            pass

class TestStateReducers:
    def test_research_state_reducer(self):
        """Test research state reduction logic"""
        
        from src.utils.state_management import research_state_reducer
        
        current_state = {
            "medical_findings": {"key_findings": ["Finding 1"]},
            "financial_findings": {"key_findings": ["Finding 2"]},
            "messages": ["Message 1"]
        }
        
        update = {
            "medical_findings": {"key_findings": ["Finding 3"]},
            "messages": ["Message 2"]
        }
        
        result = research_state_reducer(current_state, update)
        
        # Verify findings are accumulated
        assert len(result["medical_findings"]["key_findings"]) == 2
        assert "Finding 1" in result["medical_findings"]["key_findings"]
        assert "Finding 3" in result["medical_findings"]["key_findings"]
        
        # Verify messages are added
        assert len(result["messages"]) == 2
    
    def test_supervisor_state_reducer(self):
        """Test supervisor state reduction logic"""
        
        from src.utils.state_management import supervisor_state_reducer
        
        current_state = {
            "research_state": {"research_status": "pending"},
            "reporting_state": {"report_status": "pending"},
            "system_metrics": {"start_time": "2023-01-01T00:00:00"}
        }
        
        update = {
            "research_state": {"research_status": "completed"},
            "system_metrics": {"end_time": "2023-01-01T01:00:00"}
        }
        
        result = supervisor_state_reducer(current_state, update)
        
        # Verify nested state updates
        assert result["research_state"]["research_status"] == "completed"
        assert result["reporting_state"]["report_status"] == "pending"
        
        # Verify metrics are merged
        assert "start_time" in result["system_metrics"]
        assert "end_time" in result["system_metrics"]

class TestGraphStructure:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4"
        )
    
    def test_main_graph_structure(self, settings):
        """Test main graph has correct structure"""
        
        workflow = MultiAgentWorkflow(settings)
        graph = workflow.graph
        
        # Verify nodes exist
        nodes = list(graph.get_graph().nodes())
        expected_nodes = [
            "main_supervisor",
            "research_team",
            "reporting_team",
            "input_validator",
            "error_handler"
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    def test_research_team_graph_structure(self, settings):
        """Test research team subgraph structure"""
        
        workflow = MultiAgentWorkflow(settings)
        research_graph = workflow._build_research_team_graph()
        
        # Verify research team nodes
        nodes = list(research_graph.get_graph().nodes())
        expected_nodes = [
            "research_supervisor",
            "medical_researcher",
            "financial_researcher",
            "research_validator"
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    def test_reporting_team_graph_structure(self, settings):
        """Test reporting team subgraph structure"""
        
        workflow = MultiAgentWorkflow(settings)
        reporting_graph = workflow._build_reporting_team_graph()
        
        # Verify reporting team nodes
        nodes = list(reporting_graph.get_graph().nodes())
        expected_nodes = [
            "reporting_supervisor",
            "document_creator",
            "summarizer",
            "reporting_validator"
        ]
        
        for node in expected_nodes:
            assert node in nodes
```

## 🎯 Success Criteria

### Functional Requirements:
- [ ] Main graph orchestrates teams correctly
- [ ] Research team subgraph coordinates medical and financial researchers
- [ ] Reporting team subgraph coordinates document creation and summarization
- [ ] State transitions work reliably between teams
- [ ] Parallel execution improves performance
- [ ] Error handling prevents workflow failures

### Quality Requirements:
- [ ] Integration tests achieve >90% coverage
- [ ] State reducers handle concurrent updates correctly
- [ ] Graph visualization works properly
- [ ] Memory persistence functions correctly
- [ ] Performance monitoring captures metrics

### Performance Requirements:
- [ ] Complete workflow finishes in <5 minutes
- [ ] Parallel execution reduces total time by 30%+
- [ ] State transitions complete in <100ms
- [ ] Memory usage remains stable throughout execution
- [ ] Graph compilation completes in <2 seconds

## 📊 Stage 4 Metrics

### Time Allocation:
- Main graph assembly: 60 minutes
- Subgraph implementation: 45 minutes
- State management and reducers: 45 minutes
- Parallel execution framework: 30 minutes
- Integration testing: 45 minutes
- Graph visualization and debugging: 15 minutes

### Success Indicators:
- All graphs compile without errors
- State transitions work correctly
- Parallel execution functions properly
- Integration tests pass consistently
- Graph visualization displays correctly
- Performance metrics meet targets

## 🔄 Next Steps

After completing Stage 4, proceed to:
1. **Stage 5**: CLI Interface & User Experience
2. Build command-line interface
3. Implement interactive mode
4. Add progress tracking and streaming

---

*This stage creates the sophisticated workflow orchestration that enables seamless coordination between all agents while maintaining state consistency and enabling parallel execution for optimal performance.*
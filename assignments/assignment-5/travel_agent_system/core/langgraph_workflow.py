"""
LangGraph Workflow Implementation - Main StateGraph workflow for travel planning
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from .graph_state import (
    TravelPlanState, 
    create_initial_state,
    validate_input,
    is_parallel_phase_complete,
    has_sufficient_data_for_itinerary,
    mark_complete,
    get_processing_summary
)

import logging

logger = logging.getLogger(__name__)


class LangGraphTravelWorkflow:
    """
    LangGraph-based travel planning workflow implementation.
    Replaces ThreadPoolExecutor with StateGraph orchestration.
    """
    
    def __init__(self):
        """Initialize the LangGraph workflow"""
        self.graph = None
        self.checkpointer = MemorySaver()
        
        # Initialize ToolNode with travel tools
        from ..tools.langgraph_tools import TRAVEL_TOOLS
        self.tool_node = ToolNode(TRAVEL_TOOLS)
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the StateGraph workflow"""
        # Create StateGraph with state schema  
        # Note: Reducers are handled in the state schema definition via TypedDict annotations
        workflow = StateGraph(TravelPlanState)
        
        # Import node functions (will be created in nodes.py)
        from .nodes import (
            input_validation_node,
            weather_node,
            attractions_node,
            hotels_node,
            data_aggregation_node,
            itinerary_node,
            summary_generation_node,
            error_handling_node
        )
        
        # Add nodes to the graph
        workflow.add_node("input_validation", input_validation_node)
        workflow.add_node("data_aggregation", data_aggregation_node)  # Handles parallel execution internally
        workflow.add_node("itinerary", itinerary_node)
        workflow.add_node("summary_generation", summary_generation_node)
        workflow.add_node("error_handling", error_handling_node)
        
        # Set entry point
        workflow.set_entry_point("input_validation")
        
        # Add edges for workflow flow
        # Input validation -> data aggregation (with error check) or error handling
        workflow.add_conditional_edges(
            "input_validation",
            self._route_after_validation,
            {
                "continue": "data_aggregation",
                "error": "error_handling"
            }
        )
        
        # For parallel execution, we'll trigger all three agents from data_aggregation node
        # Data aggregation will handle calling the parallel agents and then aggregating results
        
        # Data aggregation handles parallel execution internally, then routes to itinerary
        
        # Data aggregation -> itinerary or error handling
        workflow.add_conditional_edges(
            "data_aggregation",
            self._route_after_data_aggregation,
            {
                "create_itinerary": "itinerary",
                "error": "error_handling"
            }
        )
        
        # Itinerary -> summary generation
        workflow.add_edge("itinerary", "summary_generation")
        
        # Summary generation -> END
        workflow.add_edge("summary_generation", END)
        
        # Error handling -> END
        workflow.add_edge("error_handling", END)
        
        # Compile the graph with enhanced checkpointing
        self.graph = workflow.compile(
            checkpointer=self.checkpointer,
            debug=False  # Disable debug output for cleaner summary display
        )
    
    def _route_after_validation(self, state: TravelPlanState) -> Literal["continue", "error"]:
        """
        Route after input validation.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Next route ("continue" or "error")
        """
        validation_errors = validate_input(state)
        
        if validation_errors:
            logger.error(f"Input validation failed: {validation_errors}")
            return "error"
        
        logger.info("Input validation passed, continuing to data aggregation")
        return "continue"
    
    def _route_after_data_aggregation(self, state: TravelPlanState) -> Literal["create_itinerary", "error"]:
        """
        Route after data aggregation.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Next route ("create_itinerary" or "error")
        """
        if not is_parallel_phase_complete(state):
            logger.warning("Not all parallel agents have completed")
            return "error"
        
        if not has_sufficient_data_for_itinerary(state):
            logger.error("Insufficient data for itinerary creation")
            return "error"
        
        logger.info("Sufficient data available, proceeding to itinerary creation")
        return "create_itinerary"
    
    async def aquery(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a travel query asynchronously using LangGraph.
        
        Args:
            query: Travel planning query
            config: Optional configuration for the workflow
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            # Parse query to extract travel parameters
            parsed_data = self._parse_query(query)
            
            # Create initial state
            initial_state = create_initial_state(**parsed_data)
            
            # Configure workflow execution
            workflow_config = config or {}
            workflow_config.setdefault("configurable", {})
            workflow_config["configurable"]["thread_id"] = f"travel_plan_{hash(query)}"
            
            # Execute workflow
            logger.info(f"Starting LangGraph workflow for query: {query}")
            
            final_state = None
            async for state in self.graph.astream(initial_state, config=workflow_config):
                # Stream through workflow execution
                logger.debug(f"Workflow state update: {list(state.keys())}")
                final_state = state
            
            # Extract results from final state
            if final_state:
                # Get the last state update
                last_node_state = list(final_state.values())[-1]
                return self._format_results(last_node_state)
            else:
                return {"error": "Workflow execution failed", "success": False}
                
        except Exception as e:
            logger.error(f"LangGraph workflow error: {str(e)}")
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "success": False
            }
    
    def query(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a travel query synchronously using LangGraph.
        
        Args:
            query: Travel planning query
            config: Optional configuration for the workflow
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            # Parse query to extract travel parameters
            parsed_data = self._parse_query(query)
            
            # Create initial state
            initial_state = create_initial_state(**parsed_data)
            
            # Configure workflow execution
            workflow_config = config or {}
            workflow_config.setdefault("configurable", {})
            workflow_config["configurable"]["thread_id"] = f"travel_plan_{hash(query)}"
            
            # Execute workflow synchronously
            logger.info(f"Starting LangGraph workflow for query: {query}")
            
            final_state = None
            for state in self.graph.stream(initial_state, config=workflow_config):
                # Stream through workflow execution
                logger.debug(f"Workflow state update: {list(state.keys())}")
                final_state = state
            
            # Extract results from final state
            if final_state:
                # Get the last state update
                last_node_state = list(final_state.values())[-1]
                return self._format_results(last_node_state)
            else:
                return {"error": "Workflow execution failed", "success": False}
                
        except Exception as e:
            logger.error(f"LangGraph workflow error: {str(e)}")
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "success": False
            }
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language travel query into structured data.
        
        Args:
            query: Natural language travel query
            
        Returns:
            Dict[str, Any]: Parsed travel parameters with raw_query for LLM processing
        """
        # Instead of parsing here, we'll pass the raw query to the input validation node
        # which will use the LLM + Pydantic parser for robust natural language understanding
        return {
            "raw_query": query  # This will trigger LLM parsing in input_validation_node
        }
    
    def _format_results(self, state: TravelPlanState) -> Dict[str, Any]:
        """
        Format the final state into user-friendly results.
        
        Args:
            state: Final workflow state
            
        Returns:
            Dict[str, Any]: Formatted results
        """
        try:
            # Get processing summary
            summary = get_processing_summary(state)
            
            # Check for success
            success = state.get("completion_time") is not None and len(state.get("errors", [])) == 0
            
            result = {
                "status": "success" if success else "error",
                "destination": state.get("destination"),
                "processing_summary": summary,
                "itinerary": state.get("itinerary_data"),
                "errors": state.get("errors", [])
            }
            
            # Add trip summary if available (generated by summary_generation_node)
            if state.get("trip_summary"):
                result["trip_summary"] = state["trip_summary"]
            
            # Add detailed data if available
            if state.get("weather_data"):
                result["weather"] = state["weather_data"]
            if state.get("attractions_data"):
                result["attractions"] = state["attractions_data"]
            if state.get("hotels_data"):
                result["hotels"] = state["hotels_data"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return {
                "success": False,
                "error": f"Result formatting failed: {str(e)}",
                "raw_state": dict(state) if state else {}
            }
    
    def get_graph_visualization(self) -> str:
        """
        Get Mermaid visualization of the workflow graph.
        
        Returns:
            str: Mermaid diagram representation
        """
        try:
            if hasattr(self.graph, 'get_graph'):
                return self.graph.get_graph().draw_mermaid()
            else:
                return "Graph visualization not available"
        except Exception as e:
            logger.error(f"Error generating graph visualization: {str(e)}")
            return f"Visualization error: {str(e)}"
    
    def save_graph_image(self, filename: str = "travel_workflow.png"):
        """
        Save workflow graph as PNG image.
        
        Args:
            filename: Output filename for the graph image
        """
        try:
            if hasattr(self.graph, 'get_graph'):
                with open(filename, 'wb') as f:
                    f.write(self.graph.get_graph().draw_mermaid_png())
                logger.info(f"Workflow graph saved as {filename}")
            else:
                logger.warning("Graph visualization not available")
        except Exception as e:
            logger.error(f"Error saving graph image: {str(e)}")
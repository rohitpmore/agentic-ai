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
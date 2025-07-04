from typing import Dict, Any, Literal, Optional
from langgraph.types import Command
from datetime import datetime
import uuid

class HandoffProtocol:
    """Standardized handoff protocol between agents"""
    
    @staticmethod
    def create_handoff(
        destination: str,
        payload: Dict[str, Any],
        urgency: Literal["low", "medium", "high"] = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Command:
        """Create a standardized handoff command"""
        
        handoff_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        return Command(
            goto=destination,
            update={
                "handoff_data": payload,
                "handoff_metadata": {
                    "id": handoff_id,
                    "timestamp": timestamp,
                    "urgency": urgency,
                    "source": "system",
                    "additional_metadata": metadata or {}
                }
            }
        )
    
    @staticmethod
    def create_error_handoff(
        error: Exception,
        current_agent: str,
        state: Dict[str, Any]
    ) -> Command:
        """Create an error handoff command"""
        
        return Command(
            goto="error_handler",
            update={
                "error_data": {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "source_agent": current_agent,
                    "timestamp": datetime.now().isoformat(),
                    "state_snapshot": state
                }
            }
        )
    
    @staticmethod
    def create_completion_handoff(
        results: Dict[str, Any],
        next_destination: str = "end"
    ) -> Command:
        """Create a completion handoff command"""
        
        return Command(
            goto=next_destination,
            update={
                "completion_data": {
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            }
        )
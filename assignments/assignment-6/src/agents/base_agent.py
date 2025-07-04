from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langgraph.types import Command
import logging
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, name: str, model: Optional[Any] = None):
        self.name = name
        self.model = model
        self.logger = logging.getLogger(f"agent.{name}")
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_processing_time": 0
        }
        
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Command:
        """Process state and return command for next action"""
        pass
        
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required state fields"""
        pass
        
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input state"""
        required_fields = self.get_required_fields()
        missing_fields = [field for field in required_fields if field not in state]
        
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
            
        return True
        
    def handle_error(self, error: Exception, state: Dict[str, Any]) -> Command:
        """Handle errors gracefully"""
        self.logger.error(f"Error in {self.name}: {str(error)}")
        self.metrics["failed_calls"] += 1
        
        return Command(
            goto="error_handler",
            update={
                "error": {
                    "agent": self.name,
                    "message": str(error),
                    "timestamp": datetime.now().isoformat(),
                    "state_snapshot": state
                }
            }
        )
        
    def record_metrics(self, start_time: datetime, success: bool = True):
        """Record performance metrics"""
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics["total_calls"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
            
        self.logger.info(f"Agent {self.name} processed in {processing_time:.2f}s")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if self.metrics["total_calls"] > 0:
            avg_time = self.metrics["total_processing_time"] / self.metrics["total_calls"]
            success_rate = self.metrics["successful_calls"] / self.metrics["total_calls"]
        else:
            avg_time = 0
            success_rate = 0
            
        return {
            **self.metrics,
            "average_processing_time": avg_time,
            "success_rate": success_rate
        }
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
                        delay = self.base_delay * (2 ** attempt)
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
        
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        if "rate limit" in str(error).lower():
            return True
        
        if "500" in str(error) or "502" in str(error) or "503" in str(error):
            return True
        
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        return True
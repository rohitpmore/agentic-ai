"""
Validation agent for quality control and feedback loops
"""

from typing import Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.state import AgentState
from config import config


class ValidationAgent:
    """
    Validation agent for response quality control and feedback loops.
    
    This agent validates the quality and relevance of responses from other agents
    and can trigger feedback loops if validation fails.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI):
        """
        Initialize the validation agent.
        
        Args:
            model: Google Gemini model instance
        """
        self.model = model
        self.max_retries = config.max_retries
        self.retry_count = 0
    
    def validate_response(self, original_question: str, response: str) -> Tuple[bool, str]:
        """
        Validate a response for quality and relevance.
        
        Args:
            original_question: The original user question
            response: The response to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_reason)
        """
        try:
            validation_prompt = f"""
            Evaluate the following response for quality and relevance to the original question.
            
            Original Question: {original_question}
            Response: {response}
            
            Criteria:
            1. Is the response relevant to the question?
            2. Is the response complete and informative?
            3. Is the response accurate based on the context?
            
            Reply with only "PASS" or "FAIL" followed by a brief reason.
            """
            
            validation_result = self.model.invoke(validation_prompt)
            validation_text = str(validation_result.content if hasattr(validation_result, 'content') else validation_result)
            
            is_valid = "PASS" in validation_text.upper()
            return is_valid, validation_text
            
        except Exception as e:
            print(f"⚠️ Validation error: {e}")
            # Fallback to basic validation
            is_valid = len(str(response)) > 20  # Basic length check
            reason = f"Fallback validation: {'PASS' if is_valid else 'FAIL'} - LLM validation failed: {str(e)}"
            return is_valid, reason
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the agent state and validate the response.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: Updated state with validation result
        """
        print("✅ Validation Node processing response")
        
        response = state["messages"][-1]
        original_question = state["messages"][0]
        
        is_valid, validation_reason = self.validate_response(str(original_question), str(response))
        
        if is_valid:
            print("✅ Response validation passed")
            print(f"✅ Validation reason: {validation_reason}")
            return {"messages": [response, "VALIDATION_PASS"]}
        else:
            print("❌ Response validation failed")
            print(f"❌ Validation reason: {validation_reason}")
            return {"messages": [response, "VALIDATION_FAIL"]}
    
    def feedback_router(self, state: AgentState) -> str:
        """
        Route based on validation results with retry logic.
        
        Args:
            state: Current agent state
            
        Returns:
            str: Routing decision
        """
        validation_status = state["messages"][-1]
        print(f"🔄 Feedback router checking: {validation_status}")
        
        if "VALIDATION_PASS" in str(validation_status):
            print("✅ Validation passed - Ending workflow")
            self.retry_count = 0  # Reset retry count on success
            return "END"
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                print(f"❌ Maximum retries ({self.max_retries}) reached - Ending workflow")
                self.retry_count = 0  # Reset retry count
                return "END"
            else:
                print(f"❌ Validation failed (attempt {self.retry_count}/{self.max_retries}) - Routing back to Supervisor for retry")
                return "RETRY"
    
    def reset_retry_count(self):
        """Reset the retry counter."""
        self.retry_count = 0
    
    def get_retry_count(self) -> int:
        """Get the current retry count."""
        return self.retry_count
    
    def validate_query_format(self, query: str) -> Tuple[bool, str]:
        """
        Validate if a query is well-formed.
        
        Args:
            query: The query to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        if not query or not query.strip():
            return False, "Query is empty or only contains whitespace"
        
        if len(query.strip()) < 3:
            return False, "Query is too short (minimum 3 characters)"
        
        if len(query) > 1000:
            return False, "Query is too long (maximum 1000 characters)"
        
        return True, "Query format is valid" 
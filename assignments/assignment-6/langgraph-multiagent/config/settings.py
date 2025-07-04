from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os

class Settings(BaseSettings):
    """Application configuration using Pydantic BaseSettings"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    
    # Model Configuration
    supervisor_model: str = Field("gpt-4", env="SUPERVISOR_MODEL")
    researcher_model: str = Field("gpt-4", env="RESEARCHER_MODEL")
    reporter_model: str = Field("gpt-3.5-turbo", env="REPORTER_MODEL")
    
    # System Configuration
    max_retries: int = Field(3, env="MAX_RETRIES")
    timeout_seconds: int = Field(300, env="TIMEOUT_SECONDS")
    recursion_limit: int = Field(25, env="RECURSION_LIMIT")
    
    # Output Configuration
    output_directory: str = Field("./outputs", env="OUTPUT_DIRECTORY")
    document_template: str = Field("research_template.docx", env="DOCUMENT_TEMPLATE")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # Performance Configuration
    enable_streaming: bool = Field(True, env="ENABLE_STREAMING")
    batch_size: int = Field(5, env="BATCH_SIZE")
    
    # Development Configuration
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    test_mode: bool = Field(False, env="TEST_MODE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_model_config(self, agent_type: str) -> Dict[str, Any]:
        """Get model configuration for specific agent type"""
        model_map = {
            "supervisor": self.supervisor_model,
            "researcher": self.researcher_model,
            "reporter": self.reporter_model
        }
        
        return {
            "model": model_map.get(agent_type, self.supervisor_model),
            "temperature": 0 if agent_type == "supervisor" else 0.1,
            "max_tokens": 2000,
            "timeout": self.timeout_seconds
        }
        
    def ensure_output_directory(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_directory, exist_ok=True)
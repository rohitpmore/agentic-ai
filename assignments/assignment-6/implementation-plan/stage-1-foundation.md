# Stage 1: Foundation & Project Setup

**Timeline:** 3-4 hours  
**Status:** ⏳ Pending  
**Priority:** High

## 📋 Overview

This stage establishes the foundational architecture for the LangGraph Multi-Agent Hierarchical Workflow System. We'll create the project structure, implement base classes, define state schemas, and setup the development environment following professional standards.

## 🎯 Key Deliverables

### ✅ Project Structure
```
langgraph-multiagent/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── supervisor.py
│   │   ├── research/
│   │   │   ├── __init__.py
│   │   │   ├── medical_researcher.py
│   │   │   └── financial_researcher.py
│   │   └── reporting/
│   │       ├── __init__.py
│   │       ├── document_creator.py
│   │       └── summarizer.py
│   ├── state/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── arxiv_tool.py
│   │   └── document_tools.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── handoff.py
│   │   └── logging_config.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_base_agent.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_team_coordination.py
│   └── e2e/
│       ├── __init__.py
│       └── test_full_workflow.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
└── CLAUDE.md
```

### ✅ Base Agent Implementation
```python
# src/agents/base_agent.py
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
```

### ✅ State Schema Definitions
```python
# src/state/schemas.py
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from datetime import datetime

class ResearchState(MessagesState):
    """State for research team operations"""
    research_topic: str
    medical_findings: Dict[str, Any]
    financial_findings: Dict[str, Any]
    research_status: Literal["pending", "in_progress", "completed", "failed"]
    research_metadata: Dict[str, Any]
    
class ReportingState(MessagesState):
    """State for reporting team operations"""
    research_data: Dict[str, Any]
    document_path: str
    summary: str
    report_status: Literal["pending", "in_progress", "completed", "failed"]
    report_metadata: Dict[str, Any]
    
class SupervisorState(MessagesState):
    """Main supervisor state combining all team states"""
    current_team: Literal["research", "reporting", "end"]
    research_state: ResearchState
    reporting_state: ReportingState
    task_description: str
    final_output: Dict[str, Any]
    error_state: Optional[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    
class AgentMetrics(TypedDict):
    """Metrics for individual agents"""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime]
    tokens_used: int
    success: bool
    error_message: Optional[str]
    processing_time: float
    
class SystemMetrics(TypedDict):
    """System-wide metrics"""
    total_agents: int
    active_agents: int
    total_processing_time: float
    successful_handoffs: int
    failed_handoffs: int
    documents_generated: int
    api_calls_made: int
```

### ✅ Configuration Management
```python
# config/settings.py
from pydantic import BaseSettings, Field
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
```

### ✅ Handoff Protocol
```python
# src/utils/handoff.py
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
```

### ✅ Logging Configuration
```python
# src/utils/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger
from typing import Optional

def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration"""
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if format_type == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set specific logger levels
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### ✅ Requirements File
```txt
# requirements.txt
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
arxiv>=2.0.0
python-docx>=1.0.0
reportlab>=4.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
python-json-logger>=2.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Development
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0

# Utilities
click>=8.0.0
rich>=13.0.0
typer>=0.9.0
```

### ✅ Environment Template
```bash
# .env.example
# API Keys
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_API_KEY=your-langchain-api-key-here

# Model Configuration
SUPERVISOR_MODEL=gpt-4
RESEARCHER_MODEL=gpt-4
REPORTER_MODEL=gpt-3.5-turbo

# System Configuration
MAX_RETRIES=3
TIMEOUT_SECONDS=300
RECURSION_LIMIT=25

# Output Configuration
OUTPUT_DIRECTORY=./outputs
DOCUMENT_TEMPLATE=research_template.docx

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance Configuration
ENABLE_STREAMING=true
BATCH_SIZE=5

# Development Configuration
DEBUG_MODE=false
TEST_MODE=false
```

### ✅ Testing Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
    --asyncio-mode=auto
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
```

## 🔧 Implementation Tasks

### Task 1: Project Structure Setup
```bash
# Create directory structure
mkdir -p langgraph-multiagent/{src/{agents/{research,reporting},state,tools,utils},tests/{unit,integration,e2e},config}

# Create __init__.py files
touch langgraph-multiagent/src/__init__.py
touch langgraph-multiagent/src/agents/__init__.py
touch langgraph-multiagent/src/agents/research/__init__.py
touch langgraph-multiagent/src/agents/reporting/__init__.py
touch langgraph-multiagent/src/state/__init__.py
touch langgraph-multiagent/src/tools/__init__.py
touch langgraph-multiagent/src/utils/__init__.py
touch langgraph-multiagent/tests/__init__.py
touch langgraph-multiagent/tests/unit/__init__.py
touch langgraph-multiagent/tests/integration/__init__.py
touch langgraph-multiagent/tests/e2e/__init__.py
touch langgraph-multiagent/config/__init__.py
```

### Task 2: Base Classes Implementation
1. Implement `BaseAgent` with validation and error handling
2. Create state schemas with proper TypedDict definitions
3. Setup configuration management with Pydantic
4. Implement handoff protocol utilities
5. Configure logging infrastructure

### Task 3: Environment Setup
1. Create virtual environment
2. Install dependencies from requirements.txt
3. Setup environment variables
4. Configure testing framework
5. Setup development tools (black, isort, flake8, mypy)

### Task 4: Initial Testing
1. Create basic unit tests for base classes
2. Test configuration loading
3. Test state validation
4. Test handoff protocol
5. Verify logging configuration

## ✅ Success Criteria

### Functional Requirements:
- [ ] Project structure is properly organized
- [ ] All imports resolve without circular dependencies
- [ ] Base classes can be instantiated successfully
- [ ] State validation works correctly
- [ ] Configuration loads from environment variables
- [ ] Logging system is properly configured
- [ ] Testing framework is operational

### Quality Requirements:
- [ ] Code follows PEP 8 standards
- [ ] Type hints are properly implemented
- [ ] Documentation is comprehensive
- [ ] Error handling is robust
- [ ] Unit tests pass with >90% coverage

### Performance Requirements:
- [ ] Configuration loads in <1 second
- [ ] State validation completes in <100ms
- [ ] Base class instantiation is efficient
- [ ] Memory usage is optimized

## 🔍 Testing Strategy

### Unit Tests:
```python
# tests/unit/test_base_agent.py
import pytest
from src.agents.base_agent import BaseAgent
from src.state.schemas import SupervisorState

class TestBaseAgent:
    def test_base_agent_initialization(self):
        # Test abstract base class cannot be instantiated
        with pytest.raises(TypeError):
            BaseAgent("test")
    
    def test_validate_input_success(self):
        # Test successful input validation
        pass
        
    def test_validate_input_failure(self):
        # Test failed input validation
        pass
        
    def test_handle_error(self):
        # Test error handling
        pass
        
    def test_record_metrics(self):
        # Test metrics recording
        pass
```

### Integration Tests:
```python
# tests/integration/test_configuration.py
import pytest
from config.settings import Settings

class TestConfiguration:
    def test_config_loading(self):
        # Test configuration loading from environment
        pass
        
    def test_model_config_generation(self):
        # Test model configuration generation
        pass
```

## 📝 Documentation Requirements

### README.md:
```markdown
# LangGraph Multi-Agent Hierarchical Workflow System

## Overview
[Project description and architecture overview]

## Installation
[Installation instructions]

## Configuration
[Configuration guide]

## Usage
[Usage examples]

## Development
[Development setup and contribution guidelines]
```

### CLAUDE.md:
```markdown
# Project-Specific CLAUDE.md

## Project Context
[Project-specific context and requirements]

## Development Guidelines
[Project-specific development guidelines]

## Testing Strategy
[Project-specific testing requirements]
```

## 🎯 Next Steps

After completing Stage 1, proceed to:
1. **Stage 2**: Core Agent Development
2. Implement supervisor agents
3. Create routing logic
4. Add error handling mechanisms

## 📊 Stage 1 Metrics

### Time Allocation:
- Project structure setup: 30 minutes
- Base classes implementation: 90 minutes
- Configuration management: 60 minutes
- Logging and utilities: 45 minutes
- Testing setup: 45 minutes
- Documentation: 30 minutes

### Success Indicators:
- All files created without errors
- Dependencies installed successfully
- Tests run without failures
- Configuration validates correctly
- Documentation is comprehensive

---

*This stage establishes the solid foundation required for the sophisticated multi-agent system, following professional development standards and best practices.*
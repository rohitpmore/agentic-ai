# Product Requirements Document: LangGraph Multi-Agent Hierarchical Workflow System

## 1. Executive Summary

This document outlines the requirements for a sophisticated multi-agent system using LangGraph's hierarchical supervisor architecture. The system consists of a main supervisor agent orchestrating two specialized teams: a Research Team (medical/pharmaceutical and financial researchers) and a Reporting Team (document creation and summarization agents).

### Key Features
- **Hierarchical Architecture**: Top-level supervisor managing specialized teams
- **Modular Design**: Each agent as an independent, testable module
- **Handoff Capability**: Seamless communication between agents using Command objects
- **State Management**: Shared and private state schemas for different agent needs
- **Test-Driven Development**: Comprehensive testing at unit, integration, and system levels

## 2. Project Overview

### 2.1 Objectives
- Build a scalable multi-agent system for research and reporting tasks
- Implement supervisor-based coordination for optimal task distribution
- Enable specialized agents to handle domain-specific responsibilities
- Generate professional documents (PDF/DOCX) with research findings
- Provide concise summaries of complex research results

### 2.2 System Architecture

```
┌─────────────────┐
│   Supervisor    │
│     Agent       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Team 1 │ │Team 2 │
│Research│ │Report │
└───┬───┘ └───┬───┘
    │         │
┌───┴───┐ ┌──┴────┐
│Medical│ │Doc    │
│Finance│ │Summary│
└───────┘ └───────┘
```

## 3. Technical Architecture

### 3.1 Core Components

#### 3.1.1 State Schemas

```python
from typing import TypedDict, List, Dict, Any, Literal, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage

class ResearchState(MessagesState):
    """State for research team operations"""
    research_topic: str
    medical_findings: Dict[str, Any]
    financial_findings: Dict[str, Any]
    research_status: Literal["pending", "in_progress", "completed"]
    
class ReportingState(MessagesState):
    """State for reporting team operations"""
    research_data: Dict[str, Any]
    document_path: str
    summary: str
    report_status: Literal["pending", "in_progress", "completed"]
    
class SupervisorState(MessagesState):
    """Main supervisor state combining all team states"""
    current_team: Literal["research", "reporting", "end"]
    research_state: ResearchState
    reporting_state: ReportingState
    task_description: str
    final_output: Dict[str, Any]
```

#### 3.1.2 Agent Specifications

**Supervisor Agent**
- **Purpose**: Orchestrate workflow between teams
- **Capabilities**: 
  - Parse user requests
  - Determine task routing
  - Monitor team progress
  - Handle errors and retries
- **Tools**: None (uses Command for routing)

**Medical/Pharmacy Researcher Agent**
- **Purpose**: Conduct medical and pharmaceutical research
- **Capabilities**:
  - Literature review
  - Drug interaction analysis
  - Clinical data interpretation
- **Tools**: arXiv API for accessing research papers

**Financial Researcher Agent**
- **Purpose**: Conduct general financial research and analysis
- **Capabilities**:
  - Market analysis
  - Economic trends analysis
  - Financial data interpretation
- **Tools**: arXiv API for accessing financial/economic research papers

**Document Creator Agent**
- **Purpose**: Generate professional documents
- **Capabilities**:
  - Format research into structured documents
  - Create PDF/DOCX files
  - Apply templates and styling
- **Tools**: Document generation libraries (python-docx, reportlab)

**Summary Agent**
- **Purpose**: Create concise summaries
- **Capabilities**:
  - Extract key findings
  - Generate executive summaries
  - Create bullet-point overviews
- **Tools**: Text summarization models

### 3.2 Handoff Implementation

```python
from langgraph.types import Command

class HandoffProtocol:
    """Standardized handoff protocol between agents"""
    
    @staticmethod
    def create_handoff(
        destination: str,
        payload: Dict[str, Any],
        urgency: Literal["low", "medium", "high"] = "medium"
    ) -> Command:
        return Command(
            goto=destination,
            update={
                "handoff_data": payload,
                "handoff_timestamp": datetime.now().isoformat(),
                "urgency": urgency
            }
        )
```

## 4. Component Implementation Guidelines

### 4.1 Project Structure

```
langgraph-multiagent/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
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
│   │   └── handoff.py
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── config/
│   └── settings.py
├── requirements.txt
└── README.md
```

### 4.2 Modular Agent Template

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langgraph.types import Command
import logging

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, model: Optional[Any] = None):
        self.name = name
        self.model = model
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Command:
        """Process state and return command for next action"""
        pass
        
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input state"""
        required_fields = self.get_required_fields()
        return all(field in state for field in required_fields)
        
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required state fields"""
        pass
```

### 4.3 Test-Driven Development Strategy

#### Unit Tests
```python
import pytest
from unittest.mock import Mock, patch
from src.agents.research.medical_researcher import MedicalResearcher

class TestMedicalResearcher:
    def test_initialization(self):
        agent = MedicalResearcher()
        assert agent.name == "medical_researcher"
        
    def test_process_valid_state(self):
        agent = MedicalResearcher()
        state = {
            "research_topic": "diabetes medications",
            "messages": []
        }
        result = agent.process(state)
        assert isinstance(result, Command)
        
    def test_invalid_state_handling(self):
        agent = MedicalResearcher()
        with pytest.raises(ValueError):
            agent.process({})
```

#### Integration Tests
```python
class TestResearchTeamIntegration:
    def test_medical_to_financial_handoff(self):
        # Test handoff between medical and financial researchers
        pass
        
    def test_research_team_supervisor(self):
        # Test research team supervisor coordination
        pass
```

## 5. Implementation Examples

### 5.1 Basic Supervisor Implementation

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

class MainSupervisor:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        
    def __call__(self, state: SupervisorState) -> Command[Literal["research_team", "reporting_team", "end"]]:
        # Analyze current state
        messages = state.get("messages", [])
        task_description = state.get("task_description", "")
        
        # Determine next team based on state
        if not state.get("research_state", {}).get("research_status") == "completed":
            return Command(
                goto="research_team",
                update={
                    "current_team": "research",
                    "messages": messages + [f"Routing to research team for: {task_description}"]
                }
            )
        elif not state.get("reporting_state", {}).get("report_status") == "completed":
            return Command(
                goto="reporting_team",
                update={
                    "current_team": "reporting",
                    "research_data": {
                        "medical_findings": state.get("medical_findings", {}),
                        "financial_findings": state.get("financial_findings", {})
                    }
                }
            )
        else:
            return Command(
                goto="end",
                update={
                    "final_output": {
                        "document_path": state["reporting_state"]["document_path"],
                        "summary": state["reporting_state"]["summary"]
                    }
                }
            )
```

### 5.2 Research Team Implementation

```python
class ResearchTeamSupervisor:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        
    def __call__(self, state: ResearchState) -> Command[Literal["medical_researcher", "financial_researcher", "main_supervisor"]]:
        # Check research progress
        medical_done = bool(state.get("medical_findings"))
        financial_done = bool(state.get("financial_findings"))
        
        if not medical_done:
            return Command(
                goto="medical_researcher",
                update={"research_status": "in_progress"}
            )
        elif not financial_done:
            return Command(
                goto="financial_researcher",
                update={"research_status": "in_progress"}
            )
        else:
            return Command(
                goto="main_supervisor",
                update={
                    "research_status": "completed",
                    "combined_findings": {
                        "medical": state["medical_findings"],
                        "financial": state["financial_findings"]
                    }
                },
                graph=Command.PARENT  # Return to parent graph
            )
```

### 5.3 Complete Graph Assembly

```python
def build_research_team_graph():
    """Build research team subgraph"""
    builder = StateGraph(ResearchState)
    
    # Add nodes
    builder.add_node("research_supervisor", ResearchTeamSupervisor())
    builder.add_node("medical_researcher", MedicalResearcher())
    builder.add_node("financial_researcher", FinancialResearcher())
    
    # Add edges
    builder.add_edge(START, "research_supervisor")
    builder.add_edge("medical_researcher", "research_supervisor")
    builder.add_edge("financial_researcher", "research_supervisor")
    
    return builder.compile()

def build_reporting_team_graph():
    """Build reporting team subgraph"""
    builder = StateGraph(ReportingState)
    
    # Add nodes
    builder.add_node("reporting_supervisor", ReportingTeamSupervisor())
    builder.add_node("document_creator", DocumentCreator())
    builder.add_node("summarizer", Summarizer())
    
    # Add edges
    builder.add_edge(START, "reporting_supervisor")
    builder.add_edge("document_creator", "reporting_supervisor")
    builder.add_edge("summarizer", "reporting_supervisor")
    
    return builder.compile()

def build_main_graph():
    """Build main hierarchical graph"""
    builder = StateGraph(SupervisorState)
    
    # Add main supervisor
    builder.add_node("main_supervisor", MainSupervisor())
    
    # Add team subgraphs
    builder.add_node("research_team", build_research_team_graph())
    builder.add_node("reporting_team", build_reporting_team_graph())
    
    # Add edges
    builder.add_edge(START, "main_supervisor")
    builder.add_edge("research_team", "main_supervisor")
    builder.add_edge("reporting_team", "main_supervisor")
    
    return builder.compile()
```

## 6. Sample Runnable Examples

### 6.1 Basic Research Request

```python
async def example_basic_research():
    """Example: Research AI applications in healthcare and finance"""
    
    # Initialize graph
    graph = build_main_graph()
    
    # Create initial state
    initial_state = {
        "messages": ["Research AI applications in healthcare and financial sectors"],
        "task_description": "Comprehensive analysis of AI in healthcare and finance",
        "research_state": {
            "research_topic": "AI applications in healthcare and finance",
            "research_status": "pending"
        },
        "reporting_state": {
            "report_status": "pending"
        }
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    print(f"Document created: {result['final_output']['document_path']}")
    print(f"Summary: {result['final_output']['summary']}")
```

### 6.2 Advanced Multi-Topic Research

```python
async def example_comparative_analysis():
    """Example: Compare machine learning applications across domains"""
    
    graph = build_main_graph()
    
    initial_state = {
        "messages": ["Compare ML applications in medical diagnostics vs financial forecasting"],
        "task_description": "Comparative analysis of ML applications",
        "research_state": {
            "research_topic": "ML applications comparison",
            "comparison_criteria": ["accuracy", "interpretability", "regulatory requirements", "implementation costs"],
            "research_status": "pending"
        },
        "reporting_state": {
            "report_format": "comparative_table",
            "report_status": "pending"
        }
    }
    
    # Stream results for real-time updates
    async for chunk in graph.astream(initial_state):
        print(f"Progress: {chunk}")
```

### 6.3 Error Handling Example

```python
async def example_with_retry():
    """Example: Handle errors and retry logic"""
    
    graph = build_main_graph()
    
    # Configure with retry logic
    config = {
        "recursion_limit": 25,
        "callbacks": [LangChainTracer()]
    }
    
    try:
        result = await graph.ainvoke(
            initial_state,
            config=config
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        # Implement retry logic
        pass
```

## 7. Testing Strategy

### 7.1 Unit Test Examples

```python
# tests/unit/test_medical_researcher.py
import pytest
from src.agents.research.medical_researcher import MedicalResearcher

# tests/unit/test_medical_researcher.py
import pytest
from src.agents.research.medical_researcher import MedicalResearcher

class TestMedicalResearcher:
    @pytest.fixture
    def agent(self):
        return MedicalResearcher()
    
    def test_drug_interaction_check(self, agent):
        state = {
            "research_topic": "metformin interactions",
            "messages": []
        }
        result = agent.process(state)
        assert "medical_findings" in result.update
        assert "drug_interactions" in result.update["medical_findings"]
    
    def test_clinical_data_extraction(self, agent):
        state = {
            "research_topic": "insulin efficacy studies",
            "messages": []
        }
        result = agent.process(state)
        assert "clinical_trials" in result.update["medical_findings"]

# tests/unit/test_financial_researcher.py
import pytest
from src.agents.research.financial_researcher import FinancialResearcher

class TestFinancialResearcher:
    @pytest.fixture
    def agent(self):
        return FinancialResearcher()
    
    def test_financial_market_analysis(self, agent):
        state = {
            "research_topic": "cryptocurrency market trends",
            "messages": []
        }
        result = agent.process(state)
        assert "financial_findings" in result.update
        assert "market_analysis" in result.update["financial_findings"]
    
    def test_economic_research(self, agent):
        state = {
            "research_topic": "inflation impact on markets",
            "messages": []
        }
        result = agent.process(state)
        assert "economic_indicators" in result.update["financial_findings"]
```

### 7.2 Integration Test Examples

```python
# tests/integration/test_team_coordination.py
import pytest
from src.main import build_research_team_graph

class TestTeamCoordination:
    @pytest.mark.asyncio
    async def test_research_team_flow(self):
        graph = build_research_team_graph()
        
        state = {
            "research_topic": "AI in healthcare and fintech",
            "research_status": "pending",
            "messages": []
        }
        
        result = await graph.ainvoke(state)
        
        assert result["research_status"] == "completed"
        assert "medical_findings" in result
        assert "financial_findings" in result
        # Both researchers work independently on the same topic
        assert result["medical_findings"]["research_timestamp"]
        assert result["financial_findings"]["analysis_timestamp"]
```

### 7.3 End-to-End Test Examples

```python
# tests/e2e/test_full_workflow.py
import pytest
import os
from src.main import build_main_graph

class TestFullWorkflow:
    @pytest.mark.asyncio
    async def test_complete_research_to_report(self):
        graph = build_main_graph()
        
        initial_state = {
            "messages": ["Research blockchain applications in healthcare and finance"],
            "task_description": "Blockchain comprehensive report",
            "research_state": {"research_status": "pending"},
            "reporting_state": {"report_status": "pending"}
        }
        
        result = await graph.ainvoke(initial_state)
        
        # Verify complete workflow
        assert result["final_output"]["document_path"]
        assert os.path.exists(result["final_output"]["document_path"])
        assert len(result["final_output"]["summary"]) > 100
```

## 8. Configuration and Environment

### 8.1 Environment Variables

```bash
# .env
OPENAI_API_KEY=your-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multiagent-research
```

### 8.2 Configuration File

```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    
    # Model Configuration
    supervisor_model: str = "gpt-4"
    researcher_model: str = "gpt-4"
    reporter_model: str = "gpt-3.5-turbo"
    
    # System Configuration
    max_retries: int = 3
    timeout_seconds: int = 300
    recursion_limit: int = 25
    
    # Output Configuration
    output_directory: str = "./outputs"
    document_template: str = "research_template.docx"
    
    class Config:
        env_file = ".env"
```

## 9. Deployment Considerations

### 9.1 Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
```

### 9.2 Requirements File

```txt
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
arxiv>=2.0.0
python-docx>=1.0.0
reportlab>=4.0.0
pydantic>=2.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
python-dotenv>=1.0.0
```

## 10. Monitoring and Observability

### 10.1 Logging Configuration

```python
import logging
from pythonjsonlogger import jsonlogger

def setup_logging():
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[logHandler])
```

### 10.2 Metrics Collection

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AgentMetrics:
    agent_name: str
    start_time: datetime
    end_time: datetime
    tokens_used: int
    success: bool
    error_message: str = None
```

## 11. Future Enhancements

1. **Dynamic Team Scaling**: Add/remove agents based on workload
2. **Advanced Memory Management**: Implement vector stores for long-term memory
3. **Multi-Language Support**: Extend document generation to multiple languages
4. **Real-time Collaboration**: Enable human-in-the-loop interactions
5. **Advanced Analytics**: Build dashboards for system performance monitoring

## 12. Success Criteria

- [ ] All agents successfully initialize and connect
- [ ] Supervisor correctly routes tasks between teams
- [ ] Research team produces comprehensive findings
- [ ] Reporting team generates valid PDF/DOCX documents
- [ ] System handles errors gracefully with retry logic
- [ ] All unit tests pass with >90% coverage
- [ ] Integration tests verify team coordination
- [ ] E2E tests confirm full workflow completion
- [ ] Performance: Complete research-to-report in <5 minutes
- [ ] Scalability: Handle multiple concurrent requests
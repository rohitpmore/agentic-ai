# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a learning repository for agentic AI, containing course assignments, tutorials, and research implementations. The main focus is on multi-agent systems, LLM-based agents, and related AI architectures.

## Project Structure

### Assignment 4 - Multi-Agent System
The main project (`assignments/assignment-4/`) implements a sophisticated multi-agent system with:
- **Supervisor Node**: Query classification and routing
- **RAG Agent**: USA Economy questions using FAISS vector search
- **LLM Agent**: General knowledge questions
- **Web Crawler Agent**: Real-time information via Tavily API
- **Validation Agent**: Quality control with feedback loops

### Assignment 5 - AI Travel Agent & Expense Planner
The travel planning project (`assignments/assignment-5/`) implements an intelligent travel agent with:
- **LangGraph StateGraph Architecture**: Workflow orchestration with specialized agent nodes
- **Agent-Tool Separation**: Agents think and reason, LangGraph tools perform specific tasks
- **Weather Agent**: Climate analysis and recommendations using OpenWeatherMap
- **Attraction Agent**: Points of interest discovery via Foursquare API
- **Hotel Agent**: Accommodation suggestions and pricing
- **Itinerary Agent**: Day-by-day trip planning with LangGraph tool integration
- **Parallel Workflow**: Multiple agents working simultaneously via LangGraph StateGraph
- **Free API Integration**: Cost-effective solution using free tier APIs

### Tutorials
The `tutorials/` directory contains learning implementations for:
- LangChain RAG systems
- SQL question-answering
- Text summarization
- LangGraph workflows

## Common Commands

### Assignment 4 Development

```bash
# Navigate to assignment directory
cd assignments/assignment-4/

# Install dependencies
pip install -r requirements.txt

# Setup environment (requires .env with GEMINI_API_KEY and TAVILY_API_KEY)
python main.py --setup

# Run interactive mode
python main.py

# Run single query
python main.py -q "What is the US GDP structure?"

# Run system tests
python main.py --test

# Run unit tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=multi_agent_system --cov-report=html
```

### Assignment 5 Development

```bash
# Navigate to assignment directory
cd assignments/assignment-5/

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment (requires .env with free API keys)
python main.py --setup

# Test API connections
python test_connections.py

# Run interactive travel planning
python main.py

# Run with specific travel query
python main.py -q "Plan a 3-day trip to Paris with $1000 budget"

# Run system tests
python main.py --test

# Run unit tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=travel_agent_system --cov-report=html
```

### Tutorial Development

```bash
# For each tutorial, navigate to directory and activate venv
cd tutorials/langchain/rag/
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook code.ipynb
```

## Architecture Overview

### Multi-Agent System Design
- **Modular Architecture**: Each agent is a self-contained class with clear interfaces
- **LangGraph Orchestration**: State-based workflow with conditional routing
- **Feedback Loops**: Failed validations trigger re-processing through different agents
- **Configuration Management**: Environment-based config with `config.py`
- **Vector Storage**: FAISS for document embeddings and retrieval
- **External APIs**: Google Gemini for LLM, Tavily for web search

### Travel Agent System Design (Assignment 5)
- **LangGraph StateGraph Architecture**: Professional workflow orchestration with node-based execution
- **Agent-Tool Separation**: Clear distinction between reasoning agents and LangGraph tools
- **Parallel Execution**: Multiple agent nodes working simultaneously via StateGraph
- **Rate Limiting**: Built-in API rate limiting with exponential backoff
- **Free API Strategy**: Cost-effective integration using free tier APIs
- **Error Handling**: Comprehensive error handling with graceful degradation
- **LangGraph Migration**: Complete migration from ThreadPoolExecutor to StateGraph

### Key Components

**Assignment 4:**
- `multi_agent_system/core/workflow.py`: Main orchestration class
- `multi_agent_system/agents/`: Individual agent implementations
- `multi_agent_system/utils/`: Shared utilities (vector store, web search, embeddings)
- `config.py`: Centralized configuration management
- `main.py`: CLI interface

**Assignment 5:**
- `travel_agent_system/core/langgraph_workflow.py`: LangGraph StateGraph orchestration
- `travel_agent_system/core/graph_state.py`: TypedDict state management
- `travel_agent_system/core/nodes.py`: Workflow node functions
- `travel_agent_system/agents/`: Specialized travel agents (Weather, Attraction, Hotel, Itinerary)
- `travel_agent_system/tools/langgraph_tools.py`: LangGraph integrated tools
- `travel_agent_system/utils/api_clients.py`: API client utilities with rate limiting
- `test_connections.py`: API connectivity testing
- `config.py`: Free API configuration management

### Testing Strategy
- Unit tests for individual agents (`tests/test_agents.py`)
- Utility function tests (`tests/test_utils.py`)
- Integration tests for full workflow (`tests/test_workflow.py`)
- Mocked external APIs to avoid real API calls during testing
- 90%+ code coverage target

## Development Notes

### Environment Setup
Each project requires specific API keys:
- Assignment 4: `GEMINI_API_KEY` and `TAVILY_API_KEY`
- Assignment 5: `OPENWEATHER_API_KEY`, `FOURSQUARE_API_KEY`, `EXCHANGERATE_API_KEY` (all free tier)
- Tutorials may require `OPENAI_API_KEY` or other service keys

### Virtual Environments
Each tutorial maintains its own virtual environment in `venv/` directories. Always activate the appropriate environment before development.

### Configuration Pattern
Projects use `.env` files for environment variables with `python-dotenv` for loading. Configuration is centralized in `config.py` files using dataclasses.

### Vector Store Management
FAISS databases are cached and reused. Use `--force-recreate` flag to rebuild vector stores when document sources change.

### Assignment 5 Development Best Practices
- **Stage-focused Development**: Complete one stage fully before moving to next ("laser focus" approach)
- **Free API Preference**: Prioritize free tier APIs with proper rate limiting implementation
- **Import Chain Management**: Create placeholder files early to avoid circular dependencies
- **Agent-Tool Separation**: Maintain clear boundaries between reasoning agents and utility tools
- **Placeholder Development**: Use minimal placeholder implementations to resolve import issues during development
- **API Rate Limiting**: Implement exponential backoff and respect free tier limits
- **Connection Testing**: Always test API connectivity before full implementation

## Advanced Development Methodologies

### Progressive Stage-Based Development
When undertaking complex migrations or major architectural changes, follow this structured approach:

1. **6-Stage Methodology**: Break complex work into well-defined stages (Foundation → Migration → Integration → Advanced Features → CLI → Testing)
2. **Real-Time Progress Tracking**: Use checkboxes (✅/⏳/❌) to track completion status
3. **Stage Validation**: Complete and test each stage before proceeding to next
4. **Deliverable-Focused**: Define clear deliverables and success criteria for each stage
5. **Timeline Management**: Estimate 17-23 hours for major architectural migrations

### Test-Driven Migration Strategy
Implement comprehensive testing throughout development:

1. **Multi-Level Testing**: Unit → Integration → Performance → Quality Assurance
2. **Verification Tasks**: Define specific testing tasks for each development stage
3. **Coverage Targets**: Maintain 90%+ code coverage throughout migration
4. **Performance Benchmarking**: Compare against original implementation metrics
5. **API Failure Testing**: Test all external API failure modes and edge cases
6. **Regression Prevention**: Create performance regression tests

### Documentation-Driven Development
Maintain documentation consistency throughout development:

1. **Parallel Documentation**: Update docs alongside code changes
2. **Legacy Reference Removal**: Systematically remove outdated information
3. **Architecture Visualization**: Generate workflow diagrams using built-in tools (`get_graph().draw_mermaid_png()`)
4. **Repository-Wide Consistency**: Audit all files for legacy references
5. **User Experience Documentation**: Update CLI usage examples and troubleshooting

### Risk Mitigation Patterns
Implement safeguards for complex changes:

1. **Stage-by-Stage Validation**: Complete testing before moving to next stage
2. **Functionality Preservation**: Maintain all existing features during migration
3. **Performance Monitoring**: Ensure no performance degradation
4. **Rollback Capability**: Keep old implementation until new one is fully validated
5. **Comprehensive Testing**: Test every component at every stage

### Professional State Management
For LangGraph and similar state-based systems:

1. **TypedDict Patterns**: Use TypedDict for LangGraph compatibility and type safety
2. **State Reducers**: Implement reducer functions for parallel state updates
3. **State Validation**: Add validation and error handling for state transitions
4. **Workflow Persistence**: Implement checkpointing and state serialization
5. **Conditional Routing**: Use routing functions that inspect state and return node names

### Advanced Error Recovery
Implement robust error handling:

1. **Retry Mechanisms**: Exponential backoff for API calls and external services
2. **Fallback Strategies**: Graceful degradation when APIs fail
3. **Error Recovery Paths**: Multiple recovery strategies for different failure modes
4. **Comprehensive Reporting**: Detailed error reporting with actionable information
5. **Quality Validation Gates**: Implement quality checks throughout workflow

### Migration Success Criteria
For any major architectural change, ensure:

- ✅ **Complete Legacy Removal**: Zero references to old implementation
- ✅ **Feature Parity**: All existing functionality preserved
- ✅ **Performance Maintained**: No degradation in execution speed
- ✅ **Best Practices**: Proper framework usage throughout
- ✅ **Comprehensive Testing**: 90%+ code coverage with all scenarios tested
- ✅ **Clean Architecture**: Professional implementation patterns
- ✅ **Documentation Consistency**: All docs reflect new implementation
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
- **Agent-Tool Separation**: Agents think and reason, tools perform specific tasks
- **Weather Agent**: Climate analysis and recommendations using OpenWeatherMap
- **Attraction Agent**: Points of interest discovery via Foursquare API
- **Hotel Agent**: Accommodation suggestions and pricing
- **Itinerary Agent**: Day-by-day trip planning with cost optimization
- **Parallel Workflow**: Multiple agents working simultaneously via LangGraph
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
- **Agent-Tool Separation**: Clear distinction between reasoning agents and utility tools
- **Parallel Execution**: Multiple agents working simultaneously for efficiency
- **Rate Limiting**: Built-in API rate limiting with exponential backoff
- **Free API Strategy**: Cost-effective integration using free tier APIs
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Stage-based Development**: 6-stage implementation approach with clear deliverables

### Key Components

**Assignment 4:**
- `multi_agent_system/core/workflow.py`: Main orchestration class
- `multi_agent_system/agents/`: Individual agent implementations
- `multi_agent_system/utils/`: Shared utilities (vector store, web search, embeddings)
- `config.py`: Centralized configuration management
- `main.py`: CLI interface

**Assignment 5:**
- `travel_agent_system/core/workflow.py`: Travel planning orchestration
- `travel_agent_system/agents/`: Specialized travel agents (Weather, Attraction, Hotel, Itinerary)
- `travel_agent_system/tools/`: Utility tools (CostCalculator, CurrencyConverter)
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
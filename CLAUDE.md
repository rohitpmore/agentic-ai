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

### Key Components
- `multi_agent_system/core/workflow.py`: Main orchestration class
- `multi_agent_system/agents/`: Individual agent implementations
- `multi_agent_system/utils/`: Shared utilities (vector store, web search, embeddings)
- `config.py`: Centralized configuration management
- `main.py`: CLI interface

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
- Tutorials may require `OPENAI_API_KEY` or other service keys

### Virtual Environments
Each tutorial maintains its own virtual environment in `venv/` directories. Always activate the appropriate environment before development.

### Configuration Pattern
Projects use `.env` files for environment variables with `python-dotenv` for loading. Configuration is centralized in `config.py` files using dataclasses.

### Vector Store Management
FAISS databases are cached and reused. Use `--force-recreate` flag to rebuild vector stores when document sources change.
# LangGraph Multi-Agent Hierarchical Workflow System

## Overview

A sophisticated multi-agent system built on LangGraph that orchestrates research and reporting workflows through hierarchical team coordination. The system features a supervisor agent that manages specialized research and reporting teams, with each team containing domain-specific agents.

## Architecture

### High-Level Components

- **Main Supervisor Agent**: Orchestrates workflow and manages team handoffs
- **Research Team**: Medical and Financial research agents
- **Reporting Team**: Document creation and summarization agents
- **State Management**: TypedDict-based state schemas for type safety
- **Handoff Protocol**: Standardized agent communication system

### Key Features

- **Professional State Management**: TypedDict schemas for LangGraph compatibility
- **Hierarchical Coordination**: Team-based agent organization
- **Robust Error Handling**: Comprehensive error recovery mechanisms
- **Performance Monitoring**: Built-in metrics and logging
- **Configurable Architecture**: Environment-based configuration

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-multiagent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPERVISOR_MODEL`: Model for supervisor agent (default: gpt-4)
- `RESEARCHER_MODEL`: Model for research agents (default: gpt-4)
- `REPORTER_MODEL`: Model for reporting agents (default: gpt-3.5-turbo)
- `LOG_LEVEL`: Logging level (default: INFO)
- `OUTPUT_DIRECTORY`: Directory for generated documents

### Model Configuration

The system supports different models for different agent types:
- Supervisor agents use precise models for decision-making
- Research agents use capable models for analysis
- Reporting agents use efficient models for document generation

## Usage

### Basic Usage

```python
from config.settings import Settings
from src.agents.base_agent import BaseAgent

# Initialize configuration
settings = Settings()

# Create and run workflow
# (Implementation continues in Stage 2)
```

### Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
```

## Development

### Project Structure

```
langgraph-multiagent/
├── src/
│   ├── agents/          # Agent implementations
│   ├── state/           # State schemas
│   ├── tools/           # Tool implementations
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── config/              # Configuration management
├── requirements.txt     # Dependencies
└── .env.example        # Environment template
```

### Development Guidelines

1. **Code Quality**: Follow PEP 8 standards
2. **Type Safety**: Use type hints throughout
3. **Testing**: Maintain >90% test coverage
4. **Documentation**: Update docs with code changes
5. **Error Handling**: Implement comprehensive error recovery

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement required abstract methods
3. Add proper type hints
4. Include comprehensive tests
5. Update documentation

## Stage 1 Implementation Status

### ✅ Completed Components

- [x] Project structure and organization
- [x] BaseAgent abstract class with validation
- [x] State schemas with TypedDict definitions
- [x] Configuration management with Pydantic
- [x] Handoff protocol utilities
- [x] Logging infrastructure
- [x] Testing framework setup
- [x] Environment configuration

### Next Steps

Continue to Stage 2: Core Agent Development
- Implement supervisor agents
- Create routing logic
- Add error handling mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure code quality standards
5. Submit pull request

## License

This project is part of the Agentic AI learning series.

---

*This foundation provides the solid base for building sophisticated multi-agent workflows with professional standards for scalability, maintainability, and reliability.*
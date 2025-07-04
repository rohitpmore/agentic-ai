# LangGraph Multi-Agent Project - CLAUDE.md

## Project Context

This is Stage 1 of the LangGraph Multi-Agent Hierarchical Workflow System implementation. The project builds a sophisticated multi-agent research and reporting system using LangGraph as the orchestration framework.

### Current Implementation Status

**Stage 1: Foundation (COMPLETED)**
- ✅ Project structure established
- ✅ BaseAgent abstract class implemented
- ✅ State schemas with TypedDict for LangGraph compatibility
- ✅ Configuration management with Pydantic
- ✅ Handoff protocol utilities
- ✅ Logging infrastructure
- ✅ Testing framework setup

### Project-Specific Guidelines

#### Architecture Principles
1. **LangGraph Integration**: All state management uses TypedDict for compatibility
2. **Hierarchical Design**: Team-based agent organization with supervisors
3. **Professional Standards**: Type safety, error handling, comprehensive testing
4. **Modular Design**: Clear separation between agents, tools, and utilities

#### Development Standards
1. **Type Safety**: All functions and classes must have proper type hints
2. **Error Handling**: Comprehensive error recovery with graceful degradation
3. **Testing**: Maintain 90%+ test coverage with unit/integration/e2e tests
4. **Documentation**: Keep README and code docs synchronized

#### Stage Development Process
- **Sequential Implementation**: Complete one stage fully before moving to next
- **Validation Gates**: Test each component before proceeding
- **Progressive Complexity**: Build from foundation to advanced features
- **Professional Quality**: Production-ready code standards throughout

### Technology Stack

#### Core Dependencies
- **LangGraph**: Workflow orchestration framework
- **LangChain**: LLM integration and tooling
- **Pydantic**: Configuration and data validation
- **TypedDict**: State schema definitions
- **Pytest**: Testing framework

#### Development Tools
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest-cov**: Coverage reporting

### Testing Strategy

#### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **E2E Tests**: Full workflow testing
- **Performance Tests**: Load and response time testing

#### Coverage Requirements
- Minimum 90% code coverage
- All public methods must be tested
- Error conditions must be tested
- State transitions must be validated

### Configuration Management

#### Environment Variables
- API keys stored securely in .env
- Model configuration per agent type
- Performance and logging configuration
- Development vs production settings

#### Settings Hierarchy
1. Default values in Settings class
2. Environment variable overrides
3. Command-line argument overrides (future)
4. Configuration file overrides (future)

### Error Handling Strategy

#### Error Categories
1. **Validation Errors**: Input state validation failures
2. **API Errors**: External service failures
3. **Processing Errors**: Agent logic failures
4. **System Errors**: Infrastructure failures

#### Recovery Mechanisms
1. **Graceful Degradation**: Continue with partial data
2. **Retry Logic**: Exponential backoff for transient errors
3. **Error Handoffs**: Route errors to specialized handlers
4. **State Preservation**: Maintain workflow state during errors

### Performance Requirements

#### Response Times
- Agent processing: < 30 seconds
- State transitions: < 1 second
- Error handling: < 5 seconds
- Workflow completion: < 5 minutes

#### Resource Usage
- Memory: < 1GB for standard workflows
- CPU: Efficient async processing
- Storage: Minimal local storage requirements
- Network: Optimized API call patterns

### Common Commands

#### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

#### Testing
```bash
# Unit tests only
pytest tests/unit/ -m unit

# Integration tests
pytest tests/integration/ -m integration

# End-to-end tests
pytest tests/e2e/ -m e2e

# Performance tests
pytest tests/ -m slow
```

### Next Stage Preparation

When ready for Stage 2:
1. Verify all Stage 1 tests pass
2. Confirm configuration loads correctly
3. Validate base classes work as expected
4. Review and update documentation
5. Create Stage 2 branch for development

### Key Files and Locations

#### Core Implementation
- `src/agents/base_agent.py`: Abstract base class for all agents
- `src/state/schemas.py`: TypedDict state definitions
- `config/settings.py`: Pydantic configuration management
- `src/utils/handoff.py`: Agent communication protocol

#### Testing
- `tests/unit/test_base_agent.py`: BaseAgent test suite
- `pytest.ini`: Testing configuration
- Coverage reports in `htmlcov/`

#### Configuration
- `.env.example`: Environment variable template
- `requirements.txt`: Python dependencies
- `.gitignore`: Version control exclusions

---

*This project follows the progressive stage-based development methodology with emphasis on professional standards, comprehensive testing, and production-ready architecture.*
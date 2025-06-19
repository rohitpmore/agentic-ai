# AI Travel Agent & Expense Planner - Implementation Plan

## Overview
AI Travel Agent & Expense Planner for trip planning to any city worldwide with real-time data integration using free APIs and OOP modular design.

## Core Requirements
- Real-time weather information
- Top attractions and activities
- Hotel cost calculation (per day × total days)
- Currency conversion to user's native currency
- Complete itinerary generation
- Total expense calculation
- Generate summary of entire output

## Architectural Improvements Based on Feedback

### Key Changes:
1. **Agent vs Tool Redesign**: Convert `CostCalculator` and `CurrencyAgent` from agents to tools
2. **Parallel Workflow**: Replace sequential execution with concurrent data gathering
3. **Better Separation**: Agents think and plan, tools perform specific tasks
4. **Improved Performance**: Parallel API calls for independent data sources

## Implementation Stages

### Stage 1: Project Setup & Core Structure ✅
**Duration**: 1-2 hours
**Tasks**:
- ✅ Create modular OOP architecture following Assignment 4 pattern
- ✅ Setup `config.py` with dataclass-based configuration
- ✅ Create base project structure with packages and modules
- ✅ Setup `requirements.txt` with free API dependencies
- ✅ Initialize testing framework structure

**Deliverables**:
- ✅ Project folder structure
- ✅ Configuration management
- ✅ Base classes and interfaces
- ✅ Testing framework setup

### Stage 2: Free API Integration Setup
**Duration**: 2-3 hours
**Free APIs to Use**:
- **Weather**: OpenWeatherMap API (free tier - 1000 calls/day)
- **Places/Attractions**: Foursquare Places API (free tier) or OpenTripMap
- **Hotels**: Use combination of free APIs or create estimation logic
- **Currency**: ExchangeRate-API or Fixer.io (free tiers)
- **Transportation**: OpenStreetMap + public transit APIs
- **LLM**: Google Gemini (already configured in Assignment 4)

**Tasks**:
- Research and setup API keys for free services
- Create API client utilities
- Implement rate limiting and error handling
- Test API connections

**Deliverables**:
- API client utilities
- Configuration for all free APIs
- Connection testing utilities

### Stage 3: Agent & Tool Implementation
**Duration**: 4-5 hours
**Reasoning Agents** (Decision-making entities):

1. **WeatherAgent**
   - Get current weather
   - Get weather forecast (5-7 days)
   - Analyze weather impact on travel plans

2. **AttractionAgent**
   - Search attractions using Places API
   - Search restaurants 
   - Search activities
   - Search transportation options
   - Prioritize based on user preferences

3. **HotelAgent**
   - Search hotels (limited free API or estimation logic)
   - Estimate hotel costs
   - Handle budget ranges
   - Recommend based on location and budget

4. **ItineraryAgent**
   - Generate day plans based on all collected data
   - Create full itinerary with timeline
   - Balance activities with travel time
   - Use tools for cost calculations and currency conversion

**Utility Tools** (Task-specific operations):

1. **CostCalculator Tool**
   - Add operation for total costs
   - Multiply operation for daily calculations
   - Calculate total cost
   - Calculate daily budget

2. **CurrencyConverter Tool**
   - Get exchange rates
   - Convert currency to user's native currency
   - Handle multiple currency conversions

**Deliverables**:
- Reasoning agent classes with decision-making logic
- Utility tools for specific operations
- Clear separation between thinking and doing
- Error handling and validation

### Stage 4: Parallel Workflow Orchestration
**Duration**: 2-3 hours
**Tasks**:
- Implement `TravelPlanState` for state management
- Create `TravelPlannerWorkflow` using LangGraph with parallel execution
- Design concurrent API call flow for independent data sources
- Implement data aggregation and merging logic
- Add error handling and retry mechanisms
- Configure LangGraph branching and joining

**Improved Workflow Flow**:
```
User Input
    ↓
{Dispatch to Parallel Agents}
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Weather Agent  │ Attraction Agent│  Hotel Agent    │
│  (concurrent)   │   (concurrent)  │  (concurrent)   │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
{Data Aggregation & Merge}
    ↓
Itinerary Agent
    ↓
{Uses Tools: CostCalculator + CurrencyConverter}
    ↓
Trip Summary Generation
    ↓
Final Output
```

**Deliverables**:
- Main workflow orchestration class
- State management system
- API call coordination
- Error handling system

### Stage 5: Integration & Summary Generation
**Duration**: 2-3 hours
**Tasks**:

1. **Data Integration**
   - Merge parallel agent results
   - Handle partial failures gracefully
   - Validate data consistency

2. **ItineraryAgent Enhancement**
   - Process aggregated data from all agents
   - Generate optimized day plans
   - Use CostCalculator tool for budget planning
   - Use CurrencyConverter tool for local pricing

3. **Trip Summary Generation**
   - Aggregate all processed data
   - Format comprehensive cost breakdown
   - Create detailed travel plan
   - Generate executive summary

**Deliverables**:
- Integrated data processing pipeline
- Enhanced itinerary generation with tool usage
- Complete trip summary with all details
- Cost breakdown in user's preferred currency
- Formatted output generation

### Stage 6: CLI Interface & Testing
**Duration**: 2-3 hours
**Tasks**:

1. **CLI Interface** (following Assignment 4 pattern)
   - Interactive mode for trip planning
   - Single query mode
   - Setup mode for API testing
   - Test mode for system validation

2. **Testing Suite**
   - Unit tests for each agent class
   - Integration tests for workflow
   - Mocked API responses for testing
   - Coverage reporting

3. **Final Integration**
   - End-to-end testing
   - Error case handling
   - Performance optimization
   - Documentation updates

**Deliverables**:
- Complete CLI interface
- Comprehensive test suite
- System integration
- Final documentation

## Technical Architecture

### Updated Project Structure
```
assignments/assignment-5/
├── config.py                 # Configuration management
├── main.py                   # CLI interface
├── requirements.txt          # Dependencies
├── travel_agent_system/      # Main package
│   ├── __init__.py
│   ├── agents/               # Reasoning agent classes
│   │   ├── __init__.py
│   │   ├── weather_agent.py
│   │   ├── attraction_agent.py
│   │   ├── hotel_agent.py
│   │   └── itinerary_agent.py
│   ├── tools/                # NEW - Utility tools
│   │   ├── __init__.py
│   │   ├── cost_calculator.py
│   │   └── currency_converter.py
│   ├── core/                 # Core workflow logic
│   │   ├── __init__.py
│   │   ├── state.py          # State management
│   │   └── workflow.py       # Parallel orchestration
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── api_clients.py    # External API clients
│       └── formatters.py     # Output formatting
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py         # NEW - Tool tests
│   ├── test_utils.py
│   └── test_workflow.py
├── instructions/             # Documentation
│   ├── implementation_plan.md
│   └── question.txt
└── .env.example             # Environment variables template
```

### Key Design Principles
1. **Agent-Tool Separation**: Agents think and plan, tools perform specific tasks
2. **Parallel Execution**: Concurrent API calls for independent data sources
3. **OOP Modular Design**: Each component is a self-contained class
4. **Free API Usage**: All external APIs are free tier
5. **Error Handling**: Robust error handling for API failures and partial data
6. **State Management**: Clear state management with data merging capabilities
7. **Testing**: Comprehensive testing with mocked APIs and parallel execution
8. **Configuration**: Environment-based configuration management
9. **Performance**: Optimized for speed through concurrent operations

## Expected Timeline
- **Total Duration**: 12-18 hours
- **Stage 1-2**: Foundation (3-5 hours)
- **Stage 3-4**: Core Implementation (6-8 hours)  
- **Stage 5-6**: Integration & Testing (3-5 hours)
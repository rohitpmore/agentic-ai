"""
Multi-Agent System with Supervisor Node

A sophisticated multi-agent system that routes queries to specialized agents:
- Supervisor: Orchestrates and classifies queries
- RAG Agent: Handles USA Economy questions using vector search
- LLM Agent: Handles general knowledge questions
- Web Crawler Agent: Handles real-time/current events using web search
- Validation Agent: Validates responses and triggers feedback loops
"""

__version__ = "1.0.0"
__author__ = "Agentic AI Assignment 4"

from .core.workflow import MultiAgentWorkflow
from .core.state import AgentState

__all__ = ["MultiAgentWorkflow", "AgentState"] 
"""
Travel Agent System - Main Package

A modular AI Travel Agent & Expense Planner system using free APIs
and LangGraph workflow orchestration.
"""

from .core.workflow import TravelPlannerWorkflow

__version__ = "1.0.0"
__all__ = ["TravelPlannerWorkflow"]
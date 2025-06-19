"""
Travel Agent System - Core Package

Core workflow logic and state management.
"""

from .state import TravelPlanState
# Note: Workflow import commented until dependency resolution
# from .workflow import TravelPlannerWorkflow

__all__ = ["TravelPlanState"]
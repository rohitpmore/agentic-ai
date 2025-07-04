#!/usr/bin/env python3
"""
Entry point for the LangGraph Multi-Agent CLI
"""

import sys
import os

# Add the current directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
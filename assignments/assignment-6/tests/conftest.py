"""
Test Configuration and Fixtures
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from config.settings import Settings
from src.main import MultiAgentWorkflow


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Create test settings"""
    return Settings(
        openai_api_key="test-api-key",
        supervisor_model="gpt-4",
        researcher_model="gpt-4",
        reporter_model="gpt-3.5-turbo",
        max_retries=3,
        timeout_seconds=300,
        output_directory="./test_outputs",
        debug_mode=True,
        test_mode=True
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_arxiv_results():
    """Mock arXiv API results"""
    return [
        {
            "title": "Test Medical Paper 1",
            "authors": ["Dr. Test Author"],
            "abstract": "This is a test abstract for medical research paper.",
            "url": "http://arxiv.org/abs/test1",
            "relevance_score": 0.85
        },
        {
            "title": "Test Financial Paper 1",
            "authors": ["Prof. Finance Expert"],
            "abstract": "This is a test abstract for financial research paper.",
            "url": "http://arxiv.org/abs/test2",
            "relevance_score": 0.78
        }
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    response = Mock()
    response.content = """
    KEY FINDINGS:
    - Test finding 1
    - Test finding 2
    - Test finding 3
    
    CLINICAL IMPLICATIONS:
    - Test implication 1
    - Test implication 2
    
    RESEARCH GAPS:
    - Test gap 1
    
    SUMMARY:
    This is a test summary of the research findings.
    """
    return response


@pytest.fixture
def sample_research_state():
    """Sample research state for testing"""
    return {
        "research_topic": "AI applications in healthcare and finance",
        "research_status": "pending",
        "medical_findings": {},
        "financial_findings": {},
        "research_metadata": {}
    }


@pytest.fixture
def sample_reporting_state():
    """Sample reporting state for testing"""
    return {
        "research_data": {
            "medical_findings": {"key_findings": ["Medical finding 1"]},
            "financial_findings": {"key_findings": ["Financial finding 1"]}
        },
        "report_status": "pending",
        "document_path": "",
        "summary": "",
        "report_metadata": {}
    }


@pytest.fixture
def sample_supervisor_state():
    """Sample supervisor state for testing"""
    return {
        "task_description": "Test research task",
        "current_team": "research",
        "research_state": {
            "research_topic": "Test topic",
            "research_status": "pending"
        },
        "reporting_state": {
            "report_status": "pending"
        },
        "messages": [],
        "system_metrics": {}
    }
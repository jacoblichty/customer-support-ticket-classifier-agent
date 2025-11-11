"""
Test configuration and fixtures for the ticket classifier tests.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock
import sys
from pathlib import Path

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ticket_classifier.models import SupportTicket
from ticket_classifier.classifier import TicketClassifier
from ticket_classifier.agent import TicketClassifierAgent
from ticket_classifier.config import TestingSettings


@pytest.fixture
def sample_ticket():
    """Create a sample support ticket for testing."""
    return SupportTicket(
        ticket_id="TEST001",
        subject="Test Issue",
        content="This is a test ticket for unit testing purposes.",
        customer_email="test@example.com",
        priority="medium"
    )


@pytest.fixture
def sample_tickets():
    """Create multiple sample tickets for batch testing."""
    return [
        SupportTicket("T001", "Login Problem", "Can't access my account", "user1@example.com"),
        SupportTicket("T002", "Billing Issue", "Charged twice", "user2@example.com"),
        SupportTicket("T003", "Feature Request", "Add dark mode", "user3@example.com"),
    ]


@pytest.fixture
def test_settings():
    """Get test settings."""
    return TestingSettings()


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "category": "technical_issue",
        "confidence_score": 0.85,
        "reasoning": "Keywords indicate technical problems"
    }


@pytest_asyncio.fixture
async def mock_openai_client(mock_openai_response):
    """Create a mocked OpenAI client."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = f'{{"category": "{mock_openai_response["category"]}", "confidence_score": {mock_openai_response["confidence_score"]}, "reasoning": "{mock_openai_response["reasoning"]}"}}'
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def classifier_with_mock_client(test_settings, mock_openai_client):
    """Create classifier with mocked OpenAI client."""
    classifier = TicketClassifier(api_key="test-key", settings=test_settings)
    classifier.client = mock_openai_client
    return classifier


@pytest_asyncio.fixture
async def agent_with_mock_classifier(test_settings, classifier_with_mock_client):
    """Create agent with mocked classifier."""
    agent = TicketClassifierAgent(settings=test_settings)
    agent.classifier = classifier_with_mock_client
    return agent
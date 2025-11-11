"""
Tests for the FastAPI application.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ticket_classifier.api import create_app
from ticket_classifier.models import TicketRequest, BatchTicketRequest
from ticket_classifier.config import UnitTestSettings


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    import time
    from ticket_classifier.api import register_routes
    
    # Create app without lifespan for testing
    settings = UnitTestSettings()
    app = FastAPI(
        title=settings.app_name,
        description="Test app",
        version=settings.app_version,
    )
    
    # Add the same middleware as in the real app
    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Register routes
    register_routes(app)
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_agent():
    """Create mock agent for testing."""
    agent = Mock()
    agent.get_health_status.return_value = {
        "status": "healthy",
        "openai_available": True,
        "total_processed": 0,
        "uptime_seconds": 123.45,
        "settings": {
            "max_batch_size": 100,
            "max_concurrent_requests": 10,
            "openai_model": "gpt-4"
        }
    }
    agent.get_statistics.return_value = {
        "total_processed": 5,
        "category_distribution": {"technical_issue": 3, "billing_inquiry": 2},
        "average_confidence": 0.85,
        "uptime_seconds": 123.45,
        "categories_available": ["technical_issue", "billing_inquiry"]
    }
    agent.classifier.categories = ["technical_issue", "billing_inquiry", "general_inquiry"]
    agent.get_recent_tickets.return_value = []
    return agent


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    @patch('ticket_classifier.api.agent')
    def test_health_endpoint_healthy(self, mock_agent_global, client, mock_agent):
        """Test health endpoint when healthy."""
        mock_agent_global.__bool__ = lambda x: True
        mock_agent_global.get_health_status = mock_agent.get_health_status
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "openai_available" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_endpoint_no_agent(self, client):
        """Test health endpoint when agent not initialized."""
        with patch('ticket_classifier.api.agent', None):
            response = client.get("/health")
        
        assert response.status_code == 503
        assert "Agent not initialized" in response.json()["detail"]
    
    @patch('ticket_classifier.api.agent')
    def test_classify_endpoint_success(self, mock_agent_global, client, mock_agent):
        """Test successful ticket classification."""
        # Mock the process_ticket method
        mock_ticket = Mock()
        mock_ticket.to_response.return_value = {
            "ticket_id": "T001",
            "subject": "Test Issue",
            "content": "Test content",
            "customer_email": "test@example.com",
            "priority": "medium",
            "category": "technical_issue",
            "confidence_score": 0.85,
            "reasoning": "Test reasoning",
            "created_at": "2025-11-11T10:00:00",
            "processed_at": "2025-11-11T10:00:01"
        }
        
        async def mock_process_ticket(ticket):
            return mock_ticket
        
        mock_agent.process_ticket = mock_process_ticket
        mock_agent_global.return_value = mock_agent
        
        request_data = {
            "ticket_id": "T001",
            "subject": "Test Issue",
            "content": "Test content",
            "customer_email": "test@example.com",
            "priority": "medium"
        }
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.post("/classify", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"] == "T001"
        assert data["category"] == "technical_issue"
    
    def test_classify_endpoint_invalid_data(self, client):
        """Test classification with invalid data."""
        request_data = {
            "ticket_id": "T001",
            # Missing required fields
        }
        
        response = client.post("/classify", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('ticket_classifier.api.agent')
    def test_classify_batch_endpoint_success(self, mock_agent_global, client, mock_agent):
        """Test successful batch classification."""
        # Mock the process_batch method
        mock_tickets = [Mock(), Mock()]
        for i, ticket in enumerate(mock_tickets):
            ticket.to_response.return_value = {
                "ticket_id": f"T00{i+1}",
                "subject": f"Test {i+1}",
                "content": f"Content {i+1}",
                "customer_email": f"test{i+1}@example.com",
                "priority": "medium",
                "category": "technical_issue",
                "confidence_score": 0.85,
                "reasoning": "Test reasoning",
                "created_at": "2025-11-11T10:00:00",
                "processed_at": "2025-11-11T10:00:01"
            }
        
        async def mock_process_batch(tickets):
            return mock_tickets
        
        mock_agent.process_batch = mock_process_batch
        mock_agent_global.return_value = mock_agent
        
        request_data = {
            "tickets": [
                {
                    "ticket_id": "T001",
                    "subject": "Test 1",
                    "content": "Content 1",
                    "customer_email": "test1@example.com"
                },
                {
                    "ticket_id": "T002",
                    "subject": "Test 2", 
                    "content": "Content 2",
                    "customer_email": "test2@example.com"
                }
            ]
        }
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.post("/classify/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["processed_tickets"]) == 2
        assert "statistics" in data
        assert "processing_time_seconds" in data
    
    def test_classify_batch_empty(self, client):
        """Test batch classification with empty list."""
        request_data = {"tickets": []}
        
        response = client.post("/classify/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('ticket_classifier.api.agent')
    def test_classify_batch_too_large(self, mock_agent_global, client, mock_agent):
        """Test batch classification exceeding size limit."""
        mock_agent_global.return_value = mock_agent
        
        # Create a batch that's too large
        tickets = []
        for i in range(101):  # Exceeds default max of 100
            tickets.append({
                "ticket_id": f"T{i:03d}",
                "subject": f"Test {i}",
                "content": f"Content {i}",
                "customer_email": f"test{i}@example.com"
            })
        
        request_data = {"tickets": tickets}
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.post("/classify/batch", json=request_data)
        
        assert response.status_code == 422  # Pydantic validation error
        response_data = response.json()
        assert response_data["detail"][0]["type"] == "too_long"
        assert "100 items" in response_data["detail"][0]["msg"]
    
    @patch('ticket_classifier.api.agent')
    def test_stats_endpoint(self, mock_agent_global, client, mock_agent):
        """Test statistics endpoint."""
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 5
        assert "category_distribution" in data
        assert "average_confidence" in data
    
    @patch('ticket_classifier.api.agent')
    def test_categories_endpoint(self, mock_agent_global, client, mock_agent):
        """Test categories endpoint."""
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/categories")
        
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "descriptions" in data
        assert "technical_issue" in data["categories"]
    
    @patch('ticket_classifier.api.agent')
    def test_recent_endpoint(self, mock_agent_global, client, mock_agent):
        """Test recent tickets endpoint."""
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/recent")
        
        assert response.status_code == 200
        data = response.json()
        assert "tickets" in data
        assert "count" in data
    
    @patch('ticket_classifier.api.agent')
    def test_recent_endpoint_with_limit(self, mock_agent_global, client, mock_agent):
        """Test recent tickets endpoint with custom limit."""
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/recent?limit=5")
        
        assert response.status_code == 200
        mock_agent.get_recent_tickets.assert_called_with(5)
    
    @patch('ticket_classifier.api.agent')
    def test_recent_endpoint_limit_too_high(self, mock_agent_global, client, mock_agent):
        """Test recent tickets endpoint with limit too high."""
        mock_agent_global.return_value = mock_agent
        
        with patch('ticket_classifier.api.agent', mock_agent):
            response = client.get("/recent?limit=200")
        
        assert response.status_code == 400
        assert "cannot exceed 100" in response.json()["detail"]


class TestAPIMiddleware:
    """Test cases for API middleware."""
    
    def test_process_time_header(self, client):
        """Test that process time header is added."""
        response = client.get("/")
        
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0
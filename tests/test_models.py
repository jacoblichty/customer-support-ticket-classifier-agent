"""
Tests for the SupportTicket and Pydantic models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from ticket_classifier.models import (
    SupportTicket, TicketRequest, TicketResponse,
    BatchTicketRequest, BatchTicketResponse, HealthResponse
)


class TestSupportTicket:
    """Test cases for SupportTicket class."""
    
    def test_create_ticket(self):
        """Test creating a support ticket."""
        ticket = SupportTicket(
            ticket_id="T001",
            subject="Test Issue",
            content="Test content",
            customer_email="test@example.com",
            priority="high"
        )
        
        assert ticket.ticket_id == "T001"
        assert ticket.subject == "Test Issue"
        assert ticket.content == "Test content"
        assert ticket.customer_email == "test@example.com"
        assert ticket.priority == "high"
        assert isinstance(ticket.created_at, datetime)
        assert ticket.category is None
        assert ticket.confidence_score is None
    
    def test_default_priority(self):
        """Test default priority is medium."""
        ticket = SupportTicket("T001", "Subject", "Content", "test@example.com")
        assert ticket.priority == "medium"
    
    def test_to_response(self, sample_ticket):
        """Test converting ticket to response model."""
        sample_ticket.category = "technical_issue"
        sample_ticket.confidence_score = 0.85
        sample_ticket.reasoning = "Test reasoning"
        
        response = sample_ticket.to_response()
        
        assert isinstance(response, TicketResponse)
        assert response.ticket_id == sample_ticket.ticket_id
        assert response.category == "technical_issue"
        assert response.confidence_score == 0.85
    
    def test_from_request(self):
        """Test creating ticket from request model."""
        request = TicketRequest(
            ticket_id="T001",
            subject="Test",
            content="Content",
            customer_email="test@example.com",
            priority="low"
        )
        
        ticket = SupportTicket.from_request(request)
        
        assert ticket.ticket_id == "T001"
        assert ticket.subject == "Test"
        assert ticket.priority == "low"
    
    def test_str_representation(self, sample_ticket):
        """Test string representation."""
        expected = "Ticket TEST001: Test Issue (Priority: medium)"
        assert str(sample_ticket) == expected
    
    def test_repr_representation(self, sample_ticket):
        """Test repr representation."""
        repr_str = repr(sample_ticket)
        assert "SupportTicket" in repr_str
        assert "TEST001" in repr_str


class TestTicketRequest:
    """Test cases for TicketRequest Pydantic model."""
    
    def test_valid_request(self):
        """Test creating a valid ticket request."""
        request = TicketRequest(
            ticket_id="T001",
            subject="Test Issue",
            content="This is a test",
            customer_email="test@example.com",
            priority="high"
        )
        
        assert request.ticket_id == "T001"
        assert request.subject == "Test Issue"
        assert request.priority == "high"
    
    def test_default_priority(self):
        """Test default priority is medium."""
        request = TicketRequest(
            ticket_id="T001",
            subject="Test",
            content="Content",
            customer_email="test@example.com"
        )
        assert request.priority == "medium"
    



class TestBatchTicketRequest:
    """Test cases for BatchTicketRequest model."""
    
    def test_valid_batch_request(self):
        """Test creating a valid batch request."""
        tickets = [
            TicketRequest(
                ticket_id="T001",
                subject="Test 1",
                content="Content 1",
                customer_email="test1@example.com"
            ),
            TicketRequest(
                ticket_id="T002", 
                subject="Test 2",
                content="Content 2",
                customer_email="test2@example.com"
            )
        ]
        
        batch_request = BatchTicketRequest(tickets=tickets)
        assert len(batch_request.tickets) == 2
    



class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_create_health_response(self):
        """Test creating a health response."""
        response = HealthResponse(
            status="healthy",
            openai_available=True,
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=123.45
        )
        
        assert response.status == "healthy"
        assert response.openai_available is True
        assert response.version == "1.0.0"
        assert response.uptime_seconds == 123.45
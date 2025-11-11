"""
Tests for the TicketClassifierAgent class.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ticket_classifier.agent import TicketClassifierAgent
from ticket_classifier.models import SupportTicket
from ticket_classifier.config import TestingSettings


class TestTicketClassifierAgent:
    """Test cases for TicketClassifierAgent class."""
    
    def test_init(self):
        """Test initializing the agent."""
        settings = TestingSettings()
        agent = TicketClassifierAgent(settings=settings)
        
        assert agent.settings == settings
        assert agent.classifier is not None
        assert len(agent.processed_tickets) == 0
        assert agent.startup_time > 0
    
    @pytest_asyncio.async_test
    async def test_process_ticket_success(self, agent_with_mock_classifier, sample_ticket):
        """Test successful ticket processing."""
        agent = agent_with_mock_classifier
        
        processed_ticket = await agent.process_ticket(sample_ticket)
        
        assert processed_ticket.category == "technical_issue"
        assert processed_ticket.confidence_score == 0.85
        assert processed_ticket.reasoning is not None
        assert processed_ticket.processed_at is not None
        assert len(agent.processed_tickets) == 1
    
    @pytest_asyncio.async_test
    async def test_process_ticket_exception(self, test_settings, sample_ticket):
        """Test ticket processing with classifier exception."""
        agent = TicketClassifierAgent(settings=test_settings)
        
        # Mock classifier to raise exception
        mock_classifier = Mock()
        mock_classifier.classify_ticket = AsyncMock(side_effect=Exception("Classification failed"))
        agent.classifier = mock_classifier
        
        with pytest.raises(Exception):
            await agent.process_ticket(sample_ticket)
        
        # Ticket should still be added to processed list with error state
        assert len(agent.processed_tickets) == 1
        processed_ticket = agent.processed_tickets[0]
        assert processed_ticket.category == "general_inquiry"
        assert processed_ticket.confidence_score == 0.1
        assert "Processing failed" in processed_ticket.reasoning
    
    @pytest_asyncio.async_test
    async def test_process_batch_success(self, agent_with_mock_classifier, sample_tickets):
        """Test successful batch processing."""
        agent = agent_with_mock_classifier
        
        processed_tickets = await agent.process_batch(sample_tickets)
        
        assert len(processed_tickets) == len(sample_tickets)
        assert len(agent.processed_tickets) == len(sample_tickets)
        
        for ticket in processed_tickets:
            assert ticket.category is not None
            assert ticket.confidence_score is not None
            assert ticket.processed_at is not None
    
    @pytest_asyncio.async_test
    async def test_process_batch_empty_list(self, agent_with_mock_classifier):
        """Test processing empty batch."""
        agent = agent_with_mock_classifier
        
        processed_tickets = await agent.process_batch([])
        
        assert len(processed_tickets) == 0
        assert len(agent.processed_tickets) == 0
    
    @pytest_asyncio.async_test
    async def test_process_batch_exceeds_max_size(self, agent_with_mock_classifier):
        """Test batch processing with size exceeding limit."""
        agent = agent_with_mock_classifier
        agent.settings.max_batch_size = 2
        
        # Create 3 tickets (exceeding limit of 2)
        tickets = [
            SupportTicket(f"T{i}", f"Subject {i}", f"Content {i}", f"user{i}@example.com")
            for i in range(3)
        ]
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            await agent.process_batch(tickets)
    
    def test_get_statistics_empty(self, agent_with_mock_classifier):
        """Test getting statistics with no processed tickets."""
        agent = agent_with_mock_classifier
        
        stats = agent.get_statistics()
        
        assert stats["total_processed"] == 0
        assert "uptime_seconds" in stats
    
    @pytest_asyncio.async_test
    async def test_get_statistics_with_data(self, agent_with_mock_classifier, sample_tickets):
        """Test getting statistics with processed tickets."""
        agent = agent_with_mock_classifier
        
        # Process some tickets
        await agent.process_batch(sample_tickets)
        
        stats = agent.get_statistics()
        
        assert stats["total_processed"] == len(sample_tickets)
        assert "category_distribution" in stats
        assert "average_confidence" in stats
        assert "uptime_seconds" in stats
        assert "categories_available" in stats
    
    def test_get_health_status(self, agent_with_mock_classifier):
        """Test getting health status."""
        agent = agent_with_mock_classifier
        
        health = agent.get_health_status()
        
        assert health["status"] == "healthy"
        assert "openai_available" in health
        assert "total_processed" in health
        assert "uptime_seconds" in health
        assert "settings" in health
    
    def test_clear_processed_tickets(self, agent_with_mock_classifier, sample_ticket):
        """Test clearing processed tickets."""
        agent = agent_with_mock_classifier
        
        # Add a ticket
        agent.processed_tickets.append(sample_ticket)
        assert len(agent.processed_tickets) == 1
        
        # Clear tickets
        agent.clear_processed_tickets()
        assert len(agent.processed_tickets) == 0
    
    def test_get_recent_tickets_empty(self, agent_with_mock_classifier):
        """Test getting recent tickets when list is empty."""
        agent = agent_with_mock_classifier
        
        recent = agent.get_recent_tickets()
        
        assert len(recent) == 0
    
    def test_get_recent_tickets_with_data(self, agent_with_mock_classifier):
        """Test getting recent tickets with data."""
        agent = agent_with_mock_classifier
        
        # Add multiple tickets
        for i in range(15):
            ticket = SupportTicket(f"T{i}", f"Subject {i}", f"Content {i}", f"user{i}@example.com")
            agent.processed_tickets.append(ticket)
        
        # Get recent tickets (default limit 10)
        recent = agent.get_recent_tickets()
        assert len(recent) == 10
        
        # Get recent tickets with custom limit
        recent = agent.get_recent_tickets(limit=5)
        assert len(recent) == 5
    
    @pytest_asyncio.async_test
    async def test_concurrent_processing(self, test_settings):
        """Test that batch processing handles concurrency correctly."""
        agent = TicketClassifierAgent(settings=test_settings)
        
        # Mock classifier with delay to test concurrency
        async def mock_classify_batch(tickets):
            # Simulate some processing delay
            import asyncio
            await asyncio.sleep(0.01)
            return [
                {
                    "category": "general_inquiry",
                    "confidence_score": 0.5,
                    "reasoning": "Mock classification"
                }
                for _ in tickets
            ]
        
        agent.classifier.classify_batch = mock_classify_batch
        
        tickets = [
            SupportTicket(f"T{i}", f"Subject {i}", f"Content {i}", f"user{i}@example.com")
            for i in range(5)
        ]
        
        import time
        start_time = time.time()
        processed_tickets = await agent.process_batch(tickets)
        processing_time = time.time() - start_time
        
        assert len(processed_tickets) == 5
        # Should be faster than processing sequentially
        assert processing_time < 1.0  # Should complete quickly due to mocking
"""
Tests for the TicketClassifier class.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from ticket_classifier.classifier import TicketClassifier
from ticket_classifier.config import UnitTestSettings


class TestTicketClassifier:
    """Test cases for TicketClassifier class."""
    
    def test_init_without_api_key(self):
        """Test initializing classifier without API key."""
        classifier = TicketClassifier(api_key=None)
        assert classifier.client is None
        assert classifier.categories == UnitTestSettings().ticket_categories
    
    def test_init_with_api_key(self):
        """Test initializing classifier with API key."""
        classifier = TicketClassifier(api_key="test-key")
        assert classifier.client is not None
        assert classifier.api_key == "test-key"
    
    def test_system_prompt_creation(self):
        """Test system prompt contains required elements."""
        classifier = TicketClassifier(api_key="test-key")
        prompt = classifier.system_prompt
        
        assert "customer support tickets" in prompt.lower()
        assert "technical_issue" in prompt
        assert "billing_inquiry" in prompt
        assert "JSON format" in prompt
        assert "confidence_score" in prompt
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_success(self, sample_ticket, mock_openai_client):
        """Test successful OpenAI classification."""
        classifier = TicketClassifier(api_key="test-key")
        classifier.client = mock_openai_client
        
        result = await classifier.classify_ticket_with_openai(sample_ticket)
        
        assert result["category"] == "technical_issue"
        assert result["confidence_score"] == 0.85
        assert "reasoning" in result
        
        # Verify OpenAI client was called
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_invalid_json(self, sample_ticket):
        """Test handling of invalid JSON response from OpenAI."""
        classifier = TicketClassifier(api_key="test-key")
        
        # Mock client with invalid JSON response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        mock_client.chat.completions.create.return_value = mock_response
        classifier.client = mock_client
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            await classifier.classify_ticket_with_openai(sample_ticket)
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_invalid_category(self, sample_ticket):
        """Test handling of invalid category from OpenAI."""
        classifier = TicketClassifier(api_key="test-key")
        
        # Mock client with invalid category
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "category": "invalid_category",
            "confidence_score": 0.85,
            "reasoning": "Test"
        })
        mock_client.chat.completions.create.return_value = mock_response
        classifier.client = mock_client
        
        result = await classifier.classify_ticket_with_openai(sample_ticket)
        
        # Should default to general_inquiry
        assert result["category"] == "general_inquiry"
        assert result["confidence_score"] == 0.5
    
    @pytest.mark.asyncio
    async def test_classify_ticket_openai_fallback(self, sample_ticket):
        """Test fallback to rule-based when OpenAI fails."""
        classifier = TicketClassifier(api_key="test-key")
        
        # Mock client that raises exception
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        classifier.client = mock_client
        
        result = await classifier.classify_ticket(sample_ticket)
        
        # Should fall back to rule-based
        assert "category" in result
        assert "confidence_score" in result
        assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_classify_ticket_no_client(self, sample_ticket):
        """Test classification without OpenAI client."""
        classifier = TicketClassifier(api_key=None)
        
        result = await classifier.classify_ticket(sample_ticket)
        
        # The sample ticket has "Test Issue" as subject, which contains "issue" keyword
        # so it should be classified as technical_issue by rule-based classifier
        assert result["category"] == "technical_issue"
        assert result["confidence_score"] > 0.5
        assert "rule-based" in result["reasoning"].lower()
    
    def test_rule_based_classification_technical_issue(self):
        """Test rule-based classification for technical issues."""
        classifier = TicketClassifier(api_key=None)
        
        from ticket_classifier.models import SupportTicket
        ticket = SupportTicket(
            "T001", 
            "App crashes", 
            "The application crashes when I click the button",
            "test@example.com"
        )
        
        result = classifier._rule_based_classification(ticket)
        
        assert result["category"] == "technical_issue"
        assert result["confidence_score"] > 0.5
    
    def test_rule_based_classification_billing(self):
        """Test rule-based classification for billing issues."""
        classifier = TicketClassifier(api_key=None)
        
        from ticket_classifier.models import SupportTicket
        ticket = SupportTicket(
            "T001",
            "Billing question",
            "I have a question about my invoice and charges",
            "test@example.com"
        )
        
        result = classifier._rule_based_classification(ticket)
        
        assert result["category"] == "billing_inquiry"
        assert result["confidence_score"] > 0.5
    
    def test_rule_based_classification_no_match(self):
        """Test rule-based classification with no keyword matches."""
        classifier = TicketClassifier(api_key=None)
        
        from ticket_classifier.models import SupportTicket
        ticket = SupportTicket(
            "T001",
            "Random question",
            "This is some random content with no specific keywords",
            "test@example.com"
        )
        
        result = classifier._rule_based_classification(ticket)
        
        assert result["category"] == "general_inquiry"
        assert result["confidence_score"] == 0.5
    
    @pytest.mark.asyncio
    async def test_classify_batch(self, sample_tickets):
        """Test batch classification."""
        classifier = TicketClassifier(api_key=None)
        
        results = await classifier.classify_batch(sample_tickets)
        
        assert len(results) == len(sample_tickets)
        for result in results:
            assert "category" in result
            assert "confidence_score" in result
            assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_classify_batch_with_exception(self, sample_tickets):
        """Test batch classification handling exceptions."""
        classifier = TicketClassifier(api_key="test-key")
        
        # Mock client that fails for one ticket
        mock_client = AsyncMock()
        call_count = 0
        
        async def mock_classify(ticket):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second ticket
                raise Exception("Classification failed")
            return {
                "category": "general_inquiry",
                "confidence_score": 0.7,
                "reasoning": "Test"
            }
        
        with patch.object(classifier, 'classify_ticket', side_effect=mock_classify):
            results = await classifier.classify_batch(sample_tickets)
        
        assert len(results) == len(sample_tickets)
        # Second result should be fallback
        assert results[1]["category"] == "general_inquiry"
        assert results[1]["confidence_score"] == 0.1
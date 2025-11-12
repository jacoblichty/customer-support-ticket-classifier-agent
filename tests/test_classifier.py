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
        """Test initializing classifier without API key should raise ValueError."""
        # Mock settings to have no API key
        from unittest.mock import patch
        mock_settings = UnitTestSettings()
        mock_settings.azure_openai_api_key = None
        mock_settings.azure_openai_endpoint = None
        
        with patch('ticket_classifier.classifier.get_settings', return_value=mock_settings):
            with pytest.raises(ValueError, match="Azure OpenAI API key and endpoint are required"):
                TicketClassifier(api_key=None)
    
    def test_init_with_api_key(self):
        """Test initializing classifier with API key and endpoint."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        assert classifier.client is not None
        assert classifier.api_key == "test-key"
        assert classifier.endpoint == "test-endpoint"
    
    def test_system_prompt_creation(self):
        """Test system prompt contains required elements."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        prompt = classifier.system_prompt
        
        assert "customer support tickets" in prompt.lower()
        assert "technical_issue" in prompt
        assert "billing_inquiry" in prompt
        assert "ONLY valid JSON" in prompt or "JSON format" in prompt
        assert "confidence_score" in prompt
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_success(self, sample_ticket, mock_openai_client):
        """Test successful Azure OpenAI classification."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        classifier.client = mock_openai_client
        
        result = await classifier.classify_ticket(sample_ticket)
        
        assert result["category"] == "technical_issue"
        assert result["confidence_score"] == 0.85
        assert "reasoning" in result
        
        # Verify OpenAI client was called
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_invalid_json(self, sample_ticket):
        """Test handling of invalid JSON response from Azure OpenAI."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        
        # Mock client with invalid JSON response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        mock_client.chat.completions.create.return_value = mock_response
        classifier.client = mock_client
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            await classifier.classify_ticket(sample_ticket)
    
    @pytest.mark.asyncio
    async def test_classify_ticket_with_openai_invalid_category(self, sample_ticket):
        """Test handling of invalid category from Azure OpenAI."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        
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
        
        result = await classifier.classify_ticket(sample_ticket)
        
        # Should default to general_inquiry
        assert result["category"] == "general_inquiry"
        assert result["confidence_score"] == 0.5
    
    @pytest.mark.asyncio
    async def test_classify_ticket_openai_failure(self, sample_ticket):
        """Test that Azure OpenAI failure raises exception."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        
        # Mock client that raises exception
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        classifier.client = mock_client
        
        # Should raise the exception, no fallback
        with pytest.raises(Exception, match="API Error"):
            await classifier.classify_ticket(sample_ticket)
    


    
    @pytest.mark.asyncio
    async def test_classify_batch(self, sample_tickets, mock_openai_client):
        """Test batch classification."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        classifier.client = mock_openai_client
        
        results = await classifier.classify_batch(sample_tickets)
        
        assert len(results) == len(sample_tickets)
        for result in results:
            assert "category" in result
            assert "confidence_score" in result
            assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_classify_batch_with_exception(self, sample_tickets):
        """Test batch classification handling exceptions."""
        classifier = TicketClassifier(api_key="test-key", endpoint="test-endpoint")
        
        # Mock client that fails for one ticket
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
            # Should raise exception when one ticket fails
            with pytest.raises(Exception, match="Classification failed"):
                await classifier.classify_batch(sample_tickets)
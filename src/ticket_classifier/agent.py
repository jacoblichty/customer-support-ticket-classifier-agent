"""
Main agent for processing support tickets.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from .models import SupportTicket
from .classifier import TicketClassifier
from .config import get_settings

logger = logging.getLogger(__name__)


class TicketClassifierAgent:
    """Main agent for processing support tickets with OpenAI integration."""
    
    def __init__(self, openai_api_key: Optional[str] = None, settings=None):
        self.settings = settings or get_settings()
        self.classifier = TicketClassifier(openai_api_key, self.settings)
        self.processed_tickets = []
        self.startup_time = time.time()
        
        logger.info("Ticket Classifier Agent initialized")
    
    async def process_ticket(self, ticket: SupportTicket) -> SupportTicket:
        """
        Process a single support ticket through the classification pipeline.
        
        Args:
            ticket: SupportTicket object to process
            
        Returns:
            Updated SupportTicket with classification results
        """
        logger.info(f"Processing ticket: {ticket.ticket_id}")
        start_time = time.time()
        
        try:
            # Classify the ticket
            result = await self.classifier.classify_ticket(ticket)
            
            # Update ticket with classification results
            ticket.category = result["category"]
            ticket.confidence_score = result["confidence_score"]
            ticket.reasoning = result["reasoning"]
            ticket.processed_at = datetime.now()
            
            processing_time = time.time() - start_time
            
            # Log the classification result
            logger.info(f"Ticket {ticket.ticket_id} classified as '{ticket.category}' "
                       f"with confidence {ticket.confidence_score:.2f} "
                       f"in {processing_time:.2f}s")
            
            self.processed_tickets.append(ticket)
            return ticket
            
        except Exception as e:
            logger.error(f"Error processing ticket {ticket.ticket_id}: {e}")
            
            # Set error state on ticket
            ticket.category = "general_inquiry"
            ticket.confidence_score = 0.1
            ticket.reasoning = f"Processing failed: {str(e)}"
            ticket.processed_at = datetime.now()
            
            self.processed_tickets.append(ticket)
            raise
    
    async def process_batch(self, tickets: List[SupportTicket]) -> List[SupportTicket]:
        """
        Process multiple tickets in batch with concurrent processing.
        
        Args:
            tickets: List of SupportTicket objects to process
            
        Returns:
            List of processed SupportTicket objects
        """
        if not tickets:
            return []
        
        if len(tickets) > self.settings.max_batch_size:
            raise ValueError(f"Batch size {len(tickets)} exceeds maximum {self.settings.max_batch_size}")
        
        logger.info(f"Processing batch of {len(tickets)} tickets")
        start_time = time.time()
        
        try:
            # Classify all tickets concurrently
            results = await self.classifier.classify_batch(tickets)
            
            # Update tickets with results
            processed_tickets = []
            for ticket, result in zip(tickets, results):
                ticket.category = result["category"]
                ticket.confidence_score = result["confidence_score"]
                ticket.reasoning = result["reasoning"]
                ticket.processed_at = datetime.now()
                processed_tickets.append(ticket)
            
            # Update processed tickets list
            self.processed_tickets.extend(processed_tickets)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed batch of {len(tickets)} tickets in {processing_time:.2f}s")
            
            return processed_tickets
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        if not self.processed_tickets:
            return {
                "total_processed": 0,
                "uptime_seconds": time.time() - self.startup_time
            }
        
        # Calculate category distribution
        category_counts = {}
        confidence_scores = []
        
        for ticket in self.processed_tickets:
            category_counts[ticket.category] = category_counts.get(ticket.category, 0) + 1
            if ticket.confidence_score is not None:
                confidence_scores.append(ticket.confidence_score)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "total_processed": len(self.processed_tickets),
            "category_distribution": category_counts,
            "average_confidence": round(avg_confidence, 3),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "categories_available": self.settings.ticket_categories
        }
    
    def get_health_status(self) -> Dict:
        """Get health status information."""
        return {
            "status": "healthy",
            "openai_available": bool(self.classifier.client),
            "total_processed": len(self.processed_tickets),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "settings": {
                "max_batch_size": self.settings.max_batch_size,
                "max_concurrent_requests": self.settings.max_concurrent_requests,
                "openai_model": self.settings.openai_model
            }
        }
    
    def clear_processed_tickets(self):
        """Clear the processed tickets list (useful for testing)."""
        logger.info(f"Clearing {len(self.processed_tickets)} processed tickets")
        self.processed_tickets.clear()
    
    def get_recent_tickets(self, limit: int = 10) -> List[SupportTicket]:
        """Get the most recently processed tickets."""
        return self.processed_tickets[-limit:] if self.processed_tickets else []
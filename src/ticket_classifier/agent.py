"""
Main agent for classifying support tickets with AI-powered analysis.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

from .models import SupportTicket
from .classifier import TicketClassifier
from .config import get_settings

logger = logging.getLogger(__name__)


class TicketClassifierAgent:
    """Main agent for classifying support tickets."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, settings=None):
        self.settings = settings or get_settings()
        self.classifier = TicketClassifier(api_key, endpoint, self.settings)
        self.processed_tickets = []
        self.startup_time = time.time()
        
        logger.info("Ticket Classifier Agent initialized")
    
    async def process_ticket(self, ticket: SupportTicket) -> SupportTicket:
        """
        Process a single support ticket with classification, confidence scoring, and reasoning.
        
        Args:
            ticket: SupportTicket object to process
            
        Returns:
            Updated SupportTicket with classification results and metadata
        """
        logger.info(f"Processing ticket: {ticket.ticket_id}")
        start_time = time.time()
        
        try:
            # Classify the ticket using Azure OpenAI
            classification_result = await self.classifier.classify_ticket(ticket)
            
            # Update ticket with classification results
            ticket.category = classification_result["category"]
            ticket.confidence_score = classification_result["confidence_score"]
            ticket.reasoning = classification_result["reasoning"]
            ticket.processed_at = datetime.now()
            
            # Add processing metadata
            processing_time = time.time() - start_time
            metadata_update = {
                'processing_time_seconds': round(processing_time, 3),
                'classification_method': 'azure_openai',
                'confidence_level': self._get_confidence_level(classification_result["confidence_score"]),
                'requires_human_review': classification_result["confidence_score"] < 0.7
            }
            
            if hasattr(ticket, 'metadata'):
                ticket.metadata.update(metadata_update)
            else:
                ticket.metadata = metadata_update
            
            logger.info(f"Ticket {ticket.ticket_id} classified as '{ticket.category}' "
                       f"with confidence {ticket.confidence_score:.2f} in {processing_time:.2f}s")
            
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
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """Convert confidence score to human-readable level."""
        if confidence_score >= 0.9:
            return "very_high"
        elif confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.7:
            return "moderate"
        elif confidence_score >= 0.5:
            return "low"
        else:
            return "very_low"
    
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
            
            # Update tickets with results and add metadata
            processed_tickets = []
            for ticket, result in zip(tickets, results):
                ticket.category = result["category"]
                ticket.confidence_score = result["confidence_score"]
                ticket.reasoning = result["reasoning"]
                ticket.processed_at = datetime.now()
                
                # Add basic metadata for batch processing
                ticket.metadata = {
                    'processing_time_seconds': round((time.time() - start_time) / len(tickets), 3),
                    'classification_method': 'azure_openai_batch',
                    'confidence_level': self._get_confidence_level(result["confidence_score"]),
                    'requires_human_review': result["confidence_score"] < 0.7
                }
                
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
                "category_distribution": {},
                "average_confidence": 0.0,
                "confidence_distribution": {},
                "human_review_needed": 0,
                "human_review_rate": 0.0,
                "uptime_seconds": round(time.time() - self.startup_time, 2),
                "categories_available": self.settings.ticket_categories
            }
        
        # Calculate category distribution
        category_counts = {}
        confidence_scores = []
        confidence_levels = {}
        human_review_needed = 0
        
        for ticket in self.processed_tickets:
            category_counts[ticket.category] = category_counts.get(ticket.category, 0) + 1
            if ticket.confidence_score is not None:
                confidence_scores.append(ticket.confidence_score)
                
                # Track confidence levels
                level = self._get_confidence_level(ticket.confidence_score)
                confidence_levels[level] = confidence_levels.get(level, 0) + 1
                
                # Track human review needs
                if ticket.confidence_score < 0.7:
                    human_review_needed += 1
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        stats = {
            "total_processed": len(self.processed_tickets),
            "category_distribution": category_counts,
            "average_confidence": round(avg_confidence, 3),
            "confidence_distribution": confidence_levels,
            "human_review_needed": human_review_needed,
            "human_review_rate": round(human_review_needed / len(self.processed_tickets), 3) if self.processed_tickets else 0.0,
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "categories_available": self.settings.ticket_categories
        }
        
        return stats
    
    def get_health_status(self) -> Dict:
        """Get health status information."""
        health_status = {
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
        
        return health_status
    
    def clear_processed_tickets(self):
        """Clear the processed tickets list (useful for testing)."""
        logger.info(f"Clearing {len(self.processed_tickets)} processed tickets")
        self.processed_tickets.clear()
    
    def get_recent_tickets(self, limit: int = 10) -> List[SupportTicket]:
        """Get the most recently processed tickets."""
        return self.processed_tickets[-limit:] if self.processed_tickets else []
    
    def get_classification_insights(self) -> Dict:
        """Get insights from the classification history."""
        if not self.processed_tickets:
            return {
                "total_classifications": 0,
                "average_confidence": 0.0,
                "category_performance": {},
                "low_confidence_rate": 0.0
            }
        
        # Calculate metrics
        total_classifications = len(self.processed_tickets)
        confidence_scores = [t.confidence_score for t in self.processed_tickets if t.confidence_score is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        low_confidence_count = sum(1 for score in confidence_scores if score < 0.7)
        
        # Category performance analysis
        category_performance = {}
        for ticket in self.processed_tickets:
            if ticket.category and ticket.confidence_score is not None:
                if ticket.category not in category_performance:
                    category_performance[ticket.category] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "low_confidence_rate": 0.0
                    }
                
                perf = category_performance[ticket.category]
                perf["count"] += 1
                perf["avg_confidence"] += ticket.confidence_score
                if ticket.confidence_score < 0.7:
                    perf["low_confidence_rate"] += 1
        
        # Finalize averages
        for category, perf in category_performance.items():
            count = perf["count"]
            perf["avg_confidence"] = round(perf["avg_confidence"] / count, 3)
            perf["low_confidence_rate"] = round(perf["low_confidence_rate"] / count, 3)
        
        return {
            "total_classifications": total_classifications,
            "average_confidence": round(avg_confidence, 3),
            "category_performance": category_performance,
            "low_confidence_rate": round(low_confidence_count / len(confidence_scores), 3) if confidence_scores else 0.0,
            "most_classified_category": max(category_performance.keys(), key=lambda k: category_performance[k]["count"]) if category_performance else None,
            "highest_confidence_category": max(category_performance.keys(), key=lambda k: category_performance[k]["avg_confidence"]) if category_performance else None
        }
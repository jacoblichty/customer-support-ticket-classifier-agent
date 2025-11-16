"""
Data models for the ticket classifier system.
"""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    """API request model for ticket classification."""
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    subject: str = Field(..., description="Subject line of the ticket", min_length=1)
    content: str = Field(..., description="Full content/body of the ticket", min_length=1)
    customer_email: str = Field(..., description="Customer's email address")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, urgent")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticket_id": "T001",
                "subject": "Login Issue",
                "content": "I can't log into my account. Getting error 'Invalid credentials'",
                "customer_email": "user@example.com",
                "priority": "high"
            }
        }
    }


class ProcessingDetails(BaseModel):
    """Classification processing and analysis details."""
    classification_method: str = Field(..., description="Method used for classification (e.g., 'azure_openai')")
    processing_time_seconds: float = Field(..., description="Time taken to process the ticket")
    confidence_level: str = Field(..., description="Human-readable confidence level (very_low, low, moderate, high, very_high)")
    requires_human_review: bool = Field(..., description="Whether human review is recommended based on confidence")

    model_config = {
        "json_schema_extra": {
            "example": {
                "classification_method": "azure_openai",
                "processing_time_seconds": 0.85,
                "confidence_level": "high",
                "requires_human_review": False
            }
        }
    }


class TicketResponse(BaseModel):
    """API response model for processed ticket."""
    ticket_id: str
    subject: str
    content: str
    customer_email: str
    priority: str
    category: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    processing_details: Optional[ProcessingDetails] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticket_id": "T001",
                "subject": "Login Issue",
                "content": "I can't log into my account",
                "customer_email": "user@example.com",
                "priority": "high",
                "category": "account_management",
                "confidence_score": 0.85,
                "reasoning": "Keywords indicate account access issues",
                "created_at": "2025-11-11T10:30:00",
                "processed_at": "2025-11-11T10:30:01",
                "processing_details": {
                    "classification_method": "azure_openai",
                    "processing_time_seconds": 0.85,
                    "confidence_level": "very_high",
                    "requires_human_review": False
                }
            }
        }
    }


class BatchTicketRequest(BaseModel):
    """API request model for batch ticket classification."""
    tickets: List[TicketRequest] = Field(..., min_length=1, max_length=100)

    model_config = {
        "json_schema_extra": {
            "example": {
                "tickets": [
                    {
                        "ticket_id": "T001",
                        "subject": "Login Issue",
                        "content": "I can't log into my account",
                        "customer_email": "user1@example.com",
                        "priority": "high"
                    },
                    {
                        "ticket_id": "T002",
                        "subject": "Billing Question",
                        "content": "Why was I charged twice?",
                        "customer_email": "user2@example.com",
                        "priority": "medium"
                    }
                ]
            }
        }
    }


class BatchTicketResponse(BaseModel):
    """API response model for batch ticket processing."""
    processed_tickets: List[TicketResponse]
    statistics: Dict
    processing_time_seconds: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "processed_tickets": [
                    {
                        "ticket_id": "T001",
                        "category": "account_management",
                        "confidence_score": 0.85,
                        "processing_details": {
                            "classification_method": "azure_openai",
                            "processing_time_seconds": 0.85,
                            "confidence_level": "very_high",
                            "requires_human_review": False
                        }
                    }
                ],
                "statistics": {
                    "total_processed": 2,
                    "category_distribution": {
                        "account_management": 1,
                        "billing_inquiry": 1
                    },
                    "average_confidence": 0.82,
                    "confidence_distribution": {
                        "very_high": 2
                    },
                    "human_review_rate": 0.0
                },
                "processing_time_seconds": 1.23
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    openai_available: bool
    timestamp: datetime
    version: str
    uptime_seconds: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "openai_available": True,
                "timestamp": "2025-11-11T10:30:00",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }
    }


class SupportTicket:
    """Internal representation of a customer support ticket."""
    
    def __init__(self, ticket_id: str, subject: str, content: str, 
                 customer_email: str, priority: str = "medium"):
        self.ticket_id = ticket_id
        self.subject = subject
        self.content = content
        self.customer_email = customer_email
        self.priority = priority
        self.created_at = datetime.now()
        self.processed_at = None
        self.category = None
        self.confidence_score = None
        self.reasoning = None
    
    def to_response(self) -> TicketResponse:
        """Convert to API response model."""
        processing_details = None
        
        # Include processing details if metadata exists
        if hasattr(self, 'metadata') and isinstance(self.metadata, dict):
            # Use processing time from metadata if available, otherwise calculate
            processing_time = self.metadata.get('processing_time_seconds', 0.0)
            if processing_time == 0.0 and self.processed_at and hasattr(self, 'created_at'):
                processing_time = (self.processed_at - self.created_at).total_seconds()
            
            processing_details = ProcessingDetails(
                classification_method=self.metadata.get('classification_method', 'azure_openai'),
                processing_time_seconds=round(processing_time, 3),
                confidence_level=self.metadata.get('confidence_level', 'moderate'),
                requires_human_review=self.metadata.get('requires_human_review', False)
            )
        
        return TicketResponse(
            ticket_id=self.ticket_id,
            subject=self.subject,
            content=self.content,
            customer_email=self.customer_email,
            priority=self.priority,
            category=self.category,
            confidence_score=self.confidence_score,
            reasoning=self.reasoning,
            created_at=self.created_at,
            processed_at=self.processed_at,
            processing_details=processing_details
        )
    
    @classmethod
    def from_request(cls, request: TicketRequest) -> 'SupportTicket':
        """Create from API request model."""
        return cls(
            ticket_id=request.ticket_id,
            subject=request.subject,
            content=request.content,
            customer_email=request.customer_email,
            priority=request.priority
        )
    
    def __str__(self) -> str:
        return f"Ticket {self.ticket_id}: {self.subject} (Priority: {self.priority})"
    
    def __repr__(self) -> str:
        return (f"SupportTicket(ticket_id='{self.ticket_id}', "
                f"subject='{self.subject[:30]}...', "
                f"priority='{self.priority}')")
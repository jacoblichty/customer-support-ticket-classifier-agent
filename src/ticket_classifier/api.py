"""
FastAPI application for the ticket classifier system.
"""

import time
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    TicketRequest, TicketResponse, BatchTicketRequest, BatchTicketResponse,
    HealthResponse, SupportTicket
)
from .agent import TicketClassifierAgent
from .config import get_settings
from .logging_config import setup_logging, get_logger

# Global variables
app_start_time = time.time()
agent: Optional[TicketClassifierAgent] = None
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    setup_logging(settings)
    
    global agent
    agent = TicketClassifierAgent(settings.azure_openai_api_key, settings.azure_openai_endpoint, settings)
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


# Initialize FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    # Enable docs in development and when debug is True
    enable_docs = settings.debug or os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered customer support ticket classification using OpenAI",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
    
    # Add request timing middleware
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


def register_routes(app: FastAPI):
    """Register all routes with the FastAPI app."""
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        settings = get_settings()
        return {
            "message": settings.app_name,
            "version": settings.app_version,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "process": "/process - Process and classify a single ticket",
                "process_batch": "/process/batch - Process and classify multiple tickets",
                "health": "/health - Health check",
                "stats": "/stats - Get processing statistics",
                "categories": "/categories - Get available categories"
            }
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        settings = get_settings()
        health_status = agent.get_health_status()
        
        return HealthResponse(
            status=health_status["status"],
            openai_available=health_status["openai_available"],
            timestamp=datetime.now(),
            version=settings.app_version,
            uptime_seconds=health_status["uptime_seconds"]
        )

    @app.post("/process", response_model=TicketResponse, tags=["Processing"])
    async def process_ticket(ticket_request: TicketRequest):
        """Process and classify a single support ticket with intelligent routing."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            logger.info(f"Received processing request for ticket {ticket_request.ticket_id}")
            
            # Convert request to ticket object
            ticket = SupportTicket.from_request(ticket_request)
            
            # Process the ticket with intelligent routing
            processed_ticket = await agent.process_ticket(ticket)
            
            # Return response with processing details
            return processed_ticket.to_response()
            
        except Exception as e:
            logger.error(f"Error processing ticket {ticket_request.ticket_id}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Processing error: {str(e)}"
            )

    @app.post("/process/batch", response_model=BatchTicketResponse, tags=["Processing"])
    async def process_tickets_batch(batch_request: BatchTicketRequest):
        """Process and classify multiple support tickets in batch with intelligent routing."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        settings = get_settings()
        
        # Validate batch size
        if len(batch_request.tickets) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(batch_request.tickets)} exceeds maximum {settings.max_batch_size}"
            )
        
        try:
            logger.info(f"Received batch processing request for {len(batch_request.tickets)} tickets")
            start_time = time.time()
            
            # Process each ticket individually to get intelligent routing details
            processed_tickets = []
            for req in batch_request.tickets:
                ticket = SupportTicket.from_request(req)
                processed_ticket = await agent.process_ticket(ticket)
                processed_tickets.append(processed_ticket)
            
            # Get statistics
            stats = agent.get_statistics()
            processing_time = time.time() - start_time
            
            # Return response with processing details for each ticket
            return BatchTicketResponse(
                processed_tickets=[ticket.to_response() for ticket in processed_tickets],
                statistics=stats,
                processing_time_seconds=round(processing_time, 3)
            )
            
        except ValueError as e:
            logger.error(f"Validation error in batch processing: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch processing error: {str(e)}"
            )

    @app.get("/stats", tags=["Statistics"])
    async def get_statistics():
        """Get processing statistics."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        return agent.get_statistics()

    @app.get("/categories", tags=["Configuration"])
    async def get_categories():
        """Get available ticket categories with descriptions."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        return {
            "categories": agent.classifier.categories,
            "descriptions": {
                "technical_issue": "Problems with software, hardware, bugs, errors, system not working",
                "billing_inquiry": "Questions about charges, payments, invoices, billing cycles",
                "account_management": "Login issues, password resets, account settings, profile changes",
                "feature_request": "Requests for new features, enhancements, suggestions for improvement",
                "general_inquiry": "General questions, information requests, how-to questions",
                "complaint": "Expressions of dissatisfaction, complaints about service or product",
                "refund_request": "Requests for money back, returns, cancellations for refund"
            }
        }

    @app.get("/recent", tags=["Statistics"])
    async def get_recent_tickets(limit: int = 10):
        """Get recently processed tickets."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        if limit > 100:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
        
        recent_tickets = agent.get_recent_tickets(limit)
        return {
            "tickets": [ticket.to_response() for ticket in recent_tickets],
            "count": len(recent_tickets)
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Create the global app instance
app = create_app()


def run_server(settings=None):
    """Run the FastAPI server."""
    if settings is None:
        settings = get_settings()
    
    logger.info(f"Starting {settings.app_name} server...")
    logger.info(f"Server will be available at: http://{settings.host}:{settings.port}")
    logger.info(f"API documentation at: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "ticket_classifier.api:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # Always use 1 worker for development
        reload=False,  # Disable reload to avoid log spam
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run_server()
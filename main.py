#!/usr/bin/env python3
"""
Customer Support Ticket Classifier Agent - Entry Point

An AI-powered system using OpenAI for automatically classifying customer support tickets
into predefined categories to improve response times and routing efficiency.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ticket_classifier.agent import TicketClassifierAgent
from ticket_classifier.models import SupportTicket
from ticket_classifier.config import get_settings, get_environment_settings
from ticket_classifier.logging_config import setup_logging, get_logger
from ticket_classifier.api import run_server

logger = get_logger(__name__)


def create_sample_tickets() -> list[SupportTicket]:
    """Create some sample tickets for testing."""
    return [
        SupportTicket("T001", "Login Issue", "I can't log into my account. Getting error 'Invalid credentials'", "user1@example.com", "high"),
        SupportTicket("T002", "Billing Question", "Why was I charged twice this month? I only have one subscription.", "user2@example.com", "medium"),
        SupportTicket("T003", "Feature Request", "Can you add dark mode to the application? It would be great for night usage.", "user3@example.com", "low"),
        SupportTicket("T004", "Bug Report", "The app crashes when I click the submit button on the contact form.", "user4@example.com", "high"),
        SupportTicket("T005", "Refund Request", "I want my money back for last month's subscription. Service was terrible.", "user5@example.com", "medium"),
        SupportTicket("T006", "Account Help", "I forgot my password and the reset email isn't working.", "user6@example.com", "medium"),
        SupportTicket("T007", "General Question", "How do I change my notification settings?", "user7@example.com", "low"),
        SupportTicket("T008", "Complaint", "I'm very dissatisfied with the customer service response time.", "user8@example.com", "high")
    ]


async def run_demo():
    """Run demo with sample tickets."""
    print("🎫 Customer Support Ticket Classifier Agent Demo")
    print("=" * 60)
    
    # Get settings and setup logging
    settings = get_environment_settings()
    setup_logging(settings)
    
    # Initialize agent
    print(f"🤖 Initializing agent with OpenAI model: {settings.openai_model}")
    agent = TicketClassifierAgent(settings.openai_api_key, settings)
    
    # Create sample tickets
    sample_tickets = create_sample_tickets()
    print(f"📋 Created {len(sample_tickets)} sample tickets")
    print()
    
    # Process tickets
    print("🔄 Processing tickets...")
    processed_tickets = await agent.process_batch(sample_tickets)
    
    # Display results
    print("\n📊 Classification Results:")
    print("-" * 60)
    
    for ticket in processed_tickets:
        print(f"🎫 Ticket ID: {ticket.ticket_id}")
        print(f"   Subject: {ticket.subject}")
        print(f"   Priority: {ticket.priority}")
        print(f"   Category: {ticket.category}")
        print(f"   Confidence: {ticket.confidence_score:.2f}")
        print(f"   Reasoning: {ticket.reasoning}")
        print("-" * 40)
    
    # Show statistics
    print("\n📈 Statistics:")
    print("-" * 30)
    stats = agent.get_statistics()
    for key, value in stats.items():
        if key == "category_distribution":
            print(f"{key}:")
            for category, count in value.items():
                print(f"  - {category}: {count}")
        else:
            print(f"{key}: {value}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Customer Support Ticket Classifier Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run demo with sample data
  python main.py --server                  # Start FastAPI server
  python main.py --server --port 8080      # Start server on custom port
  python main.py --server --env production # Start in production mode
        """
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo with sample data")
    parser.add_argument("--server", action="store_true", 
                       help="Start FastAPI server")
    parser.add_argument("--host", default=None, 
                       help="Host to bind the server to (overrides config)")
    parser.add_argument("--port", type=int, default=None, 
                       help="Port to bind the server to (overrides config)")
    parser.add_argument("--env", choices=["development", "production", "testing"], 
                       default=None, help="Environment to use")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set environment if specified
    if args.env:
        os.environ["ENVIRONMENT"] = args.env
    
    # Get settings
    settings = get_environment_settings()
    
    # Override settings with command line arguments
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.verbose:
        settings.log_level = "DEBUG"
    
    # Setup logging
    setup_logging(settings)
    
    if args.demo:
        # Run demo
        asyncio.run(run_demo())
    elif args.server:
        # Start FastAPI server
        print(f"🚀 Starting {settings.app_name}")
        print(f"🌐 Server will be available at: http://{settings.host}:{settings.port}")
        print(f"📚 API documentation at: http://{settings.host}:{settings.port}/docs")
        print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print(f"🤖 OpenAI available: {bool(settings.openai_api_key)}")
        
        run_server(settings)
    else:
        print("🎫 Customer Support Ticket Classifier Agent")
        print()
        print("Available commands:")
        print("  --demo     Run with sample data to test classification")
        print("  --server   Start FastAPI server for API access")
        print("  --help     Show detailed help and examples")
        print()
        print("Quick start:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Run: python main.py --demo")
        print("  3. Or start server: python main.py --server")


if __name__ == "__main__":
    main()
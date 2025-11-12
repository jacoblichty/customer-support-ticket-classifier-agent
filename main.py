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
    """Create basic sample tickets for simple testing (kept for backward compatibility)."""
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
    """Run comprehensive demo showcasing intelligent agent capabilities."""
    print("🧠 Customer Support Ticket Classifier - Intelligent Agent Demo")
    print("=" * 70)
    print("This demo showcases intelligent routing and autonomous decision-making capabilities")
    print()
    
    # Get settings and setup logging
    try:
        settings = get_environment_settings()
        print(f"📋 Configuration loaded:")
        print(f"   • Environment: {getattr(settings, 'environment', 'default')}")
        print(f"   • Debug mode: {settings.debug}")
        print(f"   • Azure OpenAI: {'✅ Configured' if settings.azure_openai_api_key else '❌ Not configured'}")
        print(f"   • Intelligent routing: ✅ Always enabled")
    except Exception as e:
        print(f"⚠️  Configuration warning: {e}")
        print("   Using fallback settings...")
        settings = get_settings()
    
    setup_logging(settings)
    
    # Initialize agent
    agent = TicketClassifierAgent(settings.azure_openai_api_key, settings.azure_openai_endpoint, settings)
    print(f"\n🤖 Agent initialized:")
    print(f"   • Intelligent routing: ✅ Always active")
    print(f"   • OpenAI client: {'✅ Ready' if agent.classifier.client else '❌ Not available'}")
    print(f"   • Azure OpenAI deployment: {settings.azure_openai_deployment_name}")
    
    # Create comprehensive sample tickets
    sample_tickets = create_comprehensive_sample_tickets()
    print(f"\n📋 Created {len(sample_tickets)} sample tickets showcasing different scenarios")
    
    # Process tickets with intelligent routing
    print("\n🔄 Processing tickets with intelligent routing...")
    print("=" * 70)
    
    processed_tickets = []
    total_start_time = asyncio.get_event_loop().time()
    
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n🎫 Processing Ticket {i}/{len(sample_tickets)}: {ticket.ticket_id}")
        print(f"   Subject: {ticket.subject}")
        print(f"   Priority: {ticket.priority}")
        print(f"   Content Preview: {ticket.content[:80]}...")
        
        # Process individual ticket to get intelligent metadata
        start_time = asyncio.get_event_loop().time()
        processed_ticket = await agent.process_ticket(ticket)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        processed_tickets.append(processed_ticket)
        
        # Display intelligent processing results
        print(f"   🤖 Strategy: {processed_ticket.metadata.get('processing_strategy', 'N/A')}")
        print(f"   ✅ Category: {processed_ticket.category}")
        print(f"   📊 Confidence: {processed_ticket.confidence_score:.3f}")
        print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
        print(f"   🚨 Escalated: {'Yes' if processed_ticket.metadata.get('escalation_triggered') else 'No'}")
        if processed_ticket.metadata.get('auto_resolution_attempted'):
            print(f"   🔧 Auto-resolution attempted")
        if processed_ticket.metadata.get('follow_up_scheduled'):
            print(f"   📅 Follow-up scheduled")
        print(f"   💭 Reasoning: {processed_ticket.reasoning[:120]}...")
    
    total_processing_time = asyncio.get_event_loop().time() - total_start_time
    
    # Show comprehensive statistics
    print(f"\n📊 Intelligent Processing Summary:")
    print("=" * 70)
    
    stats = agent.get_statistics()
    print(f"📈 Processing Statistics:")
    print(f"   • Total Tickets: {stats['total_processed']}")
    print(f"   • Total Processing Time: {total_processing_time:.2f}s")
    print(f"   • Average Time per Ticket: {total_processing_time/len(processed_tickets):.2f}s")
    print(f"   • Average Confidence: {stats['average_confidence']:.3f}")
    print(f"   • Intelligent Processing Rate: {stats.get('intelligent_processing_rate', 1.0):.1%}")
    
    print(f"\n🎯 Strategy Distribution:")
    strategy_dist = stats.get('strategy_distribution', {})
    for strategy, count in strategy_dist.items():
        percentage = (count / len(processed_tickets)) * 100
        print(f"   • {strategy.replace('_', ' ').title()}: {count} times ({percentage:.1f}%)")
    
    print(f"\n📋 Category Distribution:")
    for category, count in stats['category_distribution'].items():
        percentage = (count / len(processed_tickets)) * 100
        print(f"   • {category.replace('_', ' ').title()}: {count} times ({percentage:.1f}%)")
    
    if stats.get('escalations_triggered', 0) > 0:
        print(f"\n🚨 Escalation Summary:")
        print(f"   • Total Escalations: {stats['escalations_triggered']}")
        print(f"   • Escalation Rate: {(stats['escalations_triggered']/len(processed_tickets)):.1%}")
    
    # Show decision insights
    insights = agent.get_decision_insights()
    if insights['total_decisions'] > 0:
        print(f"\n🧠 Intelligent Decision Insights:")
        print(f"   • Decision History Size: {insights['total_decisions']}")
        print(f"   • Human Review Rate: {insights['human_review_rate']:.1%}")
        print(f"   • Most Used Strategy: {insights.get('most_used_strategy', 'N/A').replace('_', ' ').title()}")
        
        if insights['strategy_effectiveness']:
            print(f"\n⚡ Strategy Effectiveness:")
            for strategy, effectiveness in insights['strategy_effectiveness'].items():
                print(f"   • {strategy.replace('_', ' ').title()}: "
                      f"{effectiveness['count']} uses, "
                      f"avg confidence {effectiveness['avg_confidence']:.3f}, "
                      f"avg time {effectiveness['avg_time']:.1f}s")
    
    # Show health status
    health = agent.get_health_status()
    print(f"\n🏥 System Health:")
    print(f"   • Status: {health['status'].title()}")
    print(f"   • Uptime: {health['uptime_seconds']:.1f} seconds")
    print(f"   • Strategies Available: {len(health['intelligent_settings']['strategies_available'])}")
    
    print(f"\n✨ Key Intelligent Features Demonstrated:")
    print(f"   • Autonomous strategy selection based on ticket analysis")
    print(f"   • Context-aware processing decisions")
    print(f"   • Dynamic confidence threshold management")
    print(f"   • Smart human escalation routing")
    print(f"   • Adaptive post-processing decisions")
    print(f"   • Continuous learning from outcomes")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"    The agent processed all tickets using intelligent routing!")


def create_comprehensive_sample_tickets() -> list[SupportTicket]:
    """Create comprehensive sample tickets showcasing different scenarios and complexities."""
    return [
        # Simple, clear tickets (fast-track candidates)
        SupportTicket(
            "SIMPLE001", 
            "Password reset request",
            "Hi, I forgot my password and can't log into my account. My email is john@example.com. Please help me reset it as soon as possible. Thanks!",
            "john@example.com",
            "low"
        ),
        
        SupportTicket(
            "SIMPLE002",
            "Change email address",
            "I need to update my account email from old@email.com to new@email.com. How do I do this?",
            "old@email.com",
            "medium"
        ),
        
        # Technical issues (thorough analysis candidates)
        SupportTicket(
            "TECH001",
            "API Integration Failure with 500 Errors",
            "Our production API integration has been failing intermittently for the past 3 hours. We're getting 500 internal server errors on POST requests to /api/payments. This is affecting our payment processing and causing revenue loss. Error logs show 'Connection timeout' and 'SSL handshake failed'. We need immediate technical assistance.",
            "devops@company.com",
            "urgent"
        ),
        
        SupportTicket(
            "TECH002",
            "Application crashes when uploading large files",
            "The application keeps crashing whenever I try to upload files larger than 10MB. I get an error message saying 'Memory allocation failed' and then the whole app freezes. This is affecting my work productivity. Using Chrome browser on Windows 10 with 16GB RAM.",
            "user@business.com",
            "high"
        ),
        
        # Sensitive/security issues (escalation candidates)
        SupportTicket(
            "SECURITY001",
            "Suspected unauthorized account access",
            "I believe someone has gained unauthorized access to my account. I see login attempts from IP addresses I don't recognize, and there are charges on my account that I didn't authorize. I'm very concerned about a potential security breach and need immediate help. This may involve personal data exposure.",
            "concerned@user.com",
            "urgent"
        ),
        
        # Complex complaints (human escalation candidates)
        SupportTicket(
            "COMPLAINT001",
            "Extremely frustrated with repeated service failures",
            "This is the third time I'm contacting support about the same issue and nobody seems to understand the problem. My data keeps getting corrupted and I've lost hours of work. This is unacceptable for a paid service. I'm considering legal action and switching to a competitor if this isn't resolved immediately. I've been a customer for 3 years and this is the worst experience I've ever had.",
            "angry@customer.com",
            "high"
        ),
        
        # Billing inquiries (context-enriched candidates)
        SupportTicket(
            "BILLING001",
            "Unexpected charges on enterprise account",
            "I was charged $549.99 this month but our enterprise plan should be $399.99. Can you help me understand what the additional charges are for? I don't remember upgrading anything, and I need a detailed breakdown for our accounting department. This is affecting our budget planning.",
            "accounting@enterprise.com",
            "medium"
        ),
        
        # Feature requests (fast-track candidates)
        SupportTicket(
            "FEATURE001",
            "Dark mode feature request",
            "Would it be possible to add a dark mode to the application? Many of us work late hours and the current bright interface is quite straining on the eyes. This would be a great addition for user experience and accessibility.",
            "feedback@user.com",
            "low"
        ),
        
        # Refund requests (escalation candidates)
        SupportTicket(
            "REFUND001",
            "Refund request for annual premium subscription",
            "I need a full refund for our annual enterprise subscription ($2,000). The platform hasn't worked as advertised and we're switching to a competitor. We've had multiple unresolved technical issues and poor customer support response times. Please process this refund within 5 business days as per your refund policy.",
            "finance@company.com",
            "high"
        )
    ]


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
        print(f"🤖 Azure OpenAI available: {bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)}")
        
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
        print("  1. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables")
        print("  2. Run: python main.py --demo")
        print("  3. Or start server: python main.py --server")


if __name__ == "__main__":
    main()
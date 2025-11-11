"""
Customer Support Ticket Classifier Package

An AI-powered system using OpenAI for automatically classifying customer support tickets
into predefined categories to improve response times and routing efficiency.
"""

__version__ = "1.0.0"
__author__ = "Jacob Lichty"
__email__ = "jacob.lichty@example.com"

from .models import SupportTicket, TicketRequest, TicketResponse
from .classifier import TicketClassifier
from .agent import TicketClassifierAgent

__all__ = [
    "SupportTicket",
    "TicketRequest", 
    "TicketResponse",
    "TicketClassifier",
    "TicketClassifierAgent"
]
"""
Azure OpenAI-powered ticket classifier.
"""

import json
import logging
from typing import Dict, Optional
import asyncio
from openai import AsyncAzureOpenAI
from .models import SupportTicket
from .config import get_settings

logger = logging.getLogger(__name__)


class TicketClassifier:
    """Azure OpenAI-powered classifier for categorizing support tickets."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, settings=None):
        self.settings = settings or get_settings()
        self.categories = self.settings.ticket_categories
        
        # Initialize Azure OpenAI client
        self.api_key = api_key or self.settings.azure_openai_api_key
        self.endpoint = endpoint or self.settings.azure_openai_endpoint
        
        if not self.api_key or not self.endpoint:
            logger.warning("No Azure OpenAI API key or endpoint provided. Falling back to rule-based classification.")
            self.client = None
        else:
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.settings.azure_openai_api_version,
                timeout=self.settings.openai_timeout,
                max_retries=self.settings.openai_max_retries
            )
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for OpenAI classification."""
        categories_str = ", ".join(self.categories)
        
        category_definitions = {
            "technical_issue": "Problems with software, hardware, bugs, errors, system not working",
            "billing_inquiry": "Questions about charges, payments, invoices, billing cycles",
            "account_management": "Login issues, password resets, account settings, profile changes",
            "feature_request": "Requests for new features, enhancements, suggestions for improvement",
            "general_inquiry": "General questions, information requests, how-to questions",
            "complaint": "Expressions of dissatisfaction, complaints about service or product",
            "refund_request": "Requests for money back, returns, cancellations for refund"
        }
        
        definitions_str = "\n".join([
            f"- {cat}: {desc}" 
            for cat, desc in category_definitions.items() 
            if cat in self.categories
        ])
        
        return f"""You are an AI assistant specialized in classifying customer support tickets.

Your task is to analyze support tickets and classify them into one of these categories:
{categories_str}

For each ticket, you should:
1. Analyze the subject line and content
2. Determine the most appropriate category
3. Provide a confidence score (0.0 to 1.0)
4. Give a brief reasoning for your classification

IMPORTANT: You must respond with ONLY valid JSON in the exact format below. Do not include any text before or after the JSON.

{{
    "category": "category_name",
    "confidence_score": 0.85,
    "reasoning": "Brief explanation of why this category was chosen"
}}

Category definitions:
{definitions_str}

Always choose the most specific and appropriate category based on the primary intent of the ticket.
If you're unsure, use "general_inquiry" with a lower confidence score.

Remember: Return ONLY the JSON object, nothing else."""
    
    async def classify_ticket_with_azure_openai(self, ticket: SupportTicket) -> Dict:
        """Classify ticket using Azure OpenAI API."""
        if not self.client:
            raise ValueError("Azure OpenAI client not initialized")
        
        user_message = f"""Subject: {ticket.subject}
Content: {ticket.content}
Priority: {ticket.priority}
Customer: {ticket.customer_email}"""
        
        try:
            logger.debug(f"Sending ticket {ticket.ticket_id} to Azure OpenAI for classification")
            
            response = await self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens
            )
            
            # Get the response content
            response_content = response.choices[0].message.content
            logger.debug(f"Azure OpenAI raw response for ticket {ticket.ticket_id}: {response_content}")
            
            # Check if response is empty or None
            if not response_content or not response_content.strip():
                logger.warning(f"Azure OpenAI returned empty response for ticket {ticket.ticket_id}")
                raise ValueError("Empty response from Azure OpenAI")
            
            # Parse the JSON response
            result = json.loads(response_content.strip())
            
            # Validate the response
            if result["category"] not in self.categories:
                logger.warning(f"Azure OpenAI returned invalid category: {result['category']}")
                result["category"] = "general_inquiry"
                result["confidence_score"] = 0.5
                result["reasoning"] = "Invalid category returned, defaulted to general_inquiry"
            
            # Ensure confidence score is within valid range
            result["confidence_score"] = max(0.0, min(1.0, result["confidence_score"]))
            
            logger.debug(f"Azure OpenAI classified ticket {ticket.ticket_id} as {result['category']} "
                        f"with confidence {result['confidence_score']}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Azure OpenAI response as JSON for ticket {ticket.ticket_id}: {e}")
            logger.error(f"Raw response content: {response_content}")
            raise ValueError("Invalid JSON response from Azure OpenAI")
        except Exception as e:
            logger.error(f"Azure OpenAI API error for ticket {ticket.ticket_id}: {e}")
            raise
    
    async def classify_ticket(self, ticket: SupportTicket) -> Dict:
        """
        Classify a support ticket using Azure OpenAI or fallback to rule-based.
        
        Args:
            ticket: SupportTicket object to classify
            
        Returns:
            Dictionary with classification results
        """
        if self.client:
            try:
                return await self.classify_ticket_with_azure_openai(ticket)
            except Exception as e:
                logger.warning(f"Azure OpenAI classification failed for ticket {ticket.ticket_id}: {e}. "
                              "Using fallback classification.")
                return self._rule_based_classification(ticket)
        else:
            return self._rule_based_classification(ticket)
    
    def _rule_based_classification(self, ticket: SupportTicket) -> Dict:
        """Simple rule-based classification as fallback."""
        logger.debug(f"Using rule-based classification for ticket {ticket.ticket_id}")
        
        text = f"{ticket.subject} {ticket.content}".lower()
        
        # Define keyword patterns for each category
        patterns = {
            "technical_issue": [
                "bug", "error", "not working", "broken", "crash", "issue", "problem",
                "doesn't work", "won't work", "can't use", "fails", "failure"
            ],
            "billing_inquiry": [
                "bill", "charge", "payment", "invoice", "billing", "paid", "cost",
                "price", "fee", "subscription", "plan"
            ],
            "refund_request": [
                "refund", "money back", "return", "cancel", "cancellation",
                "reimburse", "chargeback"
            ],
            "feature_request": [
                "feature", "suggestion", "enhancement", "improvement", "add",
                "request", "would like", "can you", "please add"
            ],
            "account_management": [
                "account", "login", "password", "reset", "access", "profile",
                "settings", "username", "authentication"
            ],
            "complaint": [
                "complaint", "dissatisfied", "unhappy", "terrible", "awful",
                "disappointed", "frustrated", "angry", "upset"
            ]
        }
        
        # Calculate scores for each category
        scores = {}
        for category, keywords in patterns.items():
            if category in self.categories:
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    scores[category] = min(0.9, 0.6 + (score * 0.1))
        
        # Determine best category
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            return {
                "category": best_category[0],
                "confidence_score": best_category[1],
                "reasoning": f"Rule-based classification based on keyword matching"
            }
        else:
            return {
                "category": "general_inquiry",
                "confidence_score": 0.5,
                "reasoning": "No specific keywords found, defaulting to general inquiry"
            }
    
    async def classify_batch(self, tickets: list[SupportTicket], 
                           max_concurrent: Optional[int] = None) -> list[Dict]:
        """
        Classify multiple tickets concurrently.
        
        Args:
            tickets: List of SupportTicket objects to classify
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of classification results in the same order as input
        """
        max_concurrent = max_concurrent or self.settings.max_concurrent_requests
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_with_semaphore(ticket):
            async with semaphore:
                return await self.classify_ticket(ticket)
        
        # Process all tickets concurrently
        tasks = [classify_with_semaphore(ticket) for ticket in tickets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error classifying ticket {tickets[i].ticket_id}: {result}")
                # Provide fallback result
                processed_results.append({
                    "category": "general_inquiry",
                    "confidence_score": 0.1,
                    "reasoning": f"Classification failed: {str(result)}"
                })
            else:
                processed_results.append(result)
        
        return processed_results
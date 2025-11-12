"""
Main agent for processing support tickets with intelligent routing.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

from .models import SupportTicket
from .classifier import TicketClassifier
from .config import get_settings

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Available processing strategies for intelligent routing."""
    FAST_TRACK = "fast_track"
    THOROUGH_ANALYSIS = "thorough_analysis"
    CONTEXT_ENRICHED = "context_enriched"
    HUMAN_ESCALATION = "human_escalation"
    MULTI_STEP_VALIDATION = "multi_step_validation"


@dataclass
class ProcessingPlan:
    """Plan created by the agent for processing a ticket."""
    strategy: ProcessingStrategy
    estimated_time: int  # seconds
    confidence_threshold: float
    context_requirements: List[str]
    human_review_required: bool
    priority_boost: bool
    reasoning: str


@dataclass
class ProcessingResult:
    """Enhanced result with processing metadata."""
    ticket_id: str
    category: str
    confidence_score: float
    reasoning: str
    processing_strategy_used: ProcessingStrategy
    processing_time: float
    context_gathered: List[str]
    human_review_recommended: bool
    auto_resolution_attempted: bool
    follow_up_scheduled: bool


class TicketClassifierAgent:
    """Main agent for processing support tickets with intelligent routing capabilities."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, settings=None):
        self.settings = settings or get_settings()
        self.classifier = TicketClassifier(api_key, endpoint, self.settings)
        self.processed_tickets = []
        self.startup_time = time.time()
        self.decision_history = []  # Track decision-making for learning
        
        logger.info("Ticket Classifier Agent initialized with intelligent routing")
    
    async def process_ticket(self, ticket: SupportTicket) -> SupportTicket:
        """
        Process a single support ticket using intelligent routing.
        
        Args:
            ticket: SupportTicket object to process
            
        Returns:
            Updated SupportTicket with classification results and intelligent metadata
        """
        logger.info(f"Processing ticket: {ticket.ticket_id}")
        start_time = time.time()
        
        try:
            # Always use intelligent processing
            result = await self._process_intelligently(ticket)
            
            # Update ticket with intelligent processing results
            ticket.category = result.category
            ticket.confidence_score = result.confidence_score
            ticket.reasoning = result.reasoning
            ticket.processed_at = datetime.now()
            
            # Add intelligent processing metadata
            processing_time = time.time() - start_time
            metadata_update = {
                'processing_strategy': result.processing_strategy_used.value,
                'processing_time_seconds': round(processing_time, 3),
                'context_gathered': len(result.context_gathered) > 0,
                'escalation_triggered': result.human_review_recommended,
                'auto_resolution_attempted': result.auto_resolution_attempted,
                'follow_up_scheduled': result.follow_up_scheduled
            }
            
            if hasattr(ticket, 'metadata'):
                ticket.metadata.update(metadata_update)
            else:
                ticket.metadata = metadata_update
            
            processing_time = time.time() - start_time
            logger.info(f"Ticket {ticket.ticket_id} intelligently processed as '{ticket.category}' "
                       f"using {result.processing_strategy_used.value} strategy "
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
    
    async def _process_intelligently(self, ticket: SupportTicket) -> ProcessingResult:
        """
        Process a ticket using intelligent routing and decision-making.
        
        Args:
            ticket: SupportTicket to process
            
        Returns:
            ProcessingResult with enhanced metadata
        """
        start_time = time.time()
        
        # Create intelligent processing plan
        plan = await self._create_processing_plan(ticket)
        logger.debug(f"Created processing plan for {ticket.ticket_id}: {plan.strategy.value}")
        
        # Execute the plan
        classification_result = await self._execute_processing_plan(ticket, plan)
        
        # Gather additional context if needed
        context_gathered = []
        if plan.context_requirements:
            context_gathered = await self._gather_context(ticket, plan.context_requirements)
        
        # Determine post-processing actions
        auto_resolution_attempted = self._attempt_auto_resolution(ticket, plan, classification_result)
        follow_up_scheduled = self._schedule_follow_up(ticket, plan, classification_result)
        human_review_recommended = self._assess_human_review_need(ticket, plan, classification_result)
        
        processing_time = time.time() - start_time
        
        # Create enhanced result
        result = ProcessingResult(
            ticket_id=ticket.ticket_id,
            category=classification_result["category"],
            confidence_score=classification_result["confidence_score"],
            reasoning=classification_result["reasoning"],
            processing_strategy_used=plan.strategy,
            processing_time=processing_time,
            context_gathered=context_gathered,
            human_review_recommended=human_review_recommended,
            auto_resolution_attempted=auto_resolution_attempted,
            follow_up_scheduled=follow_up_scheduled
        )
        
        # Track decision for learning
        self._track_decision(ticket, plan, result)
        
        return result
    
    async def _create_processing_plan(self, ticket: SupportTicket) -> ProcessingPlan:
        """
        Create an intelligent processing plan based on ticket characteristics.
        
        Args:
            ticket: SupportTicket to analyze
            
        Returns:
            ProcessingPlan with strategy and requirements
        """
        # Analyze ticket characteristics using AI
        analysis = await self._analyze_ticket_characteristics(ticket)
        
        # Determine optimal strategy based on AI analysis
        strategy = ProcessingStrategy(analysis["recommended_strategy"])
        estimated_time = analysis["estimated_time"]
        confidence_threshold = analysis["confidence_threshold"]
        context_requirements = analysis["context_requirements"]
        human_review_required = analysis["human_review_required"]
        priority_boost = analysis["priority_boost"]
        reasoning = analysis["reasoning"]
        
        return ProcessingPlan(
            strategy=strategy,
            estimated_time=estimated_time,
            confidence_threshold=confidence_threshold,
            context_requirements=context_requirements,
            human_review_required=human_review_required,
            priority_boost=priority_boost,
            reasoning=reasoning
        )
    
    async def _execute_processing_plan(self, ticket: SupportTicket, plan: ProcessingPlan) -> Dict:
        """
        Execute the processing plan to classify the ticket.
        
        Args:
            ticket: SupportTicket to classify
            plan: ProcessingPlan to execute
            
        Returns:
            Classification result dictionary
        """
        if plan.strategy in [ProcessingStrategy.THOROUGH_ANALYSIS, ProcessingStrategy.MULTI_STEP_VALIDATION]:
            # Use more detailed system prompt for complex analysis
            original_prompt = self.classifier.system_prompt
            enhanced_prompt = self._create_enhanced_prompt(ticket, plan)
            self.classifier.system_prompt = enhanced_prompt
            
            try:
                result = await self.classifier.classify_ticket(ticket)
                # Validate result meets confidence threshold
                if result["confidence_score"] < plan.confidence_threshold:
                    result["reasoning"] += f" (Note: Confidence below threshold {plan.confidence_threshold})"
                return result
            finally:
                self.classifier.system_prompt = original_prompt
        else:
            # Standard classification
            return await self.classifier.classify_ticket(ticket)
    
    async def _analyze_ticket_characteristics(self, ticket: SupportTicket) -> Dict:
        """
        Use AI to analyze ticket characteristics and recommend processing strategy.
        
        Args:
            ticket: SupportTicket to analyze
            
        Returns:
            Dictionary with analysis results and strategy recommendation
        """
        analysis_prompt = f"""
You are an AI assistant that analyzes customer support tickets to determine the optimal processing strategy.

Analyze this ticket and provide a JSON response with the following structure:
{{
    "complexity_score": 0.0-1.0,
    "urgency_score": 0.0-1.0, 
    "sensitivity_score": 0.0-1.0,
    "recommended_strategy": "fast_track|thorough_analysis|context_enriched|human_escalation|multi_step_validation",
    "estimated_time": seconds_as_integer,
    "confidence_threshold": 0.0-1.0,
    "context_requirements": ["list", "of", "requirements"] or [],
    "human_review_required": true/false,
    "priority_boost": true/false,
    "reasoning": "clear explanation of the recommendation"
}}

TICKET DETAILS:
- ID: {ticket.ticket_id}
- Priority: {ticket.priority}
- Subject: {ticket.subject}
- Content: {ticket.content}
- Customer ID: {getattr(ticket, 'customer_id', 'N/A')}

PROCESSING STRATEGIES:
- fast_track: Simple, quick resolution (30-45s, threshold 0.75-0.8)
- thorough_analysis: Complex technical issues (180s, threshold 0.85)
- context_enriched: Needs additional context (120s, threshold 0.8)
- human_escalation: High sensitivity/urgency (300s, threshold 0.95)
- multi_step_validation: Requires multiple validation steps

CONTEXT REQUIREMENTS OPTIONS:
- customer_history, similar_tickets, ticket_history, related_issues, customer_context, product_context

Consider:
1. Technical complexity and language used
2. Emotional tone and urgency indicators
3. Sensitivity (security, legal, privacy, financial)
4. Customer priority level
5. Issue type and potential resolution complexity

Respond with ONLY the JSON object, no additional text.
"""

        try:
            # Check if we have a real OpenAI client
            if not hasattr(self.classifier, 'client') or self.classifier.client is None:
                raise ValueError("No OpenAI client available")
            
            # Use the classifier's OpenAI client for consistency
            response = await self.classifier.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a ticket analysis expert. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content.strip()
            analysis = json.loads(analysis_text)
            
            # Validate the response structure
            required_keys = [
                "complexity_score", "urgency_score", "sensitivity_score",
                "recommended_strategy", "estimated_time", "confidence_threshold",
                "context_requirements", "human_review_required", "priority_boost", "reasoning"
            ]
            
            for key in required_keys:
                if key not in analysis:
                    raise ValueError(f"Missing required key: {key}")
            
            # Validate strategy is valid
            valid_strategies = [strategy.value for strategy in ProcessingStrategy]
            if analysis["recommended_strategy"] not in valid_strategies:
                logger.warning(f"Invalid strategy '{analysis['recommended_strategy']}', defaulting to fast_track")
                analysis["recommended_strategy"] = "fast_track"
            
            logger.debug(f"AI analysis for {ticket.ticket_id}: {analysis['recommended_strategy']} "
                        f"(complexity: {analysis['complexity_score']}, "
                        f"urgency: {analysis['urgency_score']}, "
                        f"sensitivity: {analysis['sensitivity_score']})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI ticket analysis: {e}")
            # Fallback to simplified rule-based analysis
            return self._fallback_ticket_analysis(ticket)
    
    def _fallback_ticket_analysis(self, ticket: SupportTicket) -> Dict:
        """
        Fallback analysis when AI is not available - uses simple rule-based logic.
        
        Args:
            ticket: SupportTicket to analyze
            
        Returns:
            Dictionary with analysis results and strategy recommendation
        """
        text = f"{ticket.subject} {ticket.content}".lower()
        priority = ticket.priority.lower()
        
        # Simple complexity scoring
        complexity_score = 0.0
        if len(text) > 500:
            complexity_score += 0.4
        elif len(text) > 200:
            complexity_score += 0.2
        
        technical_words = ["api", "database", "server", "error", "integration", "authentication"]
        complexity_score += min(0.4, len([w for w in technical_words if w in text]) * 0.1)
        
        # Simple urgency scoring
        urgency_score = {"urgent": 0.9, "high": 0.7, "medium": 0.4, "low": 0.2}.get(priority, 0.3)
        urgent_words = ["urgent", "critical", "down", "broken", "immediately"]
        if any(word in text for word in urgent_words):
            urgency_score += 0.2
        
        # Simple sensitivity scoring
        sensitivity_score = 0.0
        sensitive_words = ["security", "legal", "privacy", "breach", "fraud", "confidential"]
        sensitivity_score = min(0.8, len([w for w in sensitive_words if w in text]) * 0.2)
        
        # Determine strategy based on scores
        if sensitivity_score > 0.4 or urgency_score > 0.8:
            strategy = "human_escalation"
            estimated_time = 300
            confidence_threshold = 0.95
            context_requirements = ["customer_history", "similar_tickets"]
            human_review_required = True
            priority_boost = True
            reasoning = "High sensitivity or urgency detected (fallback analysis)"
        elif complexity_score > 0.6:
            strategy = "thorough_analysis"
            estimated_time = 180
            confidence_threshold = 0.85
            context_requirements = ["ticket_history"]
            human_review_required = False
            priority_boost = urgency_score > 0.6
            reasoning = "Complex ticket requiring thorough analysis (fallback analysis)"
        elif urgency_score > 0.5 and complexity_score < 0.4:
            strategy = "fast_track"
            estimated_time = 30
            confidence_threshold = 0.8
            context_requirements = []
            human_review_required = False
            priority_boost = False
            reasoning = "Simple urgent ticket (fallback analysis)"
        else:
            strategy = "fast_track"
            estimated_time = 45
            confidence_threshold = 0.75
            context_requirements = []
            human_review_required = False
            priority_boost = False
            reasoning = "Standard ticket processing (fallback analysis)"
        
        return {
            "complexity_score": min(1.0, complexity_score),
            "urgency_score": min(1.0, urgency_score),
            "sensitivity_score": min(1.0, sensitivity_score),
            "recommended_strategy": strategy,
            "estimated_time": estimated_time,
            "confidence_threshold": confidence_threshold,
            "context_requirements": context_requirements,
            "human_review_required": human_review_required,
            "priority_boost": priority_boost,
            "reasoning": reasoning
        }

    async def _gather_context(self, ticket: SupportTicket, requirements: List[str]) -> List[str]:
        """Gather additional context as specified in requirements."""
        context = []
        
        for requirement in requirements:
            if requirement == "customer_history":
                context.append("Customer history: Premium user for 2+ years")
            elif requirement == "similar_tickets":
                context.append("Similar tickets: 3 related issues in past month")
            elif requirement == "ticket_history":
                context.append("Ticket history: First occurrence of this issue type")
            elif requirement == "related_issues":
                context.append("Related issues: System maintenance scheduled")
            elif requirement == "customer_context":
                context.append("Customer context: Enterprise account with SLA")
            elif requirement == "product_context":
                context.append("Product context: Recent feature update deployed")
        
        return context
    
    def _create_enhanced_prompt(self, ticket: SupportTicket, plan: ProcessingPlan) -> str:
        """Create an enhanced system prompt for complex analysis."""
        base_prompt = self.classifier.system_prompt
        
        enhancement = f"""

ENHANCED ANALYSIS MODE - {plan.strategy.value.upper()}
This ticket requires {plan.reasoning.lower()}

Additional considerations:
- Minimum confidence threshold: {plan.confidence_threshold}
- Estimated processing time: {plan.estimated_time}s
- Context requirements: {', '.join(plan.context_requirements) if plan.context_requirements else 'None'}

Please provide extra thorough analysis and reasoning for this classification."""
        
        return base_prompt + enhancement
    
    def _attempt_auto_resolution(self, ticket: SupportTicket, plan: ProcessingPlan, result: Dict) -> bool:
        """Determine if auto-resolution should be attempted."""
        if plan.strategy == ProcessingStrategy.FAST_TRACK and result["confidence_score"] > 0.9:
            simple_categories = ["account_management", "billing_inquiry", "general_inquiry"]
            return result["category"] in simple_categories
        return False
    
    def _schedule_follow_up(self, ticket: SupportTicket, plan: ProcessingPlan, result: Dict) -> bool:
        """Determine if follow-up should be scheduled."""
        if ticket.priority.lower() in ["high", "urgent"]:
            return True
        if result["category"] in ["complaint", "refund_request"]:
            return True
        return False
    
    def _assess_human_review_need(self, ticket: SupportTicket, plan: ProcessingPlan, result: Dict) -> bool:
        """Assess if human review is needed."""
        if plan.human_review_required:
            return True
        if result["confidence_score"] < 0.7:
            return True
        if result["category"] in ["complaint", "refund_request"] and ticket.priority.lower() == "high":
            return True
        return False
    
    def _track_decision(self, ticket: SupportTicket, plan: ProcessingPlan, result: ProcessingResult):
        """Track decision for learning and improvement."""
        decision_record = {
            "ticket_id": ticket.ticket_id,
            "timestamp": datetime.now().isoformat(),
            "strategy_chosen": plan.strategy.value,
            "confidence_achieved": result.confidence_score,
            "processing_time": result.processing_time,
            "category": result.category,
            "human_review_triggered": result.human_review_recommended
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
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
        strategy_counts = {}
        escalation_count = 0
        intelligent_processed = 0
        
        for ticket in self.processed_tickets:
            category_counts[ticket.category] = category_counts.get(ticket.category, 0) + 1
            if ticket.confidence_score is not None:
                confidence_scores.append(ticket.confidence_score)
            
            # Track intelligent routing statistics
            if hasattr(ticket, 'metadata') and isinstance(ticket.metadata, dict):
                # All tickets are now processed intelligently
                intelligent_processed += 1
                strategy = ticket.metadata.get('processing_strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                if ticket.metadata.get('escalation_triggered'):
                    escalation_count += 1
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        stats = {
            "total_processed": len(self.processed_tickets),
            "category_distribution": category_counts,
            "average_confidence": round(avg_confidence, 3),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "categories_available": self.settings.ticket_categories,
            "intelligent_processed": intelligent_processed,
            "strategy_distribution": strategy_counts,
            "escalations_triggered": escalation_count,
            "intelligent_processing_rate": round(intelligent_processed / len(self.processed_tickets), 3) if self.processed_tickets else 0.0,
            "total_decisions_tracked": len(self.decision_history)
        }
        
        return stats
    
    def get_health_status(self) -> Dict:
        """Get health status information."""
        health_status = {
            "status": "healthy",
            "openai_available": bool(self.classifier.client),
            "total_processed": len(self.processed_tickets),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "decision_history_size": len(self.decision_history),
            "settings": {
                "max_batch_size": self.settings.max_batch_size,
                "max_concurrent_requests": self.settings.max_concurrent_requests,
                "openai_model": self.settings.openai_model
            },
            "intelligent_settings": {
                "learning_enabled": getattr(self.settings, 'intelligent_agent_learning', True),
                "confidence_threshold": getattr(self.settings, 'intelligent_confidence_threshold', 0.8),
                "max_context_time": getattr(self.settings, 'max_context_gathering_time', 30),
                "strategies_available": [strategy.value for strategy in ProcessingStrategy]
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
    
    def get_decision_insights(self) -> Dict:
        """Get insights from the intelligent decision-making history."""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "average_confidence": 0.0,
                "average_processing_time": 0.0,
                "strategy_effectiveness": {},
                "human_review_rate": 0.0
            }
        
        # Calculate metrics
        total_decisions = len(self.decision_history)
        avg_confidence = sum(d["confidence_achieved"] for d in self.decision_history) / total_decisions
        avg_processing_time = sum(d["processing_time"] for d in self.decision_history) / total_decisions
        human_reviews = sum(1 for d in self.decision_history if d["human_review_triggered"])
        
        # Strategy effectiveness
        strategy_stats = {}
        for decision in self.decision_history:
            strategy = decision["strategy_chosen"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_time": 0.0,
                    "human_review_rate": 0.0
                }
            
            stats = strategy_stats[strategy]
            stats["count"] += 1
            stats["avg_confidence"] += decision["confidence_achieved"]
            stats["avg_time"] += decision["processing_time"]
            if decision["human_review_triggered"]:
                stats["human_review_rate"] += 1
        
        # Finalize averages
        for strategy, stats in strategy_stats.items():
            count = stats["count"]
            stats["avg_confidence"] = round(stats["avg_confidence"] / count, 3)
            stats["avg_time"] = round(stats["avg_time"] / count, 3)
            stats["human_review_rate"] = round(stats["human_review_rate"] / count, 3)
        
        return {
            "total_decisions": total_decisions,
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "strategy_effectiveness": strategy_stats,
            "human_review_rate": round(human_reviews / total_decisions, 3),
            "most_used_strategy": max(strategy_stats.keys(), key=lambda k: strategy_stats[k]["count"]) if strategy_stats else None
        }
"""
Hybrid DRL + LLM Optimizer

===== STEP 2: P-MODEL EXECUTION =====
Data Flow: CampaignState + CampaignContext -> SafeDRLAgent -> HybridDRLLLMOptimizer -> OptimizationResult

Implements hierarchical optimization architecture:
- DRL Macro Layer: Strategic decisions (budget, bids, audience priority)
- LLM Micro Layer: Tactical execution (creative, messaging, offers)

The DRL provides constraints and directives that guide LLM generation,
ensuring tactical execution aligns with strategic optimization goals.
"""
# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from enum import Enum

from .config import DRLConfig, GuardrailConfig
from .sac_agent import SACAgent
from .safe_agent import SafeDRLAgent, CampaignContext, ActionValidationResult
from .state_action import CampaignState, ActionSpace, DRLDirective
from .xai_narrator import OptimizationNarrator, RunNarrative
from .benchmark_model import CampaignForecaster, CampaignForecast

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization decisions"""
    BUDGET_ALLOCATION = "budget_allocation"
    BID_STRATEGY = "bid_strategy"
    AUDIENCE_TARGETING = "audience_targeting"
    CREATIVE_OPTIMIZATION = "creative_optimization"
    MESSAGING = "messaging"
    OFFER_GENERATION = "offer_generation"


@dataclass
class TacticalExecution:
    """LLM-generated tactical execution"""
    headline: str = ""
    body_copy: str = ""
    call_to_action: str = ""
    offer_text: Optional[str] = None
    product_highlights: List[str] = field(default_factory=list)
    urgency_elements: List[str] = field(default_factory=list)
    personalization_tokens: Dict[str, str] = field(default_factory=dict)
    
    # Generation metadata
    model_used: str = ""
    generation_time_ms: float = 0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "body_copy": self.body_copy,
            "call_to_action": self.call_to_action,
            "offer_text": self.offer_text,
            "product_highlights": self.product_highlights,
            "urgency_elements": self.urgency_elements,
            "personalization_tokens": self.personalization_tokens,
            "metadata": {
                "model_used": self.model_used,
                "generation_time_ms": self.generation_time_ms,
                "tokens_used": self.tokens_used,
            }
        }


@dataclass
class OptimizationResult:
    """Complete optimization result combining DRL strategy and LLM tactics"""
    # Strategic (DRL)
    directive: DRLDirective
    action: ActionSpace
    validation: ActionValidationResult
    
    # Tactical (LLM)
    tactical: Optional[TacticalExecution] = None
    
    # Combined metrics
    strategic_confidence: float = 0.0
    tactical_confidence: float = 0.0
    combined_confidence: float = 0.0

    # Recommendations
    recommended_changes: List[Dict[str, Any]] = field(default_factory=list)
    requires_review: bool = False

    # xAI narrative (human-readable explanation of the DRL decision)
    narrative: Optional[Dict[str, Any]] = None

    # Campaign outcome forecast (predicted results if recommendations are applied)
    forecast: Optional[Dict[str, Any]] = None

    # Timing
    timestamp: str = ""
    total_latency_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategic": {
                "directive": self.directive.to_dict(),
                "action": self.action.to_dict(),
                "validation_status": self.validation.status.value,
            },
            "tactical": self.tactical.to_dict() if self.tactical else None,
            "confidence": {
                "strategic": self.strategic_confidence,
                "tactical": self.tactical_confidence,
                "combined": self.combined_confidence,
            },
            "recommended_changes": self.recommended_changes,
            "requires_review": self.requires_review,
            "narrative": self.narrative,
            "forecast": self.forecast,
            "timestamp": self.timestamp,
            "total_latency_ms": self.total_latency_ms,
        }


class LLMClient:
    """
    Client for LLM API calls (placeholder for actual implementation)
    
    In production, this would connect to:
    - OpenAI GPT-4
    - Anthropic Claude
    - Custom fine-tuned models
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        # Placeholder implementation
        # In production, this would call actual LLM API
        
        start_time = datetime.now(timezone.utc)
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        # Mock response
        response = self._generate_mock_response(prompt)
        
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        metadata = {
            "model": self.model,
            "latency_ms": latency_ms,
            "tokens_used": len(response.split()) * 1.3,  # Rough estimate
        }
        
        return response, metadata
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing"""
        if "headline" in prompt.lower():
            return json.dumps({
                "headline": "Don't Miss Out - Limited Time Offer!",
                "body_copy": "Discover premium quality at unbeatable prices. Shop now and save big on your favorite items.",
                "call_to_action": "Shop Now",
                "offer_text": "Save 15% Today",
                "product_highlights": ["Premium Quality", "Fast Shipping", "Easy Returns"],
                "urgency_elements": ["Limited Time", "While Supplies Last"]
            })
        return "{}"


class HybridDRLLLMOptimizer:
    """
    [HybridDRLLLMOptimizer]
    Description: Combines DRL strategic decisions with LLM tactical execution (creative, messaging).
    Input: CampaignState (from metrics), CampaignContext (from campaign API), campaign_info dict.
    Output: OptimizationResult -> consumed by API routes, BatchOptimizer, demos.
    
    Architecture:
        User Request
            ↓
        DRL Macro Layer (Strategic)
        ├─ Budget Allocation
        ├─ Bid Strategy  
        └─ Audience Priority
            ↓
        Strategic Directive
            ↓
        LLM Micro Layer (Tactical)
        ├─ Creative Generation
        ├─ Messaging Optimization
        └─ Offer Personalization
            ↓
        Combined Optimization Result
    """
    
    def __init__(
        self,
        drl_agent: SafeDRLAgent,
        llm_client: Optional[LLMClient] = None,
        enable_tactical: bool = True,
        forecaster: Optional[CampaignForecaster] = None,
        narrator: Optional[OptimizationNarrator] = None,
    ):
        """
        Args:
            drl_agent: Safe DRL agent for strategic decisions
            llm_client: LLM client for tactical generation
            enable_tactical: Whether to enable LLM tactical layer
            forecaster: Optional campaign outcome forecaster
            narrator: Optional xAI narrative generator
        """
        self.drl_agent = drl_agent
        self.llm_client = llm_client or LLMClient()
        self.enable_tactical = enable_tactical
        self.forecaster = forecaster
        self.narrator = narrator or OptimizationNarrator()

        # Prompt templates
        self.system_prompt = self._build_system_prompt()

        logger.info("HybridDRLLLMOptimizer initialized")
    
    async def optimize(
        self,
        state: CampaignState,
        context: CampaignContext,
        campaign_info: Dict[str, Any],
        generate_tactical: bool = True
    ) -> OptimizationResult:
        """
        Run full optimization pipeline
        
        Args:
            state: Current campaign state
            context: Campaign context
            campaign_info: Additional campaign information
            generate_tactical: Whether to generate tactical content
            
        Returns:
            Complete OptimizationResult
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT: optimize() | state.ctr={state.ctr:.4f}, state.roas={state.roas:.4f}")
            logger.info(f"[IO] INPUT: context.campaign_id={context.campaign_id}, current_bid={context.current_bid}, current_budget={context.current_budget}")
            logger.info(f"[IO] INPUT: campaign_info keys={list(campaign_info.keys()) if campaign_info else []}, generate_tactical={generate_tactical}")
        start_time = datetime.now(timezone.utc)
        
        # Phase 1: DRL Strategic Decision
        action, validation = await self.drl_agent.get_action(
            state=state,
            context=context,
            exploration=True
        )
        
        # Create directive from action
        directive = DRLDirective.from_action(
            action=action,
            state=state,
            campaign_context=campaign_info
        )
        
        # Phase 2: LLM Tactical Execution (if enabled)
        tactical = None
        tactical_confidence = 1.0
        
        if self.enable_tactical and generate_tactical:
            tactical, tactical_confidence = await self._generate_tactical(
                directive=directive,
                campaign_info=campaign_info,
                state=state
            )
        
        # Compute combined confidence
        strategic_confidence = action.confidence
        combined_confidence = self._compute_combined_confidence(
            strategic_confidence,
            tactical_confidence
        )
        
        # Build recommended changes
        recommended_changes = self._build_recommendations(
            action=action,
            directive=directive,
            tactical=tactical,
            context=context
        )
        
        # Determine if review needed
        requires_review = (
            validation.requires_human_review or
            combined_confidence < 0.6
        )

        # Phase 3: xAI Narrative (explain the DRL decision in plain English)
        narrative_dict = None
        try:
            run_narrative = self.narrator.generate_run_narrative(
                state=state,
                action=action,
                directive=directive,
                recommendations=recommended_changes,
            )
            narrative_dict = {
                "situation": run_narrative.situation_summary,
                "decision": run_narrative.decision_summary,
                "reasoning": run_narrative.reasoning,
                "confidence": run_narrative.confidence_explanation,
                "reasonability": run_narrative.reasonability_check,
                "full": run_narrative.full_narrative,
            }
        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")

        # Phase 4: Campaign Outcome Forecast
        forecast_dict = None
        if self.forecaster is not None:
            try:
                forecast = self.forecaster.predict(state.to_tensor().numpy())
                forecast_dict = forecast.to_dict()
            except RuntimeError:
                # Forecaster not yet fitted — skip silently
                pass
            except Exception as e:
                logger.warning(f"Forecast generation failed: {e}")

        end_time = datetime.now(timezone.utc)
        total_latency = (end_time - start_time).total_seconds() * 1000

        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] OUTPUT: OptimizationResult | action.bid_adj={action.bid_adjustment:.4f}, budget_adj={action.budget_adjustment:.4f}")
            logger.info(f"[IO] OUTPUT: validation_status={validation.status.value}, combined_confidence={combined_confidence:.4f}")
            logger.info(f"[IO] OUTPUT: Next: API routes, BatchOptimizer.optimize_batch(), demos")
        return OptimizationResult(
            directive=directive,
            action=action,
            validation=validation,
            tactical=tactical,
            strategic_confidence=strategic_confidence,
            tactical_confidence=tactical_confidence,
            combined_confidence=combined_confidence,
            recommended_changes=recommended_changes,
            requires_review=requires_review,
            narrative=narrative_dict,
            forecast=forecast_dict,
            timestamp=start_time.isoformat(),
            total_latency_ms=total_latency,
        )
    
    async def _generate_tactical(
        self,
        directive: DRLDirective,
        campaign_info: Dict[str, Any],
        state: CampaignState
    ) -> Tuple[TacticalExecution, float]:
        """Generate tactical content using LLM"""
        
        # Build prompt with DRL constraints
        prompt = self._build_tactical_prompt(
            directive=directive,
            campaign_info=campaign_info,
            state=state
        )
        
        try:
            response, metadata = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            
            # Parse response
            tactical = self._parse_tactical_response(response, metadata)
            
            # Compute confidence based on response quality
            confidence = self._assess_tactical_quality(tactical, directive)
            
            return tactical, confidence
            
        except Exception as e:
            logger.error(f"Tactical generation failed: {e}")
            return TacticalExecution(), 0.5
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are an expert advertising copywriter and conversion optimization specialist.
Your role is to generate high-converting ad creative that aligns with strategic directives.

Guidelines:
- Follow the messaging tone specified in the directive
- Respect discount and offer limits
- Create compelling, action-oriented copy
- Use urgency appropriately based on urgency level
- Personalize for the target audience segment
- Keep headlines under 60 characters
- Keep body copy concise and scannable

Output format: JSON with keys: headline, body_copy, call_to_action, offer_text, product_highlights, urgency_elements
"""
    
    def _build_tactical_prompt(
        self,
        directive: DRLDirective,
        campaign_info: Dict[str, Any],
        state: CampaignState
    ) -> str:
        """Build prompt for tactical generation"""
        
        # Include directive context
        directive_context = directive.to_llm_prompt_context()
        
        # Campaign details
        campaign_context = f"""
Campaign Details:
- Product/Service: {campaign_info.get('product_name', 'Product')}
- Brand: {campaign_info.get('brand_name', 'Brand')}
- Target Audience: {campaign_info.get('target_audience', 'General consumers')}
- Value Proposition: {campaign_info.get('value_prop', 'Quality and value')}
- Current Performance: CTR {state.ctr:.2%}, CVR {state.cvr:.2%}
"""
        
        # Task instruction
        task = f"""
Based on the strategic directive and campaign context above, generate compelling ad creative.

Requirements:
1. Headline: Attention-grabbing, under 60 characters
2. Body Copy: 2-3 sentences highlighting key benefits
3. Call to Action: Clear, action-oriented button text
4. Offer Text: If discount allowed (max {directive.max_offer_discount:.0%}), create offer text
5. Product Highlights: 3 key selling points
6. Urgency Elements: If urgency level > 50%, add urgency phrases

Respond with JSON only.
"""
        
        return f"{directive_context}\n{campaign_context}\n{task}"
    
    def _parse_tactical_response(
        self,
        response: str,
        metadata: Dict[str, Any]
    ) -> TacticalExecution:
        """Parse LLM response into TacticalExecution"""
        try:
            data = json.loads(response)
            return TacticalExecution(
                headline=data.get("headline", ""),
                body_copy=data.get("body_copy", ""),
                call_to_action=data.get("call_to_action", ""),
                offer_text=data.get("offer_text"),
                product_highlights=data.get("product_highlights", []),
                urgency_elements=data.get("urgency_elements", []),
                model_used=metadata.get("model", ""),
                generation_time_ms=metadata.get("latency_ms", 0),
                tokens_used=int(metadata.get("tokens_used", 0)),
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return TacticalExecution(
                headline=response[:60] if response else "",
                model_used=metadata.get("model", ""),
            )
    
    def _assess_tactical_quality(
        self,
        tactical: TacticalExecution,
        directive: DRLDirective
    ) -> float:
        """Assess quality of tactical generation"""
        score = 1.0
        
        # Check required fields
        if not tactical.headline:
            score -= 0.3
        if not tactical.body_copy:
            score -= 0.2
        if not tactical.call_to_action:
            score -= 0.2
        
        # Check headline length
        if len(tactical.headline) > 60:
            score -= 0.1
        
        # Check tone alignment (simplified)
        if directive.urgency_level > 0.7 and not tactical.urgency_elements:
            score -= 0.1
        
        return max(0.0, score)
    
    def _compute_combined_confidence(
        self,
        strategic: float,
        tactical: float
    ) -> float:
        """Compute combined confidence from strategic and tactical"""
        # Weighted combination favoring strategic
        return 0.7 * strategic + 0.3 * tactical
    
    def _build_recommendations(
        self,
        action: ActionSpace,
        directive: DRLDirective,
        tactical: Optional[TacticalExecution],
        context: CampaignContext
    ) -> List[Dict[str, Any]]:
        """Build list of recommended changes"""
        recommendations = []
        
        # Bid recommendation
        if abs(action.bid_adjustment) > 0.05:
            new_bid = context.current_bid * (1 + action.bid_adjustment)
            recommendations.append({
                "type": "bid_adjustment",
                "current_value": context.current_bid,
                "recommended_value": new_bid,
                "change_percent": action.bid_adjustment,
                "rationale": self._get_bid_rationale(action, directive),
            })
        
        # Budget recommendation
        if abs(action.budget_adjustment) > 0.05:
            new_budget = context.current_budget * (1 + action.budget_adjustment)
            recommendations.append({
                "type": "budget_adjustment",
                "current_value": context.current_budget,
                "recommended_value": new_budget,
                "change_percent": action.budget_adjustment,
                "rationale": self._get_budget_rationale(action, directive),
            })
        
        # Audience recommendation
        if action.audience_action != 0:  # Not HOLD
            recommendations.append({
                "type": "audience_optimization",
                "action": directive.audience_priority,
                "rationale": self._get_audience_rationale(action),
            })
        
        # Creative recommendation
        if action.creative_action != 0:  # Not HOLD
            creative_rec = {
                "type": "creative_optimization",
                "action": directive.creative_direction,
                "rationale": self._get_creative_rationale(action),
            }
            if tactical:
                creative_rec["suggested_creative"] = {
                    "headline": tactical.headline,
                    "body": tactical.body_copy,
                    "cta": tactical.call_to_action,
                }
            recommendations.append(creative_rec)
        
        return recommendations
    
    def _get_bid_rationale(self, action: ActionSpace, directive: DRLDirective) -> str:
        """Generate rationale for bid recommendation"""
        if action.bid_adjustment > 0.1:
            return f"Increase bid to capture more impression share. Tone: {directive.messaging_tone}"
        elif action.bid_adjustment < -0.1:
            return "Reduce bid to improve efficiency and lower CPA"
        else:
            return "Minor bid adjustment to optimize position"
    
    def _get_budget_rationale(self, action: ActionSpace, directive: DRLDirective) -> str:
        """Generate rationale for budget recommendation"""
        if action.budget_adjustment > 0.1:
            return "Increase budget to scale successful performance"
        elif action.budget_adjustment < -0.1:
            return "Reduce budget due to underperformance or efficiency concerns"
        else:
            return "Minor budget adjustment for optimization"
    
    def _get_audience_rationale(self, action: ActionSpace) -> str:
        """Generate rationale for audience recommendation"""
        from .state_action import AudienceAction
        audience = AudienceAction(action.audience_action)
        
        rationales = {
            AudienceAction.HOLD: "Maintain current audience targeting",
            AudienceAction.EXPAND: "Expand audience to increase reach and volume",
            AudienceAction.REFINE: "Refine audience to improve conversion rate",
            AudienceAction.EXCLUDE: "Exclude underperforming segments",
        }
        return rationales.get(audience, "Optimize audience targeting")
    
    def _get_creative_rationale(self, action: ActionSpace) -> str:
        """Generate rationale for creative recommendation"""
        from .state_action import CreativeAction
        creative = CreativeAction(action.creative_action)
        
        rationales = {
            CreativeAction.HOLD: "Maintain current creative rotation",
            CreativeAction.ROTATE: "Rotate creatives to reduce fatigue",
            CreativeAction.PAUSE_UNDERPERFORMING: "Pause underperforming creatives",
            CreativeAction.TEST_NEW: "Test new creative variants",
        }
        return rationales.get(creative, "Optimize creative strategy")


class BatchOptimizer:
    """
    [BatchOptimizer]
    Description: Runs HybridDRLLLMOptimizer.optimize() on multiple campaigns in parallel.
    Input: List of (CampaignState, CampaignContext, campaign_info) tuples.
    Output: List[OptimizationResult] -> consumed by batch API, reporting.
    """
    
    def __init__(
        self,
        hybrid_optimizer: HybridDRLLLMOptimizer,
        max_concurrent: int = 10
    ):
        self.optimizer = hybrid_optimizer
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def optimize_batch(
        self,
        campaigns: List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]]
    ) -> List[OptimizationResult]:
        """
        Optimize multiple campaigns in parallel
        
        Args:
            campaigns: List of (state, context, info) tuples
            
        Returns:
            List of OptimizationResult
        """
        # ----- INPUT LOGGING -----
        if _QA_IO_LOGGING:
            logger.info(f"[IO] INPUT: optimize_batch() | {len(campaigns)} campaigns")
        async def optimize_with_semaphore(
            state: CampaignState,
            context: CampaignContext,
            info: Dict[str, Any]
        ) -> OptimizationResult:
            async with self.semaphore:
                return await self.optimizer.optimize(
                    state=state,
                    context=context,
                    campaign_info=info,
                    generate_tactical=True
                )
        
        tasks = [
            optimize_with_semaphore(state, context, info)
            for state, context, info in campaigns
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        # ----- OUTPUT LOGGING -----
        if _QA_IO_LOGGING:
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"[IO] OUTPUT: optimize_batch() | {success_count}/{len(campaigns)} succeeded | Next: batch API, reporting")
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Campaign {i} optimization failed: {result}")
                # Create empty result
                final_results.append(OptimizationResult(
                    directive=DRLDirective(),
                    action=ActionSpace(),
                    validation=ActionValidationResult(
                        original_action=ActionSpace(),
                        validated_action=ActionSpace(),
                        status="error",
                        blocking_reason=str(result)
                    ),
                    requires_review=True,
                ))
            else:
                final_results.append(result)
        
        return final_results

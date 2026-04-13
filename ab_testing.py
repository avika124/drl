"""
A/B Testing Framework for DRL Evaluation

Implements:
- Experiment assignment and bucketing
- Statistical significance testing
- Lift calculation with confidence intervals
- Experiment lifecycle management
- Results analysis and reporting
"""

import hashlib
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

from .xai_narrator import ParameterGlossary

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class AssignmentMethod(Enum):
    """Method for variant assignment"""
    DETERMINISTIC = "deterministic"  # Hash-based, stable
    RANDOM = "random"               # Random assignment
    STRATIFIED = "stratified"       # Stratified by attribute


@dataclass
class ExperimentVariant:
    """Configuration for an experiment variant"""
    name: str
    description: str = ""
    allocation_percent: float = 50.0
    is_control: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allocation_percent": self.allocation_percent,
            "is_control": self.is_control,
            "config": self.config,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment"""
    experiment_id: str
    name: str
    description: str = ""
    
    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)
    
    # Assignment
    assignment_method: AssignmentMethod = AssignmentMethod.DETERMINISTIC
    assignment_key: str = "campaign_id"  # Field to hash for assignment
    
    # Duration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_duration_days: int = 14
    
    # Statistical requirements
    target_sample_size: int = 1000
    min_detectable_effect: float = 0.05  # 5% MDE
    significance_level: float = 0.05     # 95% confidence
    power: float = 0.8                   # 80% power
    
    # Metrics
    primary_metric: str = "roas"
    secondary_metrics: List[str] = field(default_factory=lambda: ["cpa", "conversions", "ctr"])
    
    # Guardrails
    guardrail_metrics: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # metric: (min, max)
    early_stopping_threshold: float = 0.01  # Stop if p-value below this (very significant)
    
    def __post_init__(self):
        if not self.variants:
            # Default: 80% baseline, 20% DRL
            self.variants = [
                ExperimentVariant(
                    name="control",
                    description="Rule-based baseline",
                    allocation_percent=80.0,
                    is_control=True,
                ),
                ExperimentVariant(
                    name="treatment",
                    description="DRL optimization",
                    allocation_percent=20.0,
                    is_control=False,
                ),
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "assignment_method": self.assignment_method.value,
            "assignment_key": self.assignment_key,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "min_duration_days": self.min_duration_days,
            "target_sample_size": self.target_sample_size,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "significance_level": self.significance_level,
        }


@dataclass
class VariantMetrics:
    """Collected metrics for a variant"""
    variant_name: str
    sample_size: int = 0
    
    # Aggregated metrics
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_conversions: int = 0
    total_clicks: int = 0
    total_impressions: int = 0
    
    # Computed metrics
    roas: float = 0.0
    cpa: float = 0.0
    ctr: float = 0.0
    cvr: float = 0.0
    
    # Distribution for statistical tests
    metric_values: List[float] = field(default_factory=list)
    metric_variance: float = 0.0
    
    def compute_metrics(self):
        """Compute derived metrics from totals"""
        if self.total_spend > 0:
            self.roas = self.total_revenue / self.total_spend
        if self.total_conversions > 0:
            self.cpa = self.total_spend / self.total_conversions
        if self.total_impressions > 0:
            self.ctr = self.total_clicks / self.total_impressions
        if self.total_clicks > 0:
            self.cvr = self.total_conversions / self.total_clicks
        
        if self.metric_values:
            self.metric_variance = np.var(self.metric_values)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "sample_size": self.sample_size,
            "totals": {
                "spend": self.total_spend,
                "revenue": self.total_revenue,
                "conversions": self.total_conversions,
                "clicks": self.total_clicks,
                "impressions": self.total_impressions,
            },
            "metrics": {
                "roas": self.roas,
                "cpa": self.cpa,
                "ctr": self.ctr,
                "cvr": self.cvr,
            },
            "metric_variance": self.metric_variance,
        }


@dataclass
class StatisticalResult:
    """Result of statistical significance test"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    confidence_level: float
    test_type: str = "welch_t_test"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "absolute_lift": self.absolute_lift,
            "relative_lift": self.relative_lift,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "test_type": self.test_type,
        }


@dataclass
class ExperimentResult:
    """Complete results of an A/B experiment"""
    experiment_id: str
    status: ExperimentStatus
    
    # Duration
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    
    # Variant metrics
    variant_metrics: Dict[str, VariantMetrics] = field(default_factory=dict)
    
    # Statistical results
    primary_result: Optional[StatisticalResult] = None
    secondary_results: Dict[str, StatisticalResult] = field(default_factory=dict)
    
    # Guardrail results
    guardrail_violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = ""
    confidence_in_recommendation: float = 0.0

    # Human-readable narrative explaining the results
    narrative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "duration": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "duration_days": self.duration_days,
            },
            "variant_metrics": {k: v.to_dict() for k, v in self.variant_metrics.items()},
            "primary_result": self.primary_result.to_dict() if self.primary_result else None,
            "secondary_results": {k: v.to_dict() for k, v in self.secondary_results.items()},
            "guardrail_violations": self.guardrail_violations,
            "recommendation": self.recommendation,
            "confidence_in_recommendation": self.confidence_in_recommendation,
        }
        if self.narrative is not None:
            d["narrative"] = self.narrative
        return d


class VariantAssigner:
    """
    Assigns entities to experiment variants
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._build_allocation_ranges()
    
    def _build_allocation_ranges(self):
        """Build allocation ranges from percentages"""
        self.allocation_ranges = []
        cumulative = 0.0
        
        for variant in self.config.variants:
            start = cumulative
            end = cumulative + variant.allocation_percent
            self.allocation_ranges.append((start, end, variant.name))
            cumulative = end
    
    def assign(self, entity_id: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Assign entity to variant
        
        Args:
            entity_id: Entity identifier (e.g., campaign_id)
            attributes: Optional attributes for stratified assignment
            
        Returns:
            Variant name
        """
        if self.config.assignment_method == AssignmentMethod.DETERMINISTIC:
            return self._deterministic_assign(entity_id)
        elif self.config.assignment_method == AssignmentMethod.RANDOM:
            return self._random_assign()
        elif self.config.assignment_method == AssignmentMethod.STRATIFIED:
            return self._stratified_assign(entity_id, attributes or {})
        else:
            return self._deterministic_assign(entity_id)
    
    def _deterministic_assign(self, entity_id: str) -> str:
        """Hash-based deterministic assignment"""
        # Create hash from experiment_id + entity_id for stability
        hash_input = f"{self.config.experiment_id}:{entity_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        
        # Map to 0-100 range
        bucket = hash_value % 100
        
        for start, end, variant_name in self.allocation_ranges:
            if start <= bucket < end:
                return variant_name
        
        return self.config.variants[0].name  # Fallback to first variant
    
    def _random_assign(self) -> str:
        """Random assignment"""
        bucket = np.random.uniform(0, 100)
        
        for start, end, variant_name in self.allocation_ranges:
            if start <= bucket < end:
                return variant_name
        
        return self.config.variants[0].name
    
    def _stratified_assign(
        self,
        entity_id: str,
        attributes: Dict[str, Any]
    ) -> str:
        """Stratified assignment considering attributes"""
        # Combine entity_id with attributes for stratification
        strat_key = f"{entity_id}:{json.dumps(attributes, sort_keys=True)}"
        return self._deterministic_assign(strat_key)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on experiment data
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_variants(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        metric_name: str
    ) -> StatisticalResult:
        """
        Compare two variants using Welch's t-test
        
        Args:
            control: Control variant metrics
            treatment: Treatment variant metrics
            metric_name: Name of metric to compare
            
        Returns:
            StatisticalResult
        """
        # Get metric values
        control_value = getattr(control, metric_name, 0)
        treatment_value = getattr(treatment, metric_name, 0)
        
        # Calculate lift
        if control_value != 0:
            relative_lift = (treatment_value - control_value) / control_value
        else:
            relative_lift = 0.0 if treatment_value == 0 else float('inf')
        
        absolute_lift = treatment_value - control_value
        
        # Perform statistical test
        if control.metric_values and treatment.metric_values:
            # Welch's t-test (doesn't assume equal variance)
            t_stat, p_value = stats.ttest_ind(
                treatment.metric_values,
                control.metric_values,
                equal_var=False
            )
            
            # Confidence interval for difference
            ci = self._compute_confidence_interval(
                control.metric_values,
                treatment.metric_values
            )
        else:
            # Fallback: use aggregate metrics with assumed variance
            p_value = self._approximate_p_value(
                control_value, control.sample_size, control.metric_variance,
                treatment_value, treatment.sample_size, treatment.metric_variance
            )
            ci = (absolute_lift * 0.8, absolute_lift * 1.2)  # Rough estimate
        
        is_significant = p_value < self.significance_level
        
        return StatisticalResult(
            metric_name=metric_name,
            control_mean=control_value,
            treatment_mean=treatment_value,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=is_significant,
            confidence_level=1 - self.significance_level,
        )
    
    def _compute_confidence_interval(
        self,
        control_values: List[float],
        treatment_values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for difference in means"""
        control_arr = np.array(control_values)
        treatment_arr = np.array(treatment_values)
        
        diff_mean = np.mean(treatment_arr) - np.mean(control_arr)
        
        # Pooled standard error
        se = np.sqrt(
            np.var(control_arr, ddof=1) / len(control_arr) +
            np.var(treatment_arr, ddof=1) / len(treatment_arr)
        )
        
        # t-critical value
        df = len(control_arr) + len(treatment_arr) - 2
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        
        margin = t_crit * se
        return (diff_mean - margin, diff_mean + margin)
    
    def _approximate_p_value(
        self,
        control_mean: float,
        control_n: int,
        control_var: float,
        treatment_mean: float,
        treatment_n: int,
        treatment_var: float
    ) -> float:
        """Approximate p-value from aggregate statistics"""
        if control_n == 0 or treatment_n == 0:
            return 1.0
        
        # Estimate standard error
        se = np.sqrt(
            control_var / control_n +
            treatment_var / treatment_n
        )
        
        if se == 0:
            return 1.0
        
        # t-statistic
        t_stat = (treatment_mean - control_mean) / se
        
        # Approximate degrees of freedom (Welch-Satterthwaite)
        df = (control_var / control_n + treatment_var / treatment_n) ** 2 / (
            (control_var / control_n) ** 2 / (control_n - 1) +
            (treatment_var / treatment_n) ** 2 / (treatment_n - 1)
        ) if control_n > 1 and treatment_n > 1 else 1
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return p_value
    
    def compute_sample_size(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Compute required sample size per variant
        
        Args:
            baseline_rate: Expected baseline conversion rate
            mde: Minimum detectable effect (relative)
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Required sample size per variant
        """
        # Effect size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size formula
        n = (
            2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta) ** 2
        ) / (p2 - p1) ** 2
        
        return int(np.ceil(n))


class DRLABTestManager:
    """
    Manages A/B testing for DRL optimization evaluation
    """
    
    def __init__(
        self,
        storage_dir: str = "experiments",
        default_treatment_ratio: float = 0.2,
        glossary: Optional[ParameterGlossary] = None,
    ):
        """
        Args:
            storage_dir: Directory for experiment storage
            default_treatment_ratio: Default DRL treatment allocation
            glossary: Optional parameter glossary for narrative enrichment
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.default_treatment_ratio = default_treatment_ratio
        self.analyzer = StatisticalAnalyzer()
        self.glossary = glossary or ParameterGlossary()
        
        # Active experiments
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.assigners: Dict[str, VariantAssigner] = {}
        self.metrics: Dict[str, Dict[str, VariantMetrics]] = {}
        
        # Load existing experiments
        self._load_experiments()
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        treatment_ratio: Optional[float] = None,
        primary_metric: str = "roas",
        min_duration_days: int = 14
    ) -> ExperimentConfig:
        """
        Create new A/B experiment
        
        Args:
            name: Experiment name
            description: Description
            treatment_ratio: DRL treatment allocation (default 20%)
            primary_metric: Primary evaluation metric
            min_duration_days: Minimum experiment duration
            
        Returns:
            ExperimentConfig
        """
        treatment_ratio = treatment_ratio or self.default_treatment_ratio
        
        experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=[
                ExperimentVariant(
                    name="baseline",
                    description="Rule-based optimization",
                    allocation_percent=(1 - treatment_ratio) * 100,
                    is_control=True,
                    config={"optimizer": "rule_based"},
                ),
                ExperimentVariant(
                    name="drl",
                    description="DRL optimization",
                    allocation_percent=treatment_ratio * 100,
                    is_control=False,
                    config={"optimizer": "drl"},
                ),
            ],
            primary_metric=primary_metric,
            min_duration_days=min_duration_days,
        )
        
        self.experiments[experiment_id] = config
        self.assigners[experiment_id] = VariantAssigner(config)
        self.metrics[experiment_id] = {
            "baseline": VariantMetrics(variant_name="baseline"),
            "drl": VariantMetrics(variant_name="drl"),
        }
        
        self._save_experiment(experiment_id)
        
        logger.info(f"Created experiment: {experiment_id}")
        return config
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        config = self.experiments[experiment_id]
        config.start_date = datetime.now(timezone.utc)
        
        self._save_experiment(experiment_id)
        logger.info(f"Started experiment: {experiment_id}")
        return True
    
    def get_assignment(
        self,
        experiment_id: str,
        campaign_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get variant assignment for a campaign
        
        Args:
            experiment_id: Experiment ID
            campaign_id: Campaign ID
            
        Returns:
            Tuple of (variant_name, variant_config)
        """
        if experiment_id not in self.assigners:
            logger.warning(f"Experiment not found: {experiment_id}")
            return "baseline", {"optimizer": "rule_based"}
        
        assigner = self.assigners[experiment_id]
        variant_name = assigner.assign(campaign_id)
        
        # Get variant config
        config = self.experiments[experiment_id]
        for variant in config.variants:
            if variant.name == variant_name:
                return variant_name, variant.config
        
        return variant_name, {}
    
    def record_metrics(
        self,
        experiment_id: str,
        variant_name: str,
        metrics: Dict[str, float]
    ):
        """
        Record metrics for a variant
        
        Args:
            experiment_id: Experiment ID
            variant_name: Variant name
            metrics: Metrics dictionary
        """
        if experiment_id not in self.metrics:
            return
        
        if variant_name not in self.metrics[experiment_id]:
            return
        
        variant_metrics = self.metrics[experiment_id][variant_name]
        
        # Update totals
        variant_metrics.sample_size += 1
        variant_metrics.total_spend += metrics.get("spend", 0)
        variant_metrics.total_revenue += metrics.get("revenue", 0)
        variant_metrics.total_conversions += metrics.get("conversions", 0)
        variant_metrics.total_clicks += metrics.get("clicks", 0)
        variant_metrics.total_impressions += metrics.get("impressions", 0)
        
        # Store individual metric values for statistical testing
        primary_metric = self.experiments[experiment_id].primary_metric
        if primary_metric in metrics:
            variant_metrics.metric_values.append(metrics[primary_metric])
        
        # Recompute derived metrics
        variant_metrics.compute_metrics()
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """
        Analyze experiment results
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            ExperimentResult
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        config = self.experiments[experiment_id]
        metrics = self.metrics[experiment_id]
        
        # Get control and treatment
        control = None
        treatment = None
        
        for variant in config.variants:
            if variant.is_control:
                control = metrics.get(variant.name)
            else:
                treatment = metrics.get(variant.name)
        
        if not control or not treatment:
            raise ValueError("Missing control or treatment metrics")
        
        # Compute duration
        start_date = config.start_date or datetime.now(timezone.utc)
        end_date = datetime.now(timezone.utc)
        duration_days = (end_date - start_date).days
        
        # Determine status
        if duration_days >= config.min_duration_days:
            status = ExperimentStatus.COMPLETED
        else:
            status = ExperimentStatus.RUNNING
        
        # Primary metric analysis
        primary_result = self.analyzer.compare_variants(
            control, treatment, config.primary_metric
        )
        
        # Secondary metrics analysis
        secondary_results = {}
        for metric in config.secondary_metrics:
            secondary_results[metric] = self.analyzer.compare_variants(
                control, treatment, metric
            )
        
        # Check guardrails
        guardrail_violations = self._check_guardrails(
            treatment, config.guardrail_metrics
        )
        
        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            primary_result, secondary_results, guardrail_violations, config
        )
        
        # Generate human-readable narrative
        narrative = self._generate_narrative(
            config, primary_result, secondary_results,
            control, treatment, guardrail_violations,
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            variant_metrics=metrics,
            primary_result=primary_result,
            secondary_results=secondary_results,
            guardrail_violations=guardrail_violations,
            recommendation=recommendation,
            confidence_in_recommendation=confidence,
            narrative=narrative,
        )
    
    def _check_guardrails(
        self,
        treatment: VariantMetrics,
        guardrails: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """Check guardrail metrics"""
        violations = []
        
        for metric, (min_val, max_val) in guardrails.items():
            value = getattr(treatment, metric, None)
            if value is not None:
                if value < min_val:
                    violations.append({
                        "metric": metric,
                        "violation": "below_minimum",
                        "value": value,
                        "threshold": min_val,
                    })
                elif value > max_val:
                    violations.append({
                        "metric": metric,
                        "violation": "above_maximum",
                        "value": value,
                        "threshold": max_val,
                    })
        
        return violations
    
    def _generate_recommendation(
        self,
        primary_result: StatisticalResult,
        secondary_results: Dict[str, StatisticalResult],
        guardrail_violations: List[Dict[str, Any]],
        config: ExperimentConfig
    ) -> Tuple[str, float]:
        """Generate recommendation based on results"""
        
        # Check for guardrail violations
        if guardrail_violations:
            return (
                f"DO NOT SHIP: Guardrail violations detected - {[v['metric'] for v in guardrail_violations]}",
                0.95
            )
        
        # Check primary metric significance
        if not primary_result.is_significant:
            return (
                f"INCONCLUSIVE: {config.primary_metric} lift of {primary_result.relative_lift:.1%} "
                f"not statistically significant (p={primary_result.p_value:.3f})",
                0.5
            )
        
        # Positive significant result
        if primary_result.relative_lift > 0:
            # Check secondary metrics for any significant regressions
            regressions = [
                name for name, result in secondary_results.items()
                if result.is_significant and result.relative_lift < -0.05
            ]
            
            if regressions:
                return (
                    f"PROCEED WITH CAUTION: {config.primary_metric} improved {primary_result.relative_lift:.1%} "
                    f"but regressions in {regressions}",
                    0.7
                )
            
            return (
                f"SHIP: {config.primary_metric} improved {primary_result.relative_lift:.1%} "
                f"(p={primary_result.p_value:.4f}, CI: [{primary_result.confidence_interval[0]:.1%}, "
                f"{primary_result.confidence_interval[1]:.1%}])",
                0.9
            )
        
        # Negative significant result
        return (
            f"DO NOT SHIP: {config.primary_metric} degraded {primary_result.relative_lift:.1%} "
            f"(p={primary_result.p_value:.4f})",
            0.9
        )
    
    def _generate_narrative(
        self,
        config: ExperimentConfig,
        primary_result: StatisticalResult,
        secondary_results: Dict[str, StatisticalResult],
        control: VariantMetrics,
        treatment: VariantMetrics,
        guardrail_violations: List[Dict[str, Any]],
    ) -> str:
        """Generate a plain-English narrative summarising the A/B test."""
        metric = config.primary_metric
        gloss = self.glossary.lookup(metric)
        metric_name = gloss.get("full_name", metric)

        lines = [
            f"Experiment \"{config.name}\" ran for {(datetime.now(timezone.utc) - (config.start_date or datetime.now(timezone.utc))).days} days "
            f"with {control.sample_size} baseline and {treatment.sample_size} DRL observations.",
            "",
            f"Primary metric ({metric_name}): "
            f"baseline {primary_result.control_mean:.4f} vs DRL {primary_result.treatment_mean:.4f} "
            f"({primary_result.relative_lift:+.1%} lift, p={primary_result.p_value:.4f}).",
        ]

        if primary_result.is_significant and primary_result.relative_lift > 0:
            lines.append(
                f"This is a statistically significant improvement at the "
                f"{(1 - config.significance_level)*100:.0f}% confidence level."
            )
        elif primary_result.is_significant:
            lines.append(
                "This is a statistically significant regression. "
                "DRL is under-performing the baseline for this metric."
            )
        else:
            lines.append(
                "The difference is not statistically significant — "
                "more data or a longer run may be needed."
            )

        # Secondary highlights
        sig_secondary = [
            (name, r) for name, r in secondary_results.items()
            if r.is_significant
        ]
        if sig_secondary:
            lines.append("")
            lines.append("Secondary metrics with significant changes:")
            for name, r in sig_secondary:
                direction = "improved" if r.relative_lift > 0 else "degraded"
                lines.append(
                    f"  - {name}: {direction} by {r.relative_lift:+.1%} (p={r.p_value:.4f})"
                )

        if guardrail_violations:
            lines.append("")
            lines.append(
                f"WARNING: {len(guardrail_violations)} guardrail violation(s) detected — "
                "review before proceeding."
            )

        return "\n".join(lines)

    def _save_experiment(self, experiment_id: str):
        """Save experiment to disk"""
        config = self.experiments[experiment_id]
        path = self.storage_dir / f"{experiment_id}.json"
        
        with open(path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def _load_experiments(self):
        """Load existing experiments from disk"""
        for path in self.storage_dir.glob("exp_*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                experiment_id = data["experiment_id"]
                
                # Reconstruct config
                variants = [
                    ExperimentVariant(**v) for v in data.get("variants", [])
                ]
                
                config = ExperimentConfig(
                    experiment_id=experiment_id,
                    name=data["name"],
                    description=data.get("description", ""),
                    variants=variants,
                    primary_metric=data.get("primary_metric", "roas"),
                    secondary_metrics=data.get("secondary_metrics", []),
                )
                
                if data.get("start_date"):
                    config.start_date = datetime.fromisoformat(data["start_date"])
                
                self.experiments[experiment_id] = config
                self.assigners[experiment_id] = VariantAssigner(config)
                self.metrics[experiment_id] = {
                    v.name: VariantMetrics(variant_name=v.name)
                    for v in variants
                }
                
            except Exception as e:
                logger.error(f"Failed to load experiment {path}: {e}")
    
    def create_platform_experiment(
        self,
        platform: str,
        name: Optional[str] = None,
        description: str = "",
        treatment_ratio: Optional[float] = None,
        primary_metric: str = "roas",
        min_duration_days: int = 14,
    ) -> ExperimentConfig:
        """
        Create an A/B experiment testing a platform-specific P-Model
        against the global DRL agent for a single platform.

        Control = global agent (fallback), Treatment = platform P-Model.

        Args:
            platform: Platform name (e.g. "meta", "google").
            name: Experiment name (auto-generated if None).
            treatment_ratio: P-Model treatment allocation (default 20%).
            primary_metric: Primary evaluation metric.
            min_duration_days: Minimum experiment duration.
        """
        treatment_ratio = treatment_ratio or self.default_treatment_ratio
        name = name or f"P-Model vs Global — {platform}"

        experiment_id = f"pmodel_{platform}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description or (
                f"Tests platform-specific P-Model for {platform} "
                f"against the global DRL baseline."
            ),
            variants=[
                ExperimentVariant(
                    name="global_baseline",
                    description=f"Global DRL agent for {platform}",
                    allocation_percent=(1 - treatment_ratio) * 100,
                    is_control=True,
                    config={"optimizer": "global_drl", "platform": platform},
                ),
                ExperimentVariant(
                    name="p_model",
                    description=f"Platform-specific P-Model for {platform}",
                    allocation_percent=treatment_ratio * 100,
                    is_control=False,
                    config={"optimizer": "p_model", "platform": platform},
                ),
            ],
            primary_metric=primary_metric,
            min_duration_days=min_duration_days,
        )

        self.experiments[experiment_id] = config
        self.assigners[experiment_id] = VariantAssigner(config)
        self.metrics[experiment_id] = {
            "global_baseline": VariantMetrics(variant_name="global_baseline"),
            "p_model": VariantMetrics(variant_name="p_model"),
        }

        self._save_experiment(experiment_id)
        logger.info(f"Created P-Model experiment: {experiment_id} for {platform}")
        return config

    def create_portfolio_experiment(
        self,
        name: Optional[str] = None,
        description: str = "",
        treatment_ratio: Optional[float] = None,
        primary_metric: str = "roas",
        min_duration_days: int = 21,
    ) -> ExperimentConfig:
        """
        Create an A/B experiment testing X-Model allocation against
        the heuristic BudgetAllocator at the portfolio level.

        Control = heuristic allocation, Treatment = X-Model allocation.
        Assignment is at the portfolio/account level (not per-campaign).

        Args:
            name: Experiment name (auto-generated if None).
            treatment_ratio: X-Model treatment allocation (default 20%).
            primary_metric: Primary evaluation metric.
            min_duration_days: Minimum experiment duration.
        """
        treatment_ratio = treatment_ratio or self.default_treatment_ratio
        name = name or "X-Model vs Heuristic Allocation"

        experiment_id = f"xmodel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description or (
                "Tests the learned X-Model cross-platform allocation "
                "against the heuristic BudgetAllocator."
            ),
            variants=[
                ExperimentVariant(
                    name="heuristic",
                    description="Heuristic BudgetAllocator (marginal-return based)",
                    allocation_percent=(1 - treatment_ratio) * 100,
                    is_control=True,
                    config={"allocator": "heuristic"},
                ),
                ExperimentVariant(
                    name="x_model",
                    description="Learned X-Model cross-platform allocation",
                    allocation_percent=treatment_ratio * 100,
                    is_control=False,
                    config={"allocator": "x_model"},
                ),
            ],
            assignment_key="account_id",  # Portfolio-level assignment
            primary_metric=primary_metric,
            min_duration_days=min_duration_days,
        )

        self.experiments[experiment_id] = config
        self.assigners[experiment_id] = VariantAssigner(config)
        self.metrics[experiment_id] = {
            "heuristic": VariantMetrics(variant_name="heuristic"),
            "x_model": VariantMetrics(variant_name="x_model"),
        }

        self._save_experiment(experiment_id)
        logger.info(f"Created X-Model portfolio experiment: {experiment_id}")
        return config

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of experiment"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        config = self.experiments[experiment_id]
        metrics = self.metrics[experiment_id]
        
        summary = {
            "experiment_id": experiment_id,
            "name": config.name,
            "status": "running" if config.start_date else "draft",
            "variants": {},
        }
        
        for name, variant_metrics in metrics.items():
            summary["variants"][name] = {
                "sample_size": variant_metrics.sample_size,
                "roas": variant_metrics.roas,
                "cpa": variant_metrics.cpa,
                "ctr": variant_metrics.ctr,
            }
        
        return summary

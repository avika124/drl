"""
Audience Segmentation Constraints for DRL campaign optimisation.

Provides:
- AudienceSegment: segment definition with budget floors/ceilings and frequency caps
- AudienceConstraintManager: distributes platform budget across segments
- SegmentAllocation / AudienceConstraintResult: structured outputs
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .state_action import ActionSpace, AudienceAction


# ── Data containers ───────────────────────────────────────────────

@dataclass
class AudienceSegment:
    segment_id: str
    segment_name: str
    platform: str
    min_budget_pct: float = 0.0
    max_budget_pct: float = 1.0
    max_exposures_per_user: int = 10
    max_daily_frequency: int = 3
    performance_history: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentAllocation:
    segment_id: str
    recommended_budget_pct: float
    recommended_daily_frequency: int
    rationale: str

    def __repr__(self) -> str:
        return (
            f"SegmentAllocation({self.segment_id}, "
            f"budget={self.recommended_budget_pct:.1%}, "
            f"freq={self.recommended_daily_frequency})"
        )


@dataclass
class AudienceConstraintResult:
    allocations: List[SegmentAllocation]
    total_budget_check_passed: bool
    violations: List[str]
    narrative: str

    def __repr__(self) -> str:
        return (
            f"AudienceConstraintResult(segs={len(self.allocations)}, "
            f"budget_ok={self.total_budget_check_passed}, "
            f"violations={len(self.violations)})"
        )


# ── Manager ───────────────────────────────────────────────────────

class AudienceConstraintManager:
    """Manages budget distribution across audience segments with
    frequency capping and distribution constraints."""

    def __init__(self, segments: List[AudienceSegment]):
        self.segments: Dict[str, AudienceSegment] = {s.segment_id: s for s in segments}
        self._history: Dict[str, deque] = {}
        self._validate_constraints()

    # ── validation ────────────────────────────────────────────────

    def _validate_constraints(self) -> None:
        total_min = sum(s.min_budget_pct for s in self.segments.values())
        if total_min > 1.0:
            raise ValueError(
                f"Sum of min_budget_pct across segments is {total_min:.2f} (>1.0). "
                "Budget floors are infeasible."
            )
        for s in self.segments.values():
            if s.min_budget_pct > s.max_budget_pct:
                raise ValueError(
                    f"Segment {s.segment_id}: min_budget_pct ({s.min_budget_pct}) "
                    f"> max_budget_pct ({s.max_budget_pct})"
                )
            if s.max_exposures_per_user < 1:
                raise ValueError(
                    f"Segment {s.segment_id}: max_exposures_per_user must be >= 1"
                )

    # ── budget allocation ─────────────────────────────────────────

    def allocate_budget(
        self,
        platform_budget: float,
        action: ActionSpace,
        performance_signals: Dict[str, Dict[str, float]],
    ) -> AudienceConstraintResult:
        violations: List[str] = []
        seg_ids = list(self.segments.keys())

        # 1. Floors
        alloc = {sid: self.segments[sid].min_budget_pct for sid in seg_ids}
        remaining = 1.0 - sum(alloc.values())
        if remaining < -1e-9:
            violations.append("Sum of min_budget_pct exceeds 1.0")
            remaining = 0.0

        # 2. Score each segment
        scores: Dict[str, float] = {}
        for sid in seg_ids:
            sig = performance_signals.get(sid, self.segments[sid].performance_history)
            roas = sig.get("roas", 1.0)
            cvr = sig.get("cvr", 0.01)
            ctr = sig.get("ctr", 0.01)
            scores[sid] = roas * 0.5 + cvr * 0.3 + ctr * 0.2

        total_score = sum(scores.values()) or 1.0

        # 3. Distribute remaining proportionally, capped at max
        for sid in seg_ids:
            share = (scores[sid] / total_score) * remaining
            cap = self.segments[sid].max_budget_pct
            alloc[sid] = min(alloc[sid] + share, cap)

        # Re-normalise if total exceeds 1.0
        total_alloc = sum(alloc.values())
        if total_alloc > 1.0 + 1e-9:
            factor = 1.0 / total_alloc
            alloc = {sid: v * factor for sid, v in alloc.items()}

        # 4. Apply DRL audience action modifier
        aud = AudienceAction(action.audience_action)
        if aud == AudienceAction.EXPAND:
            best_sid = max(scores, key=scores.get)  # type: ignore[arg-type]
            bump = min(0.10, self.segments[best_sid].max_budget_pct - alloc[best_sid])
            alloc[best_sid] += max(bump, 0.0)

        elif aud == AudienceAction.REFINE:
            avg_cvr = sum(
                performance_signals.get(sid, {}).get("cvr", 0.0) for sid in seg_ids
            ) / max(len(seg_ids), 1)
            for sid in seg_ids:
                seg_cvr = performance_signals.get(sid, {}).get("cvr", 0.0)
                if seg_cvr > avg_cvr:
                    bump = min(0.05, self.segments[sid].max_budget_pct - alloc[sid])
                    alloc[sid] += max(bump, 0.0)
                elif seg_cvr < avg_cvr:
                    drop = min(0.05, alloc[sid] - self.segments[sid].min_budget_pct)
                    alloc[sid] -= max(drop, 0.0)

        elif aud == AudienceAction.EXCLUDE:
            worst_sid = min(scores, key=scores.get)  # type: ignore[arg-type]
            freed = alloc[worst_sid]
            alloc[worst_sid] = 0.0
            others = [sid for sid in seg_ids if sid != worst_sid]
            if others:
                per_other = freed / len(others)
                for sid in others:
                    alloc[sid] = min(alloc[sid] + per_other, self.segments[sid].max_budget_pct)

        # Re-normalise again
        total_alloc = sum(alloc.values())
        if total_alloc > 1.0 + 1e-9:
            factor = 1.0 / total_alloc
            alloc = {sid: v * factor for sid, v in alloc.items()}

        # 5. Build result
        allocations: List[SegmentAllocation] = []
        narrative_lines: List[str] = []
        for sid in seg_ids:
            seg = self.segments[sid]
            pct = alloc[sid]
            dollar = platform_budget * pct
            freq = min(seg.max_daily_frequency, seg.max_exposures_per_user)

            rationale = (
                f"Allocated {pct:.1%} (${dollar:,.0f}) to {seg.segment_name} "
                f"based on performance score {scores[sid]:.3f}."
            )
            allocations.append(SegmentAllocation(
                segment_id=sid,
                recommended_budget_pct=pct,
                recommended_daily_frequency=freq,
                rationale=rationale,
            ))
            narrative_lines.append(
                f"- {seg.segment_name}: {pct:.1%} of budget (${dollar:,.0f}), "
                f"frequency cap {freq}/day"
            )

        budget_ok = sum(alloc.values()) <= 1.0 + 1e-9
        narrative = (
            f"Budget ${platform_budget:,.0f} distributed across "
            f"{len(seg_ids)} segment(s):\n" + "\n".join(narrative_lines)
        )

        return AudienceConstraintResult(
            allocations=allocations,
            total_budget_check_passed=budget_ok,
            violations=violations,
            narrative=narrative,
        )

    # ── frequency capping ─────────────────────────────────────────

    def apply_frequency_caps(
        self,
        segment_id: str,
        current_frequency: int,
    ) -> Dict[str, Any]:
        seg = self.segments.get(segment_id)
        if seg is None:
            return {"should_pause": False, "remaining": 0, "recommendation": "Unknown segment."}

        cap = seg.max_exposures_per_user
        remaining = max(cap - current_frequency, 0)
        should_pause = current_frequency >= cap

        if should_pause:
            rec = f"Pause delivery to this user — frequency cap ({cap}) reached."
        elif remaining <= 2:
            rec = f"Approaching cap — {remaining} exposure(s) remaining."
        else:
            rec = f"{remaining} exposures remaining before cap."

        return {"should_pause": should_pause, "remaining": remaining, "recommendation": rec}

    # ── performance history update ────────────────────────────────

    def update_performance_history(
        self,
        segment_id: str,
        new_metrics: Dict[str, float],
        rolling_window: int = 7,
    ) -> None:
        if segment_id not in self.segments:
            return

        if segment_id not in self._history:
            self._history[segment_id] = deque(maxlen=rolling_window)

        self._history[segment_id].append(new_metrics)

        window = self._history[segment_id]
        avg: Dict[str, float] = {}
        for key in new_metrics:
            vals = [m.get(key, 0.0) for m in window]
            avg[key] = sum(vals) / max(len(vals), 1)

        self.segments[segment_id].performance_history = avg

    # ── summary ───────────────────────────────────────────────────

    def segment_summary(self) -> str:
        lines: List[str] = []
        for sid, seg in self.segments.items():
            roas_str = f"7d ROAS {seg.performance_history.get('roas', 'N/A')}"
            lines.append(
                f"{seg.segment_name} ({sid}): "
                f"budget [{seg.min_budget_pct:.0%}-{seg.max_budget_pct:.0%}], "
                f"freq cap {seg.max_daily_frequency}/day, {roas_str}"
            )
        return "\n".join(lines)

"""
Comprehensive Quality Controller Component

Implements unified quality control that aggregates critic assessments, applies
acceptance/rejection logic, manages revision cycles, and provides comprehensive
quality feedback for the adversarial system.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.plot_coherence_assessment import PlotCoherenceAssessment
from musequill.v3.models.literary_quality_assessment import LiteraryQualityAssessment
from musequill.v3.models.reader_engagement_assessment import ReaderEngagementAssessment
from musequill.v3.models.market_intelligence import MarketIntelligence


class QualityDecision(str, Enum):
    """Quality control decisions."""
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class QualityControllerConfig(BaseModel):
    """Configuration for Comprehensive Quality Controller."""
    
    plot_coherence_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for plot coherence in overall quality score"
    )
    
    literary_quality_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for literary quality in overall quality score"
    )
    
    reader_engagement_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for reader engagement in overall quality score"
    )
    
    acceptance_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum overall quality score for acceptance"
    )
    
    revision_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum quality score before rejection (allows revision)"
    )
    
    max_revision_cycles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of revision attempts"
    )
    
    enable_adaptive_thresholds: bool = Field(
        default=True,
        description="Whether to adapt thresholds based on story position and market data"
    )
    
    critical_issue_rejection: bool = Field(
        default=True,
        description="Whether critical issues automatically trigger rejection"
    )
    
    market_alignment_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Weight for market alignment in quality decisions"
    )
    
    enable_comprehensive_feedback: bool = Field(
        default=True,
        description="Whether to provide detailed improvement guidance"
    )


class QualityControllerInput(BaseModel):
    """Input data for Quality Controller."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate"
    )
    
    plot_coherence_assessment: PlotCoherenceAssessment = Field(
        description="Plot coherence critic assessment"
    )
    
    literary_quality_assessment: LiteraryQualityAssessment = Field(
        description="Literary quality critic assessment"
    )
    
    reader_engagement_assessment: ReaderEngagementAssessment = Field(
        description="Reader engagement critic assessment"
    )
    
    market_intelligence: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence for commercial alignment assessment"
    )
    
    story_position: float = Field(
        ge=0.0,
        le=1.0,
        description="Position in story (0.0 = beginning, 1.0 = end)"
    )
    
    revision_cycle: int = Field(
        default=0,
        ge=0,
        description="Current revision cycle number"
    )


class ComprehensiveQualityAssessment(BaseModel):
    """Comprehensive quality assessment combining all critic evaluations."""
    
    overall_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted overall quality score"
    )
    
    plot_coherence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Plot coherence component score"
    )
    
    literary_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Literary quality component score"
    )
    
    reader_engagement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Reader engagement component score"
    )
    
    market_alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Market alignment score"
    )
    
    decision: QualityDecision = Field(
        description="Quality control decision"
    )
    
    decision_rationale: str = Field(
        description="Explanation of decision reasoning"
    )
    
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Critical issues that must be addressed"
    )
    
    improvement_priorities: List[str] = Field(
        default_factory=list,
        description="Prioritized list of improvements needed"
    )
    
    revision_guidance: Optional[str] = Field(
        default=None,
        description="Specific guidance for revision if applicable"
    )
    
    strengths_identified: List[str] = Field(
        default_factory=list,
        description="Notable strengths to preserve"
    )
    
    quality_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Detailed breakdown of quality metrics"
    )
    
    adaptive_threshold_applied: float = Field(
        description="Actual threshold applied (may differ from base threshold)"
    )


class ComprehensiveQualityController(BaseComponent[QualityControllerInput, ComprehensiveQualityAssessment, QualityControllerConfig]):
    """
    Comprehensive Quality Controller for unified quality assessment and decision making.
    
    Aggregates assessments from all critics, applies weighted scoring, manages
    acceptance/rejection decisions, and provides detailed improvement guidance.
    """
    
    def __init__(self, config: ComponentConfiguration[QualityControllerConfig]):
        super().__init__(config)
        self._quality_history: List[Dict[str, Any]] = []
        self._threshold_adaptation_data: Dict[str, List[float]] = {}
        self._decision_statistics: Dict[QualityDecision, int] = {
            QualityDecision.ACCEPT: 0,
            QualityDecision.REVISE: 0,
            QualityDecision.REJECT: 0
        }
    
    async def initialize(self) -> bool:
        """Initialize quality control systems."""
        try:
            # Validate configuration weights sum appropriately
            total_weight = (
                self.config.specific_config.plot_coherence_weight +
                self.config.specific_config.literary_quality_weight +
                self.config.specific_config.reader_engagement_weight
            )
            
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError(f"Quality weights sum to {total_weight}, should sum to 1.0")
            
            # Initialize threshold adaptation system
            if self.config.specific_config.enable_adaptive_thresholds:
                await self._initialize_threshold_adaptation()
            
            # Initialize quality tracking
            self._quality_history = []
            self._threshold_adaptation_data = {
                'early_story': [],
                'mid_story': [],
                'late_story': []
            }
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Quality controller initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: QualityControllerInput) -> ComprehensiveQualityAssessment:
        """
        Perform comprehensive quality assessment and make acceptance decision.
        
        Args:
            input_data: Chapter variant and all critic assessments
            
        Returns:
            Comprehensive quality assessment with decision and guidance
        """
        start_time = datetime.now()
        
        try:
            # Calculate adaptive thresholds based on story position and market data
            adaptive_thresholds = await self._calculate_adaptive_thresholds(
                input_data.story_position,
                input_data.market_intelligence,
                input_data.revision_cycle
            )
            
            # Calculate weighted overall quality score
            overall_score = await self._calculate_weighted_quality_score(
                input_data.plot_coherence_assessment,
                input_data.literary_quality_assessment,
                input_data.reader_engagement_assessment
            )
            
            # Calculate market alignment score if market intelligence available
            market_alignment_score = 0.5  # Default neutral
            if input_data.market_intelligence:
                market_alignment_score = await self._calculate_market_alignment_score(
                    input_data.chapter_variant,
                    input_data.market_intelligence
                )
            
            # Identify critical issues across all assessments
            critical_issues = await self._identify_critical_issues(
                input_data.plot_coherence_assessment,
                input_data.literary_quality_assessment,
                input_data.reader_engagement_assessment
            )
            
            # Make quality decision based on scores, thresholds, and critical issues
            decision = await self._make_quality_decision(
                overall_score,
                market_alignment_score,
                critical_issues,
                adaptive_thresholds,
                input_data.revision_cycle
            )
            
            # Generate decision rationale
            decision_rationale = await self._generate_decision_rationale(
                decision,
                overall_score,
                adaptive_thresholds,
                critical_issues,
                market_alignment_score
            )
            
            # Generate improvement priorities and revision guidance
            improvement_priorities = await self._generate_improvement_priorities(
                input_data.plot_coherence_assessment,
                input_data.literary_quality_assessment,
                input_data.reader_engagement_assessment,
                decision
            )
            
            revision_guidance = None
            if decision == QualityDecision.REVISE:
                revision_guidance = await self._generate_revision_guidance(
                    improvement_priorities,
                    critical_issues,
                    input_data.revision_cycle
                )
            
            # Identify strengths to preserve
            strengths_identified = await self._identify_strengths(
                input_data.plot_coherence_assessment,
                input_data.literary_quality_assessment,
                input_data.reader_engagement_assessment
            )
            
            # Create detailed quality breakdown
            quality_breakdown = await self._create_quality_breakdown(
                input_data.plot_coherence_assessment,
                input_data.literary_quality_assessment,
                input_data.reader_engagement_assessment,
                market_alignment_score
            )
            
            # Compile comprehensive assessment
            comprehensive_assessment = ComprehensiveQualityAssessment(
                overall_quality_score=overall_score,
                plot_coherence_score=input_data.plot_coherence_assessment.overall_score,
                literary_quality_score=input_data.literary_quality_assessment.overall_score,
                reader_engagement_score=input_data.reader_engagement_assessment.overall_engagement_score,
                market_alignment_score=market_alignment_score,
                decision=decision,
                decision_rationale=decision_rationale,
                critical_issues=critical_issues,
                improvement_priorities=improvement_priorities,
                revision_guidance=revision_guidance,
                strengths_identified=strengths_identified,
                quality_breakdown=quality_breakdown,
                adaptive_threshold_applied=adaptive_thresholds['acceptance']
            )
            
            # Update quality tracking and statistics
            await self._update_quality_tracking(
                input_data,
                comprehensive_assessment,
                (datetime.now() - start_time).total_seconds()
            )
            
            return comprehensive_assessment
            
        except Exception as e:
            raise ComponentError(f"Quality control assessment failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on quality control systems."""
        try:
            # Check decision balance (shouldn't be rejecting everything)
            total_decisions = sum(self._decision_statistics.values())
            if total_decisions > 10:
                rejection_rate = self._decision_statistics[QualityDecision.REJECT] / total_decisions
                if rejection_rate > 0.8:  # Rejecting more than 80% indicates problems
                    return False
            
            # Check component error rates
            if self.state.metrics.failure_rate > 0.1:
                return False
            
            # Validate threshold adaptation is working
            if self.config.specific_config.enable_adaptive_thresholds:
                if not self._threshold_adaptation_data:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup quality control resources."""
        try:
            # Preserve critical quality history for next session
            if len(self._quality_history) > 100:
                self._quality_history = self._quality_history[-50:]
            
            return True
            
        except Exception:
            return False
    
    async def _initialize_threshold_adaptation(self) -> None:
        """Initialize adaptive threshold system."""
        # Load historical data for threshold adaptation
        # Would load from persistent storage in real implementation
        pass
    
    async def _calculate_adaptive_thresholds(self, story_position: float,
                                           market_intelligence: Optional[MarketIntelligence],
                                           revision_cycle: int) -> Dict[str, float]:
        """Calculate adaptive quality thresholds based on context."""
        
        base_acceptance = self.config.specific_config.acceptance_threshold
        base_revision = self.config.specific_config.revision_threshold
        
        # Adjust for story position
        if story_position < 0.2:  # Early story - more lenient for setup
            acceptance_modifier = -0.05
        elif story_position > 0.8:  # Late story - stricter for payoff
            acceptance_modifier = 0.05
        else:  # Middle story - standard
            acceptance_modifier = 0.0
        
        # Adjust for revision cycle (more lenient after multiple attempts)
        revision_modifier = min(0.1, revision_cycle * 0.03)
        
        # Market intelligence adjustments
        market_modifier = 0.0
        if market_intelligence:
            # If market is highly competitive, raise standards
            if market_intelligence.competitive_analysis.genre_saturation_level > 0.8:
                market_modifier = 0.05
            # If market has many opportunities, can be slightly more lenient
            elif len(market_intelligence.emerging_opportunities) > 3:
                market_modifier = -0.03
        
        adaptive_acceptance = max(0.5, min(0.95, 
            base_acceptance + acceptance_modifier - revision_modifier + market_modifier))
        adaptive_revision = max(0.3, min(adaptive_acceptance - 0.1,
            base_revision + acceptance_modifier - revision_modifier + market_modifier))
        
        return {
            'acceptance': adaptive_acceptance,
            'revision': adaptive_revision
        }
    
    async def _calculate_weighted_quality_score(self, plot_assessment: PlotCoherenceAssessment,
                                              literary_assessment: LiteraryQualityAssessment,
                                              engagement_assessment: ReaderEngagementAssessment) -> float:
        """Calculate weighted overall quality score from all assessments."""
        
        plot_weight = self.config.specific_config.plot_coherence_weight
        literary_weight = self.config.specific_config.literary_quality_weight
        engagement_weight = self.config.specific_config.reader_engagement_weight
        
        weighted_score = (
            plot_assessment.overall_score * plot_weight +
            literary_assessment.overall_score * literary_weight +
            engagement_assessment.overall_engagement_score * engagement_weight
        )
        
        return weighted_score
    
    async def _calculate_market_alignment_score(self, chapter_variant: ChapterVariant,
                                              market_intelligence: MarketIntelligence) -> float:
        """Calculate how well chapter aligns with market trends and preferences."""
        
        alignment_factors = []
        
        # Check alignment with current trends
        chapter_elements = (
            chapter_variant.emotional_beats_achieved +
            [chapter_variant.approach.value] +
            chapter_variant.new_world_elements
        )
        
        trend_alignment = market_intelligence.assess_trend_alignment(chapter_elements)
        alignment_factors.append(trend_alignment)
        
        # Check for oversaturated elements (penalty)
        oversaturation_penalty = 0.0
        for element in chapter_elements:
            if any(element.lower() in oversaturated.lower() 
                  for oversaturated in market_intelligence.oversaturated_elements):
                oversaturation_penalty += 0.1
        
        oversaturation_score = max(0.0, 1.0 - oversaturation_penalty)
        alignment_factors.append(oversaturation_score)
        
        # Check word count against preferences
        word_count_prefs = market_intelligence.get_preferences_by_type(
            ReaderPreferenceType.CHAPTER_LENGTH
        )
        word_count_alignment = 0.8  # Default neutral
        
        if word_count_prefs:
            # Simple heuristic - would be more sophisticated in real implementation
            target_length = chapter_variant.word_count_target or 2000
            actual_length = chapter_variant.word_count
            length_ratio = min(actual_length, target_length) / max(actual_length, target_length)
            word_count_alignment = length_ratio
        
        alignment_factors.append(word_count_alignment)
        
        return sum(alignment_factors) / len(alignment_factors)
    
    async def _identify_critical_issues(self, plot_assessment: PlotCoherenceAssessment,
                                       literary_assessment: LiteraryQualityAssessment,
                                       engagement_assessment: ReaderEngagementAssessment) -> List[str]:
        """Identify critical issues that require immediate attention."""
        
        critical_issues = []
        
        # Critical plot coherence issues
        if plot_assessment.critical_issues_count > 0:
            critical_issues.append(f"Critical plot inconsistencies: {plot_assessment.critical_issues_count} issues")
        
        if plot_assessment.meaningful_advancement_count == 0:
            critical_issues.append("No meaningful plot advancement")
        
        # Critical literary quality issues
        excessive_cliches = len(literary_assessment.get_cliches_by_severity(ClicheSeverity.EXCESSIVE))
        if excessive_cliches > 0:
            critical_issues.append(f"Excessive cliché usage: {excessive_cliches} instances")
        
        if literary_assessment.voice_issue_count > 3:
            critical_issues.append(f"Major character voice inconsistencies: {literary_assessment.voice_issue_count} issues")
        
        # Critical engagement issues
        if engagement_assessment.high_risk_engagement_issues > 0:
            critical_issues.append("High-risk reader engagement issues detected")
        
        if engagement_assessment.satisfaction_potential_score < 0.3:
            critical_issues.append("Very low reader satisfaction potential")
        
        return critical_issues
    
    async def _make_quality_decision(self, overall_score: float, market_alignment_score: float,
                                   critical_issues: List[str], adaptive_thresholds: Dict[str, float],
                                   revision_cycle: int) -> QualityDecision:
        """Make quality control decision based on scores and issues."""
        
        # Critical issues trigger rejection unless late in revision cycle
        if critical_issues and self.config.specific_config.critical_issue_rejection:
            if revision_cycle < self.config.specific_config.max_revision_cycles:
                return QualityDecision.REJECT
        
        # Incorporate market alignment into decision
        market_weight = self.config.specific_config.market_alignment_weight
        adjusted_score = (overall_score * (1 - market_weight) + 
                         market_alignment_score * market_weight)
        
        # Apply thresholds
        if adjusted_score >= adaptive_thresholds['acceptance']:
            return QualityDecision.ACCEPT
        elif adjusted_score >= adaptive_thresholds['revision']:
            if revision_cycle < self.config.specific_config.max_revision_cycles:
                return QualityDecision.REVISE
            else:
                # Max revisions reached - force decision
                if adjusted_score >= adaptive_thresholds['revision'] + 0.1:
                    return QualityDecision.ACCEPT
                else:
                    return QualityDecision.REJECT
        else:
            return QualityDecision.REJECT
    
    async def _generate_decision_rationale(self, decision: QualityDecision, overall_score: float,
                                         adaptive_thresholds: Dict[str, float],
                                         critical_issues: List[str],
                                         market_alignment_score: float) -> str:
        """Generate explanation for quality control decision."""
        
        rationale_parts = []
        
        # Score explanation
        rationale_parts.append(f"Overall quality score: {overall_score:.2f}")
        rationale_parts.append(f"Acceptance threshold: {adaptive_thresholds['acceptance']:.2f}")
        
        if market_alignment_score != 0.5:  # Not default neutral
            rationale_parts.append(f"Market alignment: {market_alignment_score:.2f}")
        
        # Decision reasoning
        if decision == QualityDecision.ACCEPT:
            rationale_parts.append("Score meets acceptance criteria")
            if not critical_issues:
                rationale_parts.append("No critical issues identified")
        
        elif decision == QualityDecision.REVISE:
            rationale_parts.append("Score allows revision opportunity")
            if critical_issues:
                rationale_parts.append(f"Critical issues require attention: {len(critical_issues)}")
        
        elif decision == QualityDecision.REJECT:
            if critical_issues:
                rationale_parts.append("Critical issues prevent acceptance")
            else:
                rationale_parts.append("Score below revision threshold")
        
        return ". ".join(rationale_parts) + "."
    
    async def _generate_improvement_priorities(self, plot_assessment: PlotCoherenceAssessment,
                                             literary_assessment: LiteraryQualityAssessment,
                                             engagement_assessment: ReaderEngagementAssessment,
                                             decision: QualityDecision) -> List[str]:
        """Generate prioritized list of improvements needed."""
        
        if decision == QualityDecision.ACCEPT:
            return []  # No improvements needed for accepted content
        
        priorities = []
        
        # Critical issues first (from lowest scoring assessments)
        assessment_scores = [
            (plot_assessment.overall_score, plot_assessment.generate_improvement_priorities()),
            (literary_assessment.overall_score, literary_assessment.identify_priority_improvements()),
            (engagement_assessment.overall_engagement_score, engagement_assessment.engagement_recommendations)
        ]
        
        # Sort by score (lowest first) to prioritize most problematic areas
        assessment_scores.sort(key=lambda x: x[0])
        
        for score, improvements in assessment_scores:
            if score < 0.7:  # Only include improvements from problematic areas
                priorities.extend(improvements[:3])  # Top 3 from each area
        
        # Remove duplicates while preserving order
        unique_priorities = []
        seen = set()
        for priority in priorities:
            if priority.lower() not in seen:
                unique_priorities.append(priority)
                seen.add(priority.lower())
        
        return unique_priorities[:8]  # Limit to prevent overwhelming feedback
    
    async def _generate_revision_guidance(self, improvement_priorities: List[str],
                                         critical_issues: List[str],
                                         revision_cycle: int) -> str:
        """Generate specific revision guidance."""
        
        guidance_parts = []
        
        # Focus guidance based on revision cycle
        if revision_cycle == 0:
            guidance_parts.append("First revision: Focus on the most critical structural issues.")
        elif revision_cycle == 1:
            guidance_parts.append("Second revision: Address remaining quality and consistency issues.")
        else:
            guidance_parts.append("Final revision: Polish and refine remaining elements.")
        
        # Critical issues guidance
        if critical_issues:
            guidance_parts.append("CRITICAL: " + "; ".join(critical_issues[:2]))
        
        # Top improvement priorities
        if improvement_priorities:
            guidance_parts.append("Priority improvements: " + "; ".join(improvement_priorities[:3]))
        
        # Revision-specific advice
        if revision_cycle >= 2:
            guidance_parts.append("Consider focusing on one major issue rather than trying to fix everything.")
        
        return " ".join(guidance_parts)
    
    async def _identify_strengths(self, plot_assessment: PlotCoherenceAssessment,
                                literary_assessment: LiteraryQualityAssessment,
                                engagement_assessment: ReaderEngagementAssessment) -> List[str]:
        """Identify strengths to preserve during revision."""
        
        strengths = []
        
        # Plot strengths
        if plot_assessment.overall_score >= 0.8:
            strengths.append("Strong plot coherence and logic")
        
        if plot_assessment.meaningful_advancement_count >= 2:
            strengths.append("Good plot progression across multiple threads")
        
        # Literary strengths
        strengths.extend(literary_assessment.notable_strengths)
        
        if literary_assessment.overall_score >= 0.8:
            strengths.append("High literary quality")
        
        # Engagement strengths
        strengths.extend(engagement_assessment.engagement_strengths)
        
        if engagement_assessment.overall_engagement_score >= 0.8:
            strengths.append("Strong reader engagement potential")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    async def _create_quality_breakdown(self, plot_assessment: PlotCoherenceAssessment,
                                       literary_assessment: LiteraryQualityAssessment,
                                       engagement_assessment: ReaderEngagementAssessment,
                                       market_alignment_score: float) -> Dict[str, Dict[str, float]]:
        """Create detailed breakdown of quality metrics."""
        
        return {
            "plot_coherence": {
                "continuity": plot_assessment.continuity_score,
                "advancement": plot_assessment.advancement_score,
                "logic_consistency": plot_assessment.logic_consistency_score,
                "tension_management": plot_assessment.tension_management_score
            },
            "literary_quality": {
                "language_freshness": literary_assessment.language_freshness_score,
                "character_voice": literary_assessment.character_voice_score,
                "pacing": literary_assessment.pacing_score,
                "prose_quality": literary_assessment.prose_quality_score
            },
            "reader_engagement": {
                "emotional_journey": engagement_assessment.emotional_journey_score,
                "question_answer_balance": engagement_assessment.question_answer_balance_score,
                "satisfaction_potential": engagement_assessment.satisfaction_potential_score,
                "cliffhanger_effectiveness": engagement_assessment.cliffhanger_effectiveness_score
            },
            "market_alignment": {
                "trend_alignment": market_alignment_score,
                "commercial_viability": market_alignment_score  # Simplified
            }
        }
    
    async def _update_quality_tracking(self, input_data: QualityControllerInput,
                                     assessment: ComprehensiveQualityAssessment,
                                     processing_time: float) -> None:
        """Update quality tracking and statistics."""
        
        # Update decision statistics
        self._decision_statistics[assessment.decision] += 1
        
        # Add to quality history
        quality_record = {
            'timestamp': datetime.now(),
            'chapter_number': input_data.chapter_variant.chapter_number,
            'story_position': input_data.story_position,
            'revision_cycle': input_data.revision_cycle,
            'overall_score': assessment.overall_quality_score,
            'decision': assessment.decision.value,
            'critical_issues_count': len(assessment.critical_issues),
            'processing_time': processing_time
        }
        
        self._quality_history.append(quality_record)
        
        # Maintain history size
        if len(self._quality_history) > 200:
            self._quality_history = self._quality_history[-100:]
        
        # Update threshold adaptation data
        story_phase = self._get_story_phase(input_data.story_position)
        self._threshold_adaptation_data[story_phase].append(assessment.overall_quality_score)
        
        # Maintain adaptation data size
        for phase in self._threshold_adaptation_data:
            if len(self._threshold_adaptation_data[phase]) > 50:
                self._threshold_adaptation_data[phase] = self._threshold_adaptation_data[phase][-30:]
    
    def _get_story_phase(self, story_position: float) -> str:
        """Get story phase based on position."""
        if story_position < 0.33:
            return 'early_story'
        elif story_position < 0.67:
            return 'mid_story'
        else:
            return 'late_story'
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality control statistics."""
        total_decisions = sum(self._decision_statistics.values())
        
        if total_decisions == 0:
            return {"message": "No decisions made yet"}
        
        return {
            "total_decisions": total_decisions,
            "acceptance_rate": self._decision_statistics[QualityDecision.ACCEPT] / total_decisions,
            "revision_rate": self._decision_statistics[QualityDecision.REVISE] / total_decisions,
            "rejection_rate": self._decision_statistics[QualityDecision.REJECT] / total_decisions,
            "average_processing_time": sum(record.get('processing_time', 0) for record in self._quality_history[-10:]) / min(10, len(self._quality_history)),
            "quality_trend": self._calculate_quality_trend(),
            "common_critical_issues": self._analyze_common_issues()
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate overall quality trend from recent history."""
        if len(self._quality_history) < 5:
            return "insufficient_data"
        
        recent_scores = [record['overall_score'] for record in self._quality_history[-10:]]
        early_avg = sum(recent_scores[:5]) / 5
        late_avg = sum(recent_scores[-5:]) / 5
        
        if late_avg > early_avg + 0.05:
            return "improving"
        elif late_avg < early_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _analyze_common_issues(self) -> List[str]:
        """Analyze common critical issues from recent history."""
        # Simplified analysis - would be more sophisticated in real implementation
        return ["plot_consistency", "character_voice", "pacing_issues"]
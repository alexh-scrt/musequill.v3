"""
Plot Coherence Assessment Model

Evaluates story logic, character consistency, plot advancement, and narrative
continuity for the Plot Coherence Critic in the adversarial system.
"""

from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional, Any
from enum import Enum


class SeverityLevel(str, Enum):
    """Severity levels for issues and inconsistencies."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class InconsistencyType(str, Enum):
    """Types of plot/story inconsistencies."""
    CHARACTER_KNOWLEDGE = "character_knowledge"
    CHARACTER_BEHAVIOR = "character_behavior"
    TIMELINE = "timeline"
    WORLD_RULES = "world_rules"
    RELATIONSHIP = "relationship"
    PHYSICAL_STATE = "physical_state"
    LOCATION = "location"
    MOTIVATION = "motivation"
    ABILITY = "ability"
    PLOT_LOGIC = "plot_logic"


class AdvancementType(str, Enum):
    """Types of plot advancement."""
    PROGRESSION = "progression"
    COMPLICATION = "complication"
    REVELATION = "revelation"
    RESOLUTION = "resolution"
    SETUP = "setup"
    ESCALATION = "escalation"


class Inconsistency(BaseModel):
    """A specific inconsistency or logical issue found in the chapter."""
    
    inconsistency_id: str = Field(
        description="Unique identifier for this inconsistency"
    )
    
    inconsistency_type: InconsistencyType = Field(
        description="Category of inconsistency"
    )
    
    severity: SeverityLevel = Field(
        description="How severe this inconsistency is"
    )
    
    description: str = Field(
        min_length=20,
        max_length=500,
        description="Detailed description of the inconsistency"
    )
    
    conflicting_elements: List[str] = Field(
        default_factory=list,
        description="Story elements that conflict with each other"
    )
    
    suggested_resolution: str = Field(
        description="Recommended way to resolve this inconsistency"
    )
    
    affected_characters: List[str] = Field(
        default_factory=list,
        description="Character IDs affected by this inconsistency"
    )
    
    affected_plot_threads: List[str] = Field(
        default_factory=list,
        description="Plot thread IDs affected by this inconsistency"
    )
    
    evidence_text: Optional[str] = Field(
        default=None,
        description="Specific text passages that demonstrate the inconsistency"
    )


class PlotAdvancementAnalysis(BaseModel):
    """Analysis of how plot threads were advanced in the chapter."""
    
    thread_id: str = Field(
        description="ID of the plot thread being analyzed"
    )
    
    advancement_type: AdvancementType = Field(
        description="Type of advancement that occurred"
    )
    
    advancement_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of advancement (0.0-1.0)"
    )
    
    advancement_description: str = Field(
        min_length=10,
        max_length=300,
        description="Description of how the thread was advanced"
    )
    
    meaningful_change: bool = Field(
        description="Whether advancement represents meaningful story change"
    )
    
    tension_impact: int = Field(
        ge=-5,
        le=5,
        description="Impact on thread tension (-5 to +5)"
    )
    
    setup_payoff_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Balance between new setup and payoff delivery"
    )


class ContinuityCheck(BaseModel):
    """Specific continuity verification against established story elements."""
    
    element_type: str = Field(
        description="Type of element being checked (character, world, relationship, etc.)"
    )
    
    element_id: str = Field(
        description="ID of specific story element"
    )
    
    consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How consistent with established facts (0.0-1.0)"
    )
    
    verification_notes: str = Field(
        description="Notes on consistency verification"
    )
    
    requires_attention: bool = Field(
        description="Whether this element needs attention/correction"
    )


class TensionManagementAnalysis(BaseModel):
    """Analysis of how narrative tension was managed in the chapter."""
    
    opening_tension: float = Field(
        ge=0.0,
        le=1.0,
        description="Tension level at chapter opening"
    )
    
    closing_tension: float = Field(
        ge=0.0,
        le=1.0,
        description="Tension level at chapter closing"
    )
    
    peak_tension: float = Field(
        ge=0.0,
        le=1.0,
        description="Highest tension point in chapter"
    )
    
    tension_trajectory: str = Field(
        description="Overall tension pattern (building, declining, fluctuating, flat)"
    )
    
    tension_appropriateness: float = Field(
        ge=0.0,
        le=1.0,
        description="How appropriate tension management is for story position"
    )
    
    cliffhanger_effectiveness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Effectiveness of chapter ending (if applicable)"
    )


class PlotCoherenceAssessment(BaseModel):
    """
    Comprehensive assessment of plot coherence and story logic.
    
    Evaluates continuity, character consistency, plot advancement quality,
    and narrative tension management for the Plot Coherence Critic.
    """
    
    chapter_number: int = Field(
        ge=1,
        description="Chapter being assessed"
    )
    
    continuity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall continuity with established story elements"
    )
    
    advancement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of plot thread advancement"
    )
    
    logic_consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Internal logical consistency of chapter events"
    )
    
    tension_management_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Effectiveness of tension management"
    )
    
    flagged_inconsistencies: List[Inconsistency] = Field(
        default_factory=list,
        description="Specific inconsistencies found"
    )
    
    plot_advancement_analysis: List[PlotAdvancementAnalysis] = Field(
        default_factory=list,
        description="Analysis of how plot threads were advanced"
    )
    
    continuity_checks: List[ContinuityCheck] = Field(
        default_factory=list,
        description="Verification against established story elements"
    )
    
    tension_analysis: TensionManagementAnalysis = Field(
        description="Analysis of tension management throughout chapter"
    )
    
    character_consistency_issues: List[str] = Field(
        default_factory=list,
        description="Character behavior/knowledge consistency issues"
    )
    
    world_consistency_issues: List[str] = Field(
        default_factory=list,
        description="World building/rules consistency issues"
    )
    
    timeline_issues: List[str] = Field(
        default_factory=list,
        description="Chronological or timeline consistency issues"
    )
    
    advancement_analysis: str = Field(
        min_length=50,
        max_length=1000,
        description="Overall analysis of plot advancement effectiveness"
    )
    
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific suggestions for improvement"
    )
    
    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate overall plot coherence score."""
        return (
            self.continuity_score * 0.3 +
            self.advancement_score * 0.3 +
            self.logic_consistency_score * 0.25 +
            self.tension_management_score * 0.15
        )
    
    @computed_field
    @property
    def critical_issues_count(self) -> int:
        """Number of critical severity issues."""
        return len([inc for inc in self.flagged_inconsistencies 
                   if inc.severity == SeverityLevel.CRITICAL])
    
    @computed_field
    @property
    def meaningful_advancement_count(self) -> int:
        """Number of plot threads with meaningful advancement."""
        return len([adv for adv in self.plot_advancement_analysis 
                   if adv.meaningful_change])
    
    @computed_field
    @property
    def consistency_risk_level(self) -> str:
        """Overall consistency risk assessment."""
        critical_count = self.critical_issues_count
        major_count = len([inc for inc in self.flagged_inconsistencies 
                          if inc.severity == SeverityLevel.MAJOR])
        
        if critical_count > 0:
            return "critical"
        elif major_count > 2:
            return "high"
        elif major_count > 0 or self.overall_score < 0.6:
            return "medium"
        else:
            return "low"
    
    def get_issues_by_severity(self, severity: SeverityLevel) -> List[Inconsistency]:
        """Get all issues of specific severity level."""
        return [inc for inc in self.flagged_inconsistencies 
                if inc.severity == severity]
    
    def get_issues_by_type(self, issue_type: InconsistencyType) -> List[Inconsistency]:
        """Get all issues of specific type."""
        return [inc for inc in self.flagged_inconsistencies 
                if inc.inconsistency_type == issue_type]
    
    def get_stagnant_threads(self, quality_threshold: float = 0.5) -> List[PlotAdvancementAnalysis]:
        """Get plot threads with poor advancement quality."""
        return [adv for adv in self.plot_advancement_analysis 
                if adv.advancement_quality < quality_threshold]
    
    def get_failing_continuity_checks(self, score_threshold: float = 0.7) -> List[ContinuityCheck]:
        """Get continuity checks that are failing."""
        return [check for check in self.continuity_checks 
                if check.consistency_score < score_threshold or check.requires_attention]
    
    def calculate_tension_effectiveness(self) -> float:
        """Calculate how effectively tension was managed."""
        tension_factors = []
        
        # Tension trajectory appropriateness
        tension_factors.append(self.tension_analysis.tension_appropriateness)
        
        # Tension change meaningfulness (avoid flat tension)
        tension_change = abs(self.tension_analysis.closing_tension - self.tension_analysis.opening_tension)
        if tension_change > 0.1:  # Meaningful change
            tension_factors.append(0.8)
        else:
            tension_factors.append(0.3)  # Penalize flat tension
        
        # Peak tension utilization
        peak_utilization = min(1.0, self.tension_analysis.peak_tension * 1.2)
        tension_factors.append(peak_utilization)
        
        # Cliffhanger bonus (if applicable)
        if self.tension_analysis.cliffhanger_effectiveness is not None:
            tension_factors.append(self.tension_analysis.cliffhanger_effectiveness)
        
        return sum(tension_factors) / len(tension_factors)
    
    def generate_improvement_priorities(self) -> List[str]:
        """Generate prioritized list of improvements needed."""
        priorities = []
        
        # Critical issues first
        critical_issues = self.get_issues_by_severity(SeverityLevel.CRITICAL)
        for issue in critical_issues:
            priorities.append(f"CRITICAL: {issue.description}")
        
        # Low-scoring continuity checks
        failing_checks = self.get_failing_continuity_checks()
        for check in failing_checks[:3]:  # Limit to top 3
            priorities.append(f"Continuity: {check.verification_notes}")
        
        # Poor plot advancement
        stagnant_threads = self.get_stagnant_threads()
        for thread in stagnant_threads[:2]:  # Limit to top 2
            priorities.append(f"Plot advancement: {thread.advancement_description}")
        
        # Overall score issues
        if self.overall_score < 0.6:
            priorities.append("Overall plot coherence needs significant improvement")
        
        return priorities[:5]  # Return top 5 priorities
    
    def is_acceptable_for_publication(self, threshold: float = 0.7) -> bool:
        """
        Determine if chapter meets publication standards for plot coherence.
        
        Args:
            threshold: Minimum overall score for acceptance
            
        Returns:
            True if chapter meets coherence standards
        """
        # Must meet overall threshold
        if self.overall_score < threshold:
            return False
        
        # Cannot have critical issues
        if self.critical_issues_count > 0:
            return False
        
        # Cannot have too many major issues
        major_issues = len(self.get_issues_by_severity(SeverityLevel.MAJOR))
        if major_issues > 1:
            return False
        
        # Must have meaningful plot advancement
        if self.meaningful_advancement_count == 0:
            return False
        
        return True
    
    def calculate_reader_confusion_risk(self) -> float:
        """
        Calculate risk of reader confusion based on inconsistencies.
        
        Returns:
            Confusion risk score from 0.0 (clear) to 1.0 (very confusing)
        """
        confusion_factors = []
        
        # Weight inconsistencies by severity
        severity_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.MAJOR: 0.7,
            SeverityLevel.MODERATE: 0.4,
            SeverityLevel.MINOR: 0.1
        }
        
        total_confusion_weight = sum(
            severity_weights[inc.severity] 
            for inc in self.flagged_inconsistencies
        )
        
        # Normalize to 0-1 scale (assuming 5+ major issues = max confusion)
        confusion_from_inconsistencies = min(1.0, total_confusion_weight / 5.0)
        confusion_factors.append(confusion_from_inconsistencies)
        
        # Logic consistency impact
        logic_confusion = 1.0 - self.logic_consistency_score
        confusion_factors.append(logic_confusion)
        
        # Character consistency issues
        char_confusion = min(1.0, len(self.character_consistency_issues) / 3.0)
        confusion_factors.append(char_confusion)
        
        return sum(confusion_factors) / len(confusion_factors)
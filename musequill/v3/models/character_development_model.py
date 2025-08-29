"""
Character Development Input/Output Models

Defines the input and output data structures for the character development component,
including development plans, assessments, and character state updates.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

from pydantic import BaseModel, Field, computed_field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from musequill.v3.models.character_arc import CharacterArc, RelationshipStatus, CharacterStability
from musequill.v3.models.market_intelligence import ReaderPreference, SuccessPattern


class CharacterDevelopmentPriority(str, Enum):
    """Priority level for character development."""
    CRITICAL = "critical"     # Must be developed this chapter
    HIGH = "high"            # Should be developed soon
    MEDIUM = "medium"        # Can wait a few chapters
    LOW = "low"             # Can be delayed significantly
    DEFERRED = "deferred"    # Development on hold


class DevelopmentType(str, Enum):
    """Type of character development."""
    GROWTH_MILESTONE = "growth_milestone"     # Character learns/grows
    OBSTACLE_RESOLUTION = "obstacle_resolution"  # Overcomes internal/external obstacle
    RELATIONSHIP_EVOLUTION = "relationship_evolution"  # Relationship changes
    VOICE_REFINEMENT = "voice_refinement"     # Voice/speech pattern evolution
    EMOTIONAL_SHIFT = "emotional_shift"       # Emotional state change
    STABILITY_CHANGE = "stability_change"     # Psychological stability change
    CONFLICT_INTRODUCTION = "conflict_introduction"  # New internal conflict
    MOTIVATION_SHIFT = "motivation_shift"     # Goals/motivations change


class CharacterDevelopmentPlan(BaseModel):
    """Plan for developing a specific character."""
    
    character_id: str = Field(description="ID of character to develop")
    character_name: str = Field(description="Name of character for reference")
    
    priority: CharacterDevelopmentPriority = Field(
        description="Priority level for this development"
    )
    
    development_type: DevelopmentType = Field(
        description="Type of development planned"
    )
    
    development_description: str = Field(
        min_length=10,
        max_length=500,
        description="Detailed description of planned development"
    )
    
    target_chapter: int = Field(
        ge=1,
        description="Chapter where development should occur"
    )
    
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites that must be met before development"
    )
    
    expected_outcomes: List[str] = Field(
        default_factory=list,
        description="Expected outcomes from this development"
    )
    
    plot_integration_points: List[str] = Field(
        default_factory=list,
        description="Points where development integrates with plot"
    )
    
    relationship_impacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Expected impacts on relationships with other characters"
    )
    
    voice_evolution_notes: Optional[str] = Field(
        default=None,
        description="Notes on how character voice should evolve"
    )
    
    market_alignment: Optional[str] = Field(
        default=None,
        description="How development aligns with market preferences"
    )
    
    confidence_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence in development plan effectiveness"
    )


class CharacterDevelopmentAssessment(BaseModel):
    """Assessment of character's current development state."""
    
    character_id: str = Field(description="ID of assessed character")
    
    chapters_since_development: int = Field(
        ge=0,
        description="Chapters since last significant development"
    )
    
    development_staleness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score indicating how stale character development is (0=fresh, 1=very stale)"
    )
    
    voice_consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for character voice consistency (0=inconsistent, 1=very consistent)"
    )
    
    arc_progression_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for character arc progression health (0=stagnant, 1=progressing well)"
    )
    
    relationship_health_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for relationship dynamics health (0=static, 1=evolving well)"
    )
    
    unresolved_obstacles: List[str] = Field(
        default_factory=list,
        description="Obstacles character still needs to overcome"
    )
    
    stagnant_relationships: List[str] = Field(
        default_factory=list,
        description="Character IDs of relationships that haven't evolved recently"
    )
    
    voice_inconsistencies: List[str] = Field(
        default_factory=list,
        description="Detected voice inconsistencies that need attention"
    )
    
    development_opportunities: List[str] = Field(
        default_factory=list,
        description="Identified opportunities for character development"
    )
    
    market_alignment_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How well character aligns with market preferences"
    )
    
    @computed_field
    @property
    def overall_health_score(self) -> float:
        """Calculate overall character development health score."""
        scores = [
            1.0 - self.development_staleness_score,  # Invert staleness
            self.voice_consistency_score,
            self.arc_progression_score,
            self.relationship_health_score,
            self.market_alignment_score
        ]
        return sum(scores) / len(scores)
    
    @computed_field
    @property
    def needs_immediate_development(self) -> bool:
        """Determine if character needs immediate development attention."""
        return (self.development_staleness_score > 0.7 or 
                self.overall_health_score < 0.5 or
                len(self.unresolved_obstacles) > 3)


class CharacterDevelopmentInput(BaseModel):
    """Input for character development component."""
    
    characters: Dict[str, CharacterArc] = Field(
        description="Current character arcs to analyze and develop"
    )
    
    current_chapter: int = Field(
        ge=1,
        description="Current chapter number in the story"
    )
    
    plot_outline: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current plot outline and chapter objectives"
    )
    
    genre: str = Field(
        default="general",
        description="Story genre for market-appropriate development"
    )
    
    market_preferences: Optional[List[ReaderPreference]] = Field(
        default=None,
        description="Market intelligence about reader preferences"
    )
    
    success_patterns: Optional[List[SuccessPattern]] = Field(
        default=None,
        description="Identified success patterns from market research"
    )
    
    research_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Research insights about character development techniques"
    )
    
    chapter_objectives: List[str] = Field(
        default_factory=list,
        description="Specific objectives for the current chapter"
    )
    
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Development constraints (time, plot, etc.)"
    )
    
    force_development: List[str] = Field(
        default_factory=list,
        description="Character IDs that must be developed regardless of staleness"
    )


class CharacterDevelopmentResult(BaseModel):
    """Result of character development analysis and planning."""
    
    assessments: Dict[str, CharacterDevelopmentAssessment] = Field(
        description="Assessment of each character's development state"
    )
    
    development_plans: List[CharacterDevelopmentPlan] = Field(
        description="Planned character developments for upcoming chapters"
    )
    
    updated_characters: Dict[str, CharacterArc] = Field(
        description="Updated character arcs with new development data"
    )
    
    priority_developments: List[CharacterDevelopmentPlan] = Field(
        description="High-priority developments that should happen soon"
    )
    
    relationship_updates: Dict[str, Dict[str, RelationshipStatus]] = Field(
        default_factory=dict,
        description="Planned relationship status updates"
    )
    
    voice_evolution_plans: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Plans for character voice evolution"
    )
    
    market_alignment_improvements: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Suggested improvements for market alignment by character"
    )
    
    development_rationale: Dict[str, str] = Field(
        default_factory=dict,
        description="Rationale for development decisions by character"
    )
    
    overall_development_health: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall health score for character development across all characters"
    )
    
    recommended_chapter_focus: List[str] = Field(
        description="Characters recommended for focus in upcoming chapters"
    )
    
    warning_flags: List[str] = Field(
        default_factory=list,
        description="Warning flags for potential development issues"
    )
    
    success_predictions: Dict[str, float] = Field(
        default_factory=dict,
        description="Predicted success scores for planned developments by character"
    )
    
    processing_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the development processing"
    )
    
    @computed_field
    @property
    def characters_needing_development(self) -> List[str]:
        """Get list of character IDs that need development attention."""
        return [
            char_id for char_id, assessment in self.assessments.items()
            if assessment.needs_immediate_development
        ]
    
    @computed_field
    @property
    def total_planned_developments(self) -> int:
        """Total number of planned developments."""
        return len(self.development_plans)
    
    @computed_field
    @property
    def high_priority_development_count(self) -> int:
        """Number of high-priority developments."""
        return len([plan for plan in self.development_plans 
                   if plan.priority in [CharacterDevelopmentPriority.CRITICAL, 
                                      CharacterDevelopmentPriority.HIGH]])
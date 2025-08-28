"""
Character Arc Model

Represents character development progression with emotional state tracking,
relationship dynamics, voice consistency, and narrative function management.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Dict, List, Any, Optional
from enum import Enum


class CharacterStability(str, Enum):
    """Character emotional stability state."""
    STABLE = "stable"
    VOLATILE = "volatile" 
    TRANSFORMING = "transforming"


class NarrativeFunction(str, Enum):
    """Character's primary narrative function in the story."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    CATALYST = "catalyst"
    SUPPORT = "support"
    MENTOR = "mentor"
    FOIL = "foil"


class RelationshipStatus(str, Enum):
    """Status of relationship between characters."""
    ALLIED = "allied"
    ANTAGONISTIC = "antagonistic"
    ROMANTIC = "romantic"
    FAMILIAL = "familial"
    PROFESSIONAL = "professional"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"
    COMPLICATED = "complicated"


class VoiceCharacteristics(BaseModel):
    """Character voice and speech pattern characteristics."""
    
    vocabulary_level: str = Field(
        description="Vocabulary complexity (simple, moderate, complex, academic)"
    )
    
    speech_patterns: List[str] = Field(
        default_factory=list,
        description="Common phrases, verbal tics, speech patterns"
    )
    
    formality_level: str = Field(
        description="Formality of speech (casual, informal, formal, archaic)"
    )
    
    emotional_expressiveness: str = Field(
        description="How emotions are expressed (reserved, moderate, expressive, dramatic)"
    )
    
    dialogue_quirks: List[str] = Field(
        default_factory=list,
        description="Unique dialogue characteristics and quirks"
    )


class CharacterArc(BaseModel):
    """
    Character development arc with progression tracking and voice consistency.
    
    Manages character emotional states, relationship dynamics, growth trajectory,
    and voice characteristics for narrative consistency.
    """
    
    character_id: str = Field(
        description="Unique identifier for this character"
    )
    
    name: str = Field(
        min_length=1,
        max_length=100,
        description="Character's name"
    )
    
    emotional_state: str = Field(
        min_length=1,
        max_length=200,
        description="Current dominant emotional state or mood"
    )
    
    stability: CharacterStability = Field(
        default=CharacterStability.STABLE,
        description="Character's current emotional/psychological stability"
    )
    
    relationship_dynamics: Dict[str, RelationshipStatus] = Field(
        default_factory=dict,
        description="Relationships with other characters (character_id -> status)"
    )
    
    growth_trajectory: List[str] = Field(
        default_factory=list,
        description="Character lessons learned and growth milestones achieved"
    )
    
    remaining_obstacles: List[str] = Field(
        default_factory=list,
        description="Internal/external obstacles character still needs to overcome"
    )
    
    narrative_function: NarrativeFunction = Field(
        description="Character's primary role in the story structure"
    )
    
    voice_characteristics: VoiceCharacteristics = Field(
        description="Character's distinctive voice and speech patterns"
    )
    
    last_development_chapter: int = Field(
        ge=1,
        description="Chapter where character last had significant development"
    )
    
    introduction_chapter: int = Field(
        ge=1,
        description="Chapter where character was first introduced"
    )
    
    personality_traits: List[str] = Field(
        default_factory=list,
        description="Core personality traits that remain consistent"
    )
    
    goals_motivations: List[str] = Field(
        default_factory=list,
        description="Character's primary goals and motivations"
    )
    
    internal_conflicts: List[str] = Field(
        default_factory=list,
        description="Internal psychological conflicts and dilemmas"
    )
    
    @field_validator('last_development_chapter')
    @classmethod
    def validate_development_chapter(cls, v, info):
        """Ensure development doesn't occur before introduction."""
        if 'introduction_chapter' in info.data:
            intro_chapter = info.data['introduction_chapter']
            if v < intro_chapter:
                raise ValueError(
                    f'Development chapter {v} cannot be before introduction chapter {intro_chapter}'
                )
        return v
    
    @computed_field
    @property
    def development_staleness(self) -> int:
        """Calculate chapters since last development (requires current_chapter context)."""
        # Note: This would typically be calculated with current chapter context
        # Kept as computed field for consistency but requires external context
        return 0
    
    @computed_field
    @property
    def relationship_count(self) -> int:
        """Number of tracked relationships for this character."""
        return len(self.relationship_dynamics)
    
    def chapters_since_development(self, current_chapter: int) -> int:
        """
        Calculate chapters since last character development.
        
        Args:
            current_chapter: Current story chapter
            
        Returns:
            Number of chapters since last development
        """
        return max(0, current_chapter - self.last_development_chapter)
    
    def add_relationship(self, character_id: str, status: RelationshipStatus) -> None:
        """
        Add or update relationship with another character.
        
        Args:
            character_id: ID of the other character
            status: Nature of the relationship
        """
        self.relationship_dynamics[character_id] = status
    
    def update_relationship(self, character_id: str, new_status: RelationshipStatus) -> None:
        """
        Update existing relationship status.
        
        Args:
            character_id: ID of the other character
            new_status: New relationship status
        """
        if character_id not in self.relationship_dynamics:
            raise ValueError(f"No existing relationship with character {character_id}")
        
        self.relationship_dynamics[character_id] = new_status
    
    def add_growth_milestone(self, milestone: str, chapter: int) -> None:
        """
        Record character growth achievement.
        
        Args:
            milestone: Description of growth or lesson learned
            chapter: Chapter where growth occurred
        """
        if chapter < self.introduction_chapter:
            raise ValueError("Growth cannot occur before character introduction")
            
        self.growth_trajectory.append(milestone)
        self.last_development_chapter = max(self.last_development_chapter, chapter)
    
    def resolve_obstacle(self, obstacle: str, resolution: str, chapter: int) -> None:
        """
        Mark obstacle as resolved and record growth.
        
        Args:
            obstacle: Obstacle being resolved
            resolution: How it was resolved
            chapter: Chapter where resolution occurred
        """
        if obstacle in self.remaining_obstacles:
            self.remaining_obstacles.remove(obstacle)
            
        growth_note = f"Resolved: {obstacle} - {resolution}"
        self.add_growth_milestone(growth_note, chapter)
    
    def is_voice_consistent(self, dialogue_sample: str) -> bool:
        """
        Check if dialogue sample is consistent with character voice.
        
        Args:
            dialogue_sample: Sample of character dialogue to validate
            
        Returns:
            True if dialogue appears consistent with established voice
            
        Note:
            This is a placeholder for more sophisticated voice analysis.
            In full implementation, would use NLP to analyze patterns.
        """
        # Placeholder implementation - would use more sophisticated analysis
        sample_lower = dialogue_sample.lower()
        
        # Check for speech patterns
        for pattern in self.voice_characteristics.speech_patterns:
            if pattern.lower() in sample_lower:
                return True
                
        # Check for dialogue quirks
        for quirk in self.voice_characteristics.dialogue_quirks:
            if quirk.lower() in sample_lower:
                return True
        
        # If no specific patterns found, consider consistent
        # (More sophisticated implementation would analyze vocabulary, formality, etc.)
        return True
    
    def get_relationship_status(self, character_id: str) -> Optional[RelationshipStatus]:
        """
        Get relationship status with specific character.
        
        Args:
            character_id: ID of other character
            
        Returns:
            Relationship status or None if no relationship tracked
        """
        return self.relationship_dynamics.get(character_id)
    
    def needs_development(self, current_chapter: int, staleness_threshold: int = 5) -> bool:
        """
        Check if character needs development based on staleness.
        
        Args:
            current_chapter: Current story chapter
            staleness_threshold: Max chapters without development
            
        Returns:
            True if character development is overdue
        """
        return self.chapters_since_development(current_chapter) >= staleness_threshold
    
    def has_unresolved_conflicts(self) -> bool:
        """
        Check if character has unresolved obstacles or conflicts.
        
        Returns:
            True if character has remaining obstacles or internal conflicts
        """
        return len(self.remaining_obstacles) > 0 or len(self.internal_conflicts) > 0


# Type alias for character collections
CharacterArcCollection = dict[str, CharacterArc]
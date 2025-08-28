"""
Chapter Objective Model

Defines specific, measurable objectives for individual chapter generation
including plot advancement, character development, and reader engagement goals.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from enum import Enum


class ObjectivePriority(str, Enum):
    """Priority level for chapter objectives."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmotionalBeat(str, Enum):
    """Types of emotional beats for chapters."""
    TENSION_BUILD = "tension_build"
    RELIEF = "relief"
    SHOCK = "shock"
    ROMANCE = "romance"
    HUMOR = "humor"
    SADNESS = "sadness"
    TRIUMPH = "triumph"
    FEAR = "fear"
    ANGER = "anger"
    HOPE = "hope"
    DESPAIR = "despair"
    REVELATION = "revelation"


class SceneType(str, Enum):
    """Types of scenes required in chapter."""
    DIALOGUE = "dialogue"
    ACTION = "action"
    INTERNAL_REFLECTION = "internal_reflection"
    DESCRIPTION = "description"
    FLASHBACK = "flashback"
    CONFRONTATION = "confrontation"
    DISCOVERY = "discovery"
    TRANSITION = "transition"
    CLIFFHANGER = "cliffhanger"


class PlotAdvancement(BaseModel):
    """Specific plot thread advancement requirement."""
    
    thread_id: str = Field(
        description="ID of plot thread to advance"
    )
    
    advancement_type: str = Field(
        description="Type of advancement (progress, complicate, resolve, introduce_conflict)"
    )
    
    advancement_description: str = Field(
        min_length=10,
        max_length=300,
        description="Specific advancement required"
    )
    
    importance: ObjectivePriority = Field(
        default=ObjectivePriority.MEDIUM,
        description="Priority level for this advancement"
    )
    
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Measurable criteria for successful advancement"
    )


class CharacterDevelopmentTarget(BaseModel):
    """Specific character development requirement."""
    
    character_id: str = Field(
        description="ID of character to develop"
    )
    
    development_type: str = Field(
        description="Type of development (growth, relationship, voice, motivation)"
    )
    
    development_goal: str = Field(
        min_length=10,
        max_length=300,
        description="Specific development objective"
    )
    
    target_scenes: List[SceneType] = Field(
        default_factory=list,
        description="Scene types where this development should occur"
    )
    
    success_metrics: List[str] = Field(
        default_factory=list,
        description="How to measure successful character development"
    )


class ReaderEngagementGoal(BaseModel):
    """Reader engagement objective for the chapter."""
    
    engagement_type: str = Field(
        description="Type of engagement (question_plant, promise_fulfill, surprise_deploy, tension_build)"
    )
    
    target_element: str = Field(
        description="Specific element to engage with (question_id, promise_id, etc.)"
    )
    
    engagement_description: str = Field(
        min_length=10,
        max_length=300,
        description="How to achieve this engagement goal"
    )
    
    expected_reader_response: str = Field(
        description="Expected emotional/intellectual response from reader"
    )


class WordCountTarget(BaseModel):
    """Word count specifications for the chapter."""
    
    target_words: int = Field(
        gt=500,
        lt=10000,
        description="Target word count for chapter"
    )
    
    min_acceptable: int = Field(
        gt=0,
        description="Minimum acceptable word count"
    )
    
    max_acceptable: int = Field(
        description="Maximum acceptable word count"
    )
    
    scene_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional word count targets per scene type"
    )
    
    @field_validator('min_acceptable')
    @classmethod
    def validate_min_acceptable(cls, v, info):
        """Ensure min is less than target."""
        if 'target_words' in info.data and v >= info.data['target_words']:
            raise ValueError('Minimum word count must be less than target')
        return v
    
    @field_validator('max_acceptable')
    @classmethod
    def validate_max_acceptable(cls, v, info):
        """Ensure max is greater than target."""
        if 'target_words' in info.data and v <= info.data['target_words']:
            raise ValueError('Maximum word count must be greater than target')
        return v


class ChapterObjective(BaseModel):
    """
    Comprehensive chapter generation objective with specific, measurable goals.
    
    Provides detailed guidance for the Generator component on what the chapter
    should accomplish in terms of plot advancement, character development,
    world building, and reader engagement.
    """
    
    chapter_number: int = Field(
        ge=1,
        description="Target chapter number"
    )
    
    primary_goal: str = Field(
        min_length=20,
        max_length=200,
        description="Main narrative goal for this chapter"
    )
    
    secondary_goals: List[str] = Field(
        default_factory=list,
        description="Additional objectives for the chapter"
    )
    
    plot_advancements: List[PlotAdvancement] = Field(
        default_factory=list,
        description="Specific plot thread progressions required"
    )
    
    character_development_targets: List[CharacterDevelopmentTarget] = Field(
        default_factory=list,
        description="Character development objectives"
    )
    
    reader_engagement_goals: List[ReaderEngagementGoal] = Field(
        default_factory=list,
        description="Reader engagement and satisfaction objectives"
    )
    
    world_building_elements: List[str] = Field(
        default_factory=list,
        description="World building aspects to incorporate or establish"
    )
    
    emotional_beats: List[EmotionalBeat] = Field(
        default_factory=list,
        description="Required emotional beats for the chapter"
    )
    
    scene_requirements: List[SceneType] = Field(
        default_factory=list,
        description="Types of scenes that must be included"
    )
    
    word_count_target: WordCountTarget = Field(
        description="Word count specifications"
    )
    
    constraints: List[str] = Field(
        default_factory=list,
        description="Things to avoid or restrictions to observe"
    )
    
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Overall success criteria for the chapter"
    )
    
    chapter_purpose: str = Field(
        min_length=50,
        max_length=500,
        description="Detailed explanation of this chapter's narrative purpose"
    )
    
    continuity_requirements: List[str] = Field(
        default_factory=list,
        description="Continuity elements that must be maintained"
    )
    
    @field_validator('plot_advancements')
    @classmethod
    def validate_plot_advancements(cls, v):
        """Ensure at least one plot advancement per chapter."""
        if not v:
            raise ValueError('Chapter must advance at least one plot thread')
        return v
    
    def get_critical_objectives(self) -> Dict[str, List]:
        """
        Get all critical priority objectives.
        
        Returns:
            Dict containing all critical priority elements
        """
        critical_elements = {
            "plot_advancements": [pa for pa in self.plot_advancements 
                                 if pa.importance == ObjectivePriority.CRITICAL],
            "character_targets": [ct for ct in self.character_development_targets 
                                 if hasattr(ct, 'importance') and ct.importance == ObjectivePriority.CRITICAL],
            "engagement_goals": [eg for eg in self.reader_engagement_goals
                                if hasattr(eg, 'priority') and eg.priority == ObjectivePriority.CRITICAL]
        }
        
        return critical_elements
    
    def get_word_count_range(self) -> tuple[int, int]:
        """
        Get acceptable word count range.
        
        Returns:
            Tuple of (min_words, max_words)
        """
        return (self.word_count_target.min_acceptable, self.word_count_target.max_acceptable)
    
    def has_emotional_beat(self, beat: EmotionalBeat) -> bool:
        """
        Check if chapter requires specific emotional beat.
        
        Args:
            beat: Emotional beat to check for
            
        Returns:
            True if beat is required
        """
        return beat in self.emotional_beats
    
    def requires_scene_type(self, scene_type: SceneType) -> bool:
        """
        Check if chapter requires specific scene type.
        
        Args:
            scene_type: Scene type to check for
            
        Returns:
            True if scene type is required
        """
        return scene_type in self.scene_requirements
    
    def get_character_development_goals(self, character_id: str) -> List[CharacterDevelopmentTarget]:
        """
        Get development targets for specific character.
        
        Args:
            character_id: Character to get targets for
            
        Returns:
            List of development targets for the character
        """
        return [target for target in self.character_development_targets 
                if target.character_id == character_id]
    
    def get_plot_advancement_goals(self, thread_id: str) -> List[PlotAdvancement]:
        """
        Get advancement goals for specific plot thread.
        
        Args:
            thread_id: Plot thread to get goals for
            
        Returns:
            List of advancement goals for the thread
        """
        return [advancement for advancement in self.plot_advancements
                if advancement.thread_id == thread_id]
    
    def calculate_complexity_score(self) -> float:
        """
        Calculate complexity score for the chapter objectives.
        
        Returns:
            Complexity score from 0.0 (simple) to 1.0 (highly complex)
        """
        complexity_factors = []
        
        # Plot advancement complexity
        plot_complexity = len(self.plot_advancements) * 0.1
        complexity_factors.append(min(1.0, plot_complexity))
        
        # Character development complexity
        char_complexity = len(self.character_development_targets) * 0.15
        complexity_factors.append(min(1.0, char_complexity))
        
        # Engagement goals complexity
        engagement_complexity = len(self.reader_engagement_goals) * 0.1
        complexity_factors.append(min(1.0, engagement_complexity))
        
        # Scene requirements complexity
        scene_complexity = len(self.scene_requirements) * 0.05
        complexity_factors.append(min(1.0, scene_complexity))
        
        # Emotional beats complexity
        emotion_complexity = len(self.emotional_beats) * 0.05
        complexity_factors.append(min(1.0, emotion_complexity))
        
        # Average complexity with some non-linear scaling
        avg_complexity = sum(complexity_factors) / len(complexity_factors)
        
        # Boost complexity for chapters with many constraints
        constraint_boost = min(0.2, len(self.constraints) * 0.02)
        
        final_complexity = min(1.0, avg_complexity + constraint_boost)
        return final_complexity
    
    def validate_objective_consistency(self) -> List[str]:
        """
        Validate that objectives don't conflict with each other.
        
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check for conflicting emotional beats
        conflicting_beats = [
            (EmotionalBeat.TRIUMPH, EmotionalBeat.DESPAIR),
            (EmotionalBeat.HUMOR, EmotionalBeat.SADNESS),
            (EmotionalBeat.HOPE, EmotionalBeat.DESPAIR),
            (EmotionalBeat.RELIEF, EmotionalBeat.TENSION_BUILD)
        ]
        
        for beat1, beat2 in conflicting_beats:
            if beat1 in self.emotional_beats and beat2 in self.emotional_beats:
                issues.append(f"Conflicting emotional beats: {beat1.value} and {beat2.value}")
        
        # Check word count feasibility vs objectives
        objective_count = (len(self.plot_advancements) + 
                          len(self.character_development_targets) + 
                          len(self.reader_engagement_goals))
        
        if objective_count > 10 and self.word_count_target.target_words < 2000:
            issues.append("Too many objectives for target word count")
        
        # Check for duplicate character targets
        character_ids = [target.character_id for target in self.character_development_targets]
        if len(character_ids) != len(set(character_ids)):
            issues.append("Duplicate character development targets found")
        
        return issues
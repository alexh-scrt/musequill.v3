"""
Chapter Variant Model

Represents a generated chapter variant with approach metadata, content analysis,
and objective fulfillment tracking for discriminator evaluation.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ChapterApproach(str, Enum):
    """Different narrative approaches for chapter generation."""
    CHARACTER_FOCUSED = "character_focused"
    PLOT_DRIVEN = "plot_driven"
    WORLD_BUILDING = "world_building"
    ACTION_HEAVY = "action_heavy"
    DIALOGUE_HEAVY = "dialogue_heavy"
    INTROSPECTIVE = "introspective"
    ATMOSPHERIC = "atmospheric"
    PACING_ACCELERATED = "pacing_accelerated"
    PACING_CONTEMPLATIVE = "pacing_contemplative"


class SceneStructure(BaseModel):
    """Structure of an individual scene within the chapter."""
    
    scene_id: str = Field(
        description="Unique identifier for this scene"
    )
    
    scene_type: str = Field(
        description="Type of scene (dialogue, action, description, etc.)"
    )
    
    word_count: int = Field(
        ge=0,
        description="Approximate word count for this scene"
    )
    
    characters_present: List[str] = Field(
        default_factory=list,
        description="Character IDs present in this scene"
    )
    
    location: str = Field(
        description="Location where scene takes place"
    )
    
    primary_purpose: str = Field(
        description="Main narrative purpose of this scene"
    )
    
    emotional_tone: str = Field(
        description="Dominant emotional tone of the scene"
    )
    
    plot_elements_advanced: List[str] = Field(
        default_factory=list,
        description="Plot thread IDs advanced in this scene"
    )


class ObjectiveFulfillment(BaseModel):
    """Tracking of how well chapter objectives were met."""
    
    objective_id: str = Field(
        description="ID of the objective being tracked"
    )
    
    objective_type: str = Field(
        description="Type of objective (plot_advancement, character_development, etc.)"
    )
    
    fulfillment_level: float = Field(
        ge=0.0,
        le=1.0,
        description="How well objective was fulfilled (0.0-1.0)"
    )
    
    fulfillment_description: str = Field(
        description="Description of how objective was addressed"
    )
    
    evidence_text: Optional[str] = Field(
        default=None,
        description="Specific text passages that fulfill this objective"
    )


class GenerationMetadata(BaseModel):
    """Metadata about the generation process for this variant."""
    
    generation_attempt: int = Field(
        ge=1,
        description="Which attempt this represents (1st, 2nd, etc.)"
    )
    
    generation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this variant was generated"
    )
    
    generation_time_seconds: float = Field(
        ge=0.0,
        description="Time taken to generate this variant"
    )
    
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM parameters used for generation"
    )
    
    prompt_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens in generation prompt"
    )
    
    completion_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens in generated completion"
    )
    
    revision_history: List[str] = Field(
        default_factory=list,
        description="History of revisions made to this variant"
    )


class ChapterVariant(BaseModel):
    """
    A generated chapter variant with comprehensive metadata and analysis.
    
    Represents one approach to fulfilling chapter objectives, containing
    the actual content plus detailed analysis for discriminator evaluation.
    """
    
    variant_id: str = Field(
        description="Unique identifier for this chapter variant"
    )
    
    chapter_number: int = Field(
        ge=1,
        description="Target chapter number"
    )
    
    approach: ChapterApproach = Field(
        description="Narrative approach taken for this variant"
    )
    
    chapter_text: str = Field(
        min_length=500,
        description="Complete chapter content"
    )
    
    chapter_title: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional chapter title"
    )
    
    word_count: int = Field(
        gt=0,
        description="Actual word count of generated content"
    )
    
    scene_structure: List[SceneStructure] = Field(
        default_factory=list,
        description="Breakdown of scenes within the chapter"
    )
    
    objectives_addressed: List[ObjectiveFulfillment] = Field(
        default_factory=list,
        description="Analysis of how well objectives were met"
    )
    
    characters_featured: List[str] = Field(
        default_factory=list,
        description="Character IDs that appear in this chapter"
    )
    
    plot_threads_advanced: List[str] = Field(
        default_factory=list,
        description="Plot thread IDs that were progressed"
    )
    
    new_world_elements: List[str] = Field(
        default_factory=list,
        description="New world building elements introduced"
    )
    
    emotional_beats_achieved: List[str] = Field(
        default_factory=list,
        description="Emotional beats successfully incorporated"
    )
    
    dialogue_percentage: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Approximate percentage of chapter that is dialogue"
    )
    
    pacing_assessment: str = Field(
        default="moderate",
        description="Self-assessed pacing (slow, moderate, fast, varied)"
    )
    
    generation_metadata: GenerationMetadata = Field(
        description="Metadata about the generation process"
    )
    
    self_critique_notes: List[str] = Field(
        default_factory=list,
        description="Generator's own assessment of strengths/weaknesses"
    )
    
    @field_validator('word_count')
    @classmethod
    def validate_word_count_accuracy(cls, v, info):
        """Validate word count matches actual content within reasonable tolerance."""
        if 'chapter_text' in info.data:
            actual_count = len(info.data['chapter_text'].split())
            tolerance = max(50, actual_count * 0.1)  # 10% tolerance or minimum 50 words
            
            if abs(v - actual_count) > tolerance:
                raise ValueError(
                    f'Declared word count {v} differs significantly from actual count {actual_count}'
                )
        return v
    
    @computed_field
    @property
    def scene_count(self) -> int:
        """Number of scenes in this chapter variant."""
        return len(self.scene_structure)
    
    @computed_field
    @property
    def average_scene_length(self) -> float:
        """Average word count per scene."""
        if not self.scene_structure:
            return 0.0
        return sum(scene.word_count for scene in self.scene_structure) / len(self.scene_structure)
    
    @computed_field
    @property
    def objective_fulfillment_score(self) -> float:
        """Overall objective fulfillment score (0.0-1.0)."""
        if not self.objectives_addressed:
            return 0.0
        
        total_score = sum(obj.fulfillment_level for obj in self.objectives_addressed)
        return total_score / len(self.objectives_addressed)
    
    @computed_field
    @property
    def character_focus_distribution(self) -> Dict[str, int]:
        """Distribution of scenes per character."""
        char_counts = {}
        for scene in self.scene_structure:
            for char_id in scene.characters_present:
                char_counts[char_id] = char_counts.get(char_id, 0) + 1
        return char_counts
    
    def get_unfulfilled_objectives(self, threshold: float = 0.7) -> List[ObjectiveFulfillment]:
        """
        Get objectives that were not adequately fulfilled.
        
        Args:
            threshold: Minimum fulfillment level to consider adequate
            
        Returns:
            List of objectives below threshold
        """
        return [obj for obj in self.objectives_addressed 
                if obj.fulfillment_level < threshold]
    
    def get_scenes_by_type(self, scene_type: str) -> List[SceneStructure]:
        """
        Get all scenes of a specific type.
        
        Args:
            scene_type: Type of scene to filter for
            
        Returns:
            List of matching scenes
        """
        return [scene for scene in self.scene_structure 
                if scene.scene_type == scene_type]
    
    def get_character_presence_ratio(self, character_id: str) -> float:
        """
        Calculate what percentage of scenes feature a specific character.
        
        Args:
            character_id: Character to analyze
            
        Returns:
            Ratio from 0.0-1.0 of scenes featuring this character
        """
        if not self.scene_structure:
            return 0.0
        
        scenes_with_character = sum(1 for scene in self.scene_structure 
                                   if character_id in scene.characters_present)
        
        return scenes_with_character / len(self.scene_structure)
    
    def calculate_narrative_density(self) -> float:
        """
        Calculate narrative density based on plot/character elements per word.
        
        Returns:
            Density score (higher = more happens per word)
        """
        if self.word_count == 0:
            return 0.0
        
        narrative_elements = (
            len(self.plot_threads_advanced) +
            len(self.characters_featured) +
            len(self.new_world_elements) +
            len(self.emotional_beats_achieved)
        )
        
        # Normalize to reasonable scale (elements per 1000 words)
        density = (narrative_elements / self.word_count) * 1000
        return min(10.0, density)  # Cap at 10 for very dense chapters
    
    def identify_pacing_issues(self) -> List[str]:
        """
        Identify potential pacing problems in the chapter structure.
        
        Returns:
            List of identified pacing issues
        """
        issues = []
        
        # Check for very uneven scene lengths
        if len(self.scene_structure) > 1:
            scene_lengths = [scene.word_count for scene in self.scene_structure]
            max_length = max(scene_lengths)
            min_length = min(scene_lengths)
            
            if max_length > min_length * 3:
                issues.append("Highly uneven scene lengths may affect pacing")
        
        # Check for too many consecutive same-type scenes
        if len(self.scene_structure) > 2:
            consecutive_count = 1
            for i in range(1, len(self.scene_structure)):
                if self.scene_structure[i].scene_type == self.scene_structure[i-1].scene_type:
                    consecutive_count += 1
                else:
                    if consecutive_count > 3:
                        issues.append(f"Too many consecutive {self.scene_structure[i-1].scene_type} scenes")
                    consecutive_count = 1
        
        # Check dialogue balance
        if self.dialogue_percentage > 0.8:
            issues.append("Very dialogue-heavy chapter may lack action/description balance")
        elif self.dialogue_percentage < 0.1:
            issues.append("Very little dialogue may make chapter feel static")
        
        return issues
    
    def get_approach_strengths(self) -> List[str]:
        """
        Get the inherent strengths of this variant's approach.
        
        Returns:
            List of approach-specific strengths
        """
        approach_strengths = {
            ChapterApproach.CHARACTER_FOCUSED: [
                "Deep character development",
                "Strong emotional resonance",
                "Rich internal perspective"
            ],
            ChapterApproach.PLOT_DRIVEN: [
                "Strong forward momentum",
                "Clear story progression", 
                "High reader engagement"
            ],
            ChapterApproach.WORLD_BUILDING: [
                "Rich setting details",
                "Immersive atmosphere",
                "Expanded story universe"
            ],
            ChapterApproach.ACTION_HEAVY: [
                "High energy and tension",
                "Strong pacing",
                "Visceral engagement"
            ],
            ChapterApproach.DIALOGUE_HEAVY: [
                "Strong character voice",
                "Natural information delivery",
                "Relationship development"
            ]
        }
        
        return approach_strengths.get(self.approach, ["Balanced narrative approach"])
    
    def validate_structural_consistency(self) -> List[str]:
        """
        Validate internal consistency of the variant structure.
        
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check scene word counts sum reasonably to total
        if self.scene_structure:
            scene_total = sum(scene.word_count for scene in self.scene_structure)
            if abs(scene_total - self.word_count) > self.word_count * 0.2:
                issues.append("Scene word counts don't align with total word count")
        
        # Check character consistency
        scene_characters = set()
        for scene in self.scene_structure:
            scene_characters.update(scene.characters_present)
        
        featured_characters = set(self.characters_featured)
        if scene_characters - featured_characters:
            issues.append("Characters appear in scenes but not in featured character list")
        
        # Check plot thread consistency
        scene_threads = set()
        for scene in self.scene_structure:
            scene_threads.update(scene.plot_elements_advanced)
        
        advanced_threads = set(self.plot_threads_advanced)
        if scene_threads - advanced_threads:
            issues.append("Plot threads advanced in scenes but not in main thread list")
        
        return issues
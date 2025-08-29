"""
Character Development Configuration

Configuration model for the character development component,
defining parameters for character arc progression and development strategies.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class CharacterDevelopmentStrategy(str, Enum):
    """Strategy for character development progression."""
    GRADUAL_ARC = "gradual_arc"           # Steady progression over multiple chapters
    CRISIS_DRIVEN = "crisis_driven"       # Development through major conflicts
    RELATIONSHIP_FOCUSED = "relationship_focused"  # Development through relationships
    INTERNAL_JOURNEY = "internal_journey"  # Focus on internal conflicts and growth
    PLOT_RESPONSIVE = "plot_responsive"    # Development responds to plot events
    MARKET_OPTIMIZED = "market_optimized"  # Development based on market preferences


class VoiceConsistencyLevel(str, Enum):
    """Level of voice consistency enforcement."""
    STRICT = "strict"      # Maintain exact voice patterns
    FLEXIBLE = "flexible"  # Allow voice evolution with development
    ADAPTIVE = "adaptive"  # Voice adapts to character growth


class CharacterDevelopmentConfig(BaseModel):
    """Configuration for character development component."""
    
    # Core Development Settings
    development_strategy: CharacterDevelopmentStrategy = Field(
        default=CharacterDevelopmentStrategy.GRADUAL_ARC,
        description="Primary strategy for character development"
    )
    
    voice_consistency_level: VoiceConsistencyLevel = Field(
        default=VoiceConsistencyLevel.FLEXIBLE,
        description="How strictly to maintain character voice consistency"
    )
    
    max_characters_per_chapter: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum number of characters to develop per chapter"
    )
    
    staleness_threshold: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Chapters without development before character is considered stale"
    )
    
    # Growth Tracking
    enable_growth_milestones: bool = Field(
        default=True,
        description="Track and record character growth milestones"
    )
    
    enable_obstacle_resolution: bool = Field(
        default=True,
        description="Track resolution of character obstacles"
    )
    
    enable_relationship_dynamics: bool = Field(
        default=True,
        description="Track and evolve character relationships"
    )
    
    # Voice and Consistency
    voice_pattern_analysis: bool = Field(
        default=True,
        description="Analyze and maintain character voice patterns"
    )
    
    dialogue_consistency_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight given to dialogue consistency in character development"
    )
    
    # Market Integration
    use_market_intelligence: bool = Field(
        default=True,
        description="Integrate market intelligence for character development"
    )
    
    market_preference_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight given to market preferences in character development"
    )
    
    # Research Integration
    enable_research_integration: bool = Field(
        default=True,
        description="Use research insights for character development"
    )
    
    research_query_templates: Dict[str, str] = Field(
        default_factory=lambda: {
            "character_archetypes": "{genre} character archetypes reader preferences",
            "development_patterns": "{genre} character development successful patterns",
            "voice_techniques": "{genre} character voice dialogue techniques",
            "relationship_dynamics": "{genre} character relationships compelling dynamics"
        },
        description="Templates for research queries by category"
    )
    
    # Development Constraints
    min_development_chapters: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum chapters between major character developments"
    )
    
    max_growth_per_chapter: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum growth milestones per chapter per character"
    )
    
    preserve_core_traits: bool = Field(
        default=True,
        description="Preserve core personality traits during development"
    )
    
    # Quality Control
    enable_voice_validation: bool = Field(
        default=True,
        description="Validate character voice consistency"
    )
    
    enable_arc_validation: bool = Field(
        default=True,
        description="Validate character arc progression"
    )
    
    development_quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Quality threshold for accepting character development"
    )
    
    # Performance Settings
    max_processing_time_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum processing time for character development"
    )
    
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing of multiple characters"
    )
    
    # Output Control
    include_development_rationale: bool = Field(
        default=True,
        description="Include rationale for development decisions in output"
    )
    
    development_detail_level: str = Field(
        default="comprehensive",
        description="Detail level for development output (basic, standard, comprehensive)"
    )
    
    # Advanced Features
    enable_predictive_development: bool = Field(
        default=False,
        description="Predict future character development needs"
    )
    
    enable_cross_character_analysis: bool = Field(
        default=True,
        description="Analyze character interactions and relationship impacts"
    )
    
    adaptive_development_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "plot_driven": 0.4,
            "character_driven": 0.4,
            "market_driven": 0.2
        },
        description="Weights for different development drivers"
    )
    
    def get_research_query(self, category: str, genre: str = "general") -> str:
        """Get research query for specific category and genre."""
        template = self.research_query_templates.get(category, "{genre} {category}")
        return template.format(genre=genre, category=category)
    
    def should_develop_character(self, chapters_since_development: int) -> bool:
        """Determine if character should be developed based on staleness."""
        return chapters_since_development >= self.staleness_threshold
    
    def get_max_growth_for_chapter(self, character_count: int) -> int:
        """Calculate maximum growth milestones for current chapter."""
        if character_count == 0:
            return 0
        return min(self.max_growth_per_chapter, max(1, self.max_growth_per_chapter // character_count))
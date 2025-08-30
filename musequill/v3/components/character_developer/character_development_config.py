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
    FAST_PROTOTYPING = "fast_prototyping"


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
    
def create_character_development_config_from_dict(config_dict: Dict[str, Any]) -> CharacterDevelopmentConfig:
    """Create CharacterDevelopmentConfig from configuration dictionary."""
    
    character_dev_settings = config_dict or {}
    
    # Map strategy string to enum
    strategy_str = character_dev_settings.get('development_strategy', 'gradual_arc')
    strategy_mapping = {
        'gradual_arc': CharacterDevelopmentStrategy.GRADUAL_ARC,
        'crisis_driven': CharacterDevelopmentStrategy.CRISIS_DRIVEN,
        'relationship_focused': CharacterDevelopmentStrategy.RELATIONSHIP_FOCUSED,
        'internal_journey': CharacterDevelopmentStrategy.INTERNAL_JOURNEY,
        'plot_responsive': CharacterDevelopmentStrategy.PLOT_RESPONSIVE,
        'market_optimized': CharacterDevelopmentStrategy.MARKET_OPTIMIZED
    }
    development_strategy = strategy_mapping.get(strategy_str.lower(), CharacterDevelopmentStrategy.GRADUAL_ARC)
    
    # Map voice consistency string to enum
    voice_str = character_dev_settings.get('voice_consistency_level', 'flexible')
    voice_mapping = {
        'strict': VoiceConsistencyLevel.STRICT,
        'flexible': VoiceConsistencyLevel.FLEXIBLE,
        'adaptive': VoiceConsistencyLevel.ADAPTIVE
    }
    voice_consistency = voice_mapping.get(voice_str.lower(), VoiceConsistencyLevel.FLEXIBLE)
    
    # Get adaptive weights with defaults
    adaptive_weights = character_dev_settings.get('adaptive_development_weights', {})
    default_adaptive_weights = {
        'plot_driven': 0.4,
        'character_driven': 0.4,
        'market_driven': 0.2
    }
    default_adaptive_weights.update(adaptive_weights)
    
    # Get research templates with defaults
    research_templates = character_dev_settings.get('research_query_templates', {})
    default_research_templates = {
        'character_archetypes': '{genre} character archetypes reader preferences',
        'development_patterns': '{genre} character development successful patterns',
        'voice_techniques': '{genre} character voice dialogue techniques',
        'relationship_dynamics': '{genre} character relationships compelling dynamics'
    }
    default_research_templates.update(research_templates)
    
    return CharacterDevelopmentConfig(
        # Core Development Settings
        development_strategy=development_strategy,
        voice_consistency_level=voice_consistency,
        max_characters_per_chapter=character_dev_settings.get('max_characters_per_chapter', 5),
        staleness_threshold=character_dev_settings.get('staleness_threshold', 5),
        
        # Growth Tracking
        enable_growth_milestones=character_dev_settings.get('enable_growth_milestones', True),
        enable_obstacle_resolution=character_dev_settings.get('enable_obstacle_resolution', True),
        enable_relationship_dynamics=character_dev_settings.get('enable_relationship_dynamics', True),
        
        # Voice and Consistency
        voice_pattern_analysis=character_dev_settings.get('voice_pattern_analysis', True),
        dialogue_consistency_weight=character_dev_settings.get('dialogue_consistency_weight', 0.7),
        
        # Market Integration
        use_market_intelligence=character_dev_settings.get('use_market_intelligence', True),
        market_preference_weight=character_dev_settings.get('market_preference_weight', 0.3),
        
        # Research Integration
        enable_research_integration=character_dev_settings.get('enable_research_integration', True),
        research_query_templates=default_research_templates,
        
        # Development Constraints
        min_development_chapters=character_dev_settings.get('min_development_chapters', 3),
        max_growth_per_chapter=character_dev_settings.get('max_growth_per_chapter', 3),
        preserve_core_traits=character_dev_settings.get('preserve_core_traits', True),
        
        # Quality Control
        enable_voice_validation=character_dev_settings.get('enable_voice_validation', True),
        enable_arc_validation=character_dev_settings.get('enable_arc_validation', True),
        development_quality_threshold=character_dev_settings.get('development_quality_threshold', 0.7),
        
        # Performance Settings
        max_processing_time_seconds=character_dev_settings.get('max_processing_time_seconds', 60),
        enable_parallel_processing=character_dev_settings.get('enable_parallel_processing', True),
        
        # Output Control
        include_development_rationale=character_dev_settings.get('include_development_rationale', True),
        development_detail_level=character_dev_settings.get('development_detail_level', 'comprehensive'),
        
        # Advanced Features
        enable_predictive_development=character_dev_settings.get('enable_predictive_development', False),
        enable_cross_character_analysis=character_dev_settings.get('enable_cross_character_analysis', True),
        adaptive_development_weights=default_adaptive_weights
    )

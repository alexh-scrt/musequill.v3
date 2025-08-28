"""
World State Model

Manages story world consistency including established facts, active rules,
location states, and temporal context for narrative continuity.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
from enum import Enum


class LocationAccessibility(str, Enum):
    """Accessibility status of story locations."""
    OPEN = "open"
    RESTRICTED = "restricted"
    CLOSED = "closed"
    UNKNOWN = "unknown"
    DESTROYED = "destroyed"


class LocationMood(str, Enum):
    """Atmospheric mood of locations."""
    WELCOMING = "welcoming"
    NEUTRAL = "neutral"
    TENSE = "tense"
    THREATENING = "threatening"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    MELANCHOLIC = "melancholic"


class TemporalPressure(str, Enum):
    """Level of time pressure in the story."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RuleType(str, Enum):
    """Types of world rules."""
    PHYSICS = "physics"
    MAGIC = "magic"
    SOCIAL = "social"
    POLITICAL = "political"
    ECONOMIC = "economic"
    CULTURAL = "cultural"
    TECHNOLOGICAL = "technological"


class EstablishedFact(BaseModel):
    """A confirmed fact in the story world."""
    
    fact_id: str = Field(
        description="Unique identifier for this fact"
    )
    
    description: str = Field(
        min_length=5,
        max_length=500,
        description="Description of the established fact"
    )
    
    chapter_established: int = Field(
        ge=1,
        description="Chapter where this fact was established"
    )
    
    importance_level: int = Field(
        ge=1,
        le=5,
        description="Importance level (1=minor detail, 5=crucial plot element)"
    )
    
    related_characters: List[str] = Field(
        default_factory=list,
        description="Character IDs related to this fact"
    )
    
    related_locations: List[str] = Field(
        default_factory=list,
        description="Location IDs related to this fact"
    )


class WorldRule(BaseModel):
    """A rule governing how the story world operates."""
    
    rule_id: str = Field(
        description="Unique identifier for this rule"
    )
    
    rule_type: RuleType = Field(
        description="Category of rule"
    )
    
    description: str = Field(
        min_length=10,
        max_length=1000,
        description="Detailed description of the rule"
    )
    
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations or exceptions to this rule"
    )
    
    chapter_introduced: int = Field(
        ge=1,
        description="Chapter where this rule was first established"
    )
    
    consistency_critical: bool = Field(
        default=True,
        description="Whether violations of this rule would break story consistency"
    )


class LocationState(BaseModel):
    """Current state and properties of a story location."""
    
    location_id: str = Field(
        description="Unique identifier for this location"
    )
    
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Name of the location"
    )
    
    accessibility: LocationAccessibility = Field(
        default=LocationAccessibility.OPEN,
        description="Current accessibility status"
    )
    
    mood_atmosphere: LocationMood = Field(
        default=LocationMood.NEUTRAL,
        description="Current atmospheric mood"
    )
    
    physical_description: str = Field(
        default="",
        max_length=1000,
        description="Physical description of the location"
    )
    
    significant_objects: List[str] = Field(
        default_factory=list,
        description="Important objects or features present"
    )
    
    characters_present: List[str] = Field(
        default_factory=list,
        description="Character IDs currently at this location"
    )
    
    last_scene_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Last chapter where a scene occurred here"
    )
    
    location_specific_rules: List[str] = Field(
        default_factory=list,
        description="Rules or constraints specific to this location"
    )


class TemporalContext(BaseModel):
    """Temporal context and time-related story elements."""
    
    current_time_period: str = Field(
        description="Current time period in the story (e.g., 'morning', 'winter 1823')"
    )
    
    time_pressure: TemporalPressure = Field(
        default=TemporalPressure.NONE,
        description="Current level of time pressure"
    )
    
    deadlines: Dict[str, str] = Field(
        default_factory=dict,
        description="Active deadlines (deadline_id -> description)"
    )
    
    seasonal_context: Optional[str] = Field(
        default=None,
        description="Current season and its story relevance"
    )
    
    time_flow_rate: str = Field(
        default="normal",
        description="How time flows in the story (normal, accelerated, compressed)"
    )
    
    significant_dates: Dict[str, str] = Field(
        default_factory=dict,
        description="Important dates and their significance"
    )


class WorldState(BaseModel):
    """
    Complete state of the story world including facts, rules, locations, and time.
    
    Maintains consistency across all world-building elements and provides
    validation for new story developments against established world rules.
    """
    
    established_facts: Dict[str, EstablishedFact] = Field(
        default_factory=dict,
        description="All confirmed facts in the story world"
    )
    
    world_rules: Dict[str, WorldRule] = Field(
        default_factory=dict,
        description="Rules governing how the world operates"
    )
    
    location_states: Dict[str, LocationState] = Field(
        default_factory=dict,
        description="Current state of all story locations"
    )
    
    temporal_context: TemporalContext = Field(
        description="Time-related context and constraints"
    )
    
    current_setting: str = Field(
        description="Primary location ID where current action is taking place"
    )
    
    world_mood: str = Field(
        default="neutral",
        max_length=100,
        description="Overall mood/atmosphere of the world state"
    )
    
    last_updated_chapter: int = Field(
        ge=1,
        description="Last chapter where world state was updated"
    )
    
    @computed_field
    @property
    def total_facts(self) -> int:
        """Total number of established facts."""
        return len(self.established_facts)
    
    @computed_field
    @property
    def critical_rules_count(self) -> int:
        """Number of consistency-critical rules."""
        return len([r for r in self.world_rules.values() if r.consistency_critical])
    
    @computed_field
    @property
    def active_locations(self) -> int:
        """Number of locations with recent activity."""
        return len([loc for loc in self.location_states.values() 
                   if loc.last_scene_chapter is not None])
    
    def add_fact(self, fact: EstablishedFact) -> None:
        """
        Add a new established fact to the world state.
        
        Args:
            fact: EstablishedFact instance to add
        """
        self.established_facts[fact.fact_id] = fact
    
    def add_rule(self, rule: WorldRule) -> None:
        """
        Add a new world rule.
        
        Args:
            rule: WorldRule instance to add
        """
        self.world_rules[rule.rule_id] = rule
    
    def add_location(self, location: LocationState) -> None:
        """
        Add a new location to the world.
        
        Args:
            location: LocationState instance to add
        """
        self.location_states[location.location_id] = location
    
    def update_location_mood(self, location_id: str, new_mood: LocationMood) -> None:
        """
        Update the mood of a specific location.
        
        Args:
            location_id: ID of location to update
            new_mood: New mood for the location
        """
        if location_id not in self.location_states:
            raise ValueError(f"Location {location_id} not found")
        
        self.location_states[location_id].mood_atmosphere = new_mood
    
    def move_character_to_location(self, character_id: str, location_id: str) -> None:
        """
        Move a character to a specific location.
        
        Args:
            character_id: ID of character to move
            location_id: Destination location ID
        """
        if location_id not in self.location_states:
            raise ValueError(f"Location {location_id} not found")
        
        # Remove character from all other locations
        for location in self.location_states.values():
            if character_id in location.characters_present:
                location.characters_present.remove(character_id)
        
        # Add character to new location
        self.location_states[location_id].characters_present.append(character_id)
    
    def get_characters_at_location(self, location_id: str) -> List[str]:
        """
        Get list of characters currently at a location.
        
        Args:
            location_id: Location to check
            
        Returns:
            List of character IDs at the location
        """
        if location_id not in self.location_states:
            return []
        
        return self.location_states[location_id].characters_present.copy()
    
    def check_rule_consistency(self, proposed_action: str, rule_type: RuleType) -> bool:
        """
        Check if a proposed action is consistent with established rules.
        
        Args:
            proposed_action: Description of proposed story development
            rule_type: Type of rule to check against
            
        Returns:
            True if action appears consistent with established rules
            
        Note:
            This is a placeholder for more sophisticated rule checking.
            Full implementation would use NLP to analyze rule violations.
        """
        # Placeholder implementation - would use more sophisticated analysis
        relevant_rules = [r for r in self.world_rules.values() 
                         if r.rule_type == rule_type and r.consistency_critical]
        
        # For now, assume consistency unless proven otherwise
        # Real implementation would analyze proposed_action against rule descriptions
        return True
    
    def get_facts_by_importance(self, min_importance: int = 1) -> List[EstablishedFact]:
        """
        Get facts filtered by importance level.
        
        Args:
            min_importance: Minimum importance level to include
            
        Returns:
            List of facts meeting importance criteria
        """
        return [fact for fact in self.established_facts.values() 
                if fact.importance_level >= min_importance]
    
    def update_scene_location(self, location_id: str, chapter: int) -> None:
        """
        Record that a scene occurred at a location in a specific chapter.
        
        Args:
            location_id: Location where scene occurred
            chapter: Chapter number
        """
        if location_id not in self.location_states:
            raise ValueError(f"Location {location_id} not found")
        
        self.location_states[location_id].last_scene_chapter = chapter
        self.current_setting = location_id
        self.last_updated_chapter = max(self.last_updated_chapter, chapter)
    
    def get_stale_locations(self, current_chapter: int, staleness_threshold: int = 10) -> List[str]:
        """
        Identify locations that haven't been used recently.
        
        Args:
            current_chapter: Current story chapter
            staleness_threshold: Chapters without activity to consider stale
            
        Returns:
            List of location IDs that are stale
        """
        stale_locations = []
        
        for location_id, location in self.location_states.items():
            if location.last_scene_chapter is None:
                stale_locations.append(location_id)
            elif current_chapter - location.last_scene_chapter >= staleness_threshold:
                stale_locations.append(location_id)
        
        return stale_locations
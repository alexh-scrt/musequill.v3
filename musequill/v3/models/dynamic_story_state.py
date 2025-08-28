"""
Dynamic Story State Model

Central container that orchestrates all story state components including plot threads,
character arcs, world state, and reader expectations. Provides unified interface
for story progression tracking and narrative intelligence.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from .plot_thread import PlotThread, ThreadStatus, PlotThreadCollection
from .character_arc import CharacterArc, CharacterArcCollection
from .world_state import WorldState
from .reader_expectations import ReaderExpectations


class NarrativeTension(str, Enum):
    """Overall narrative tension state."""
    BUILDING = "building"
    CLIMACTIC = "climactic"
    RESOLVING = "resolving"
    TRANSITIONAL = "transitional"


class StoryPhase(str, Enum):
    """Current phase in the story structure."""
    OPENING = "opening"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"


class DynamicStoryState(BaseModel):
    """
    Unified story state container that manages all narrative elements.
    
    Orchestrates plot threads, character development, world consistency,
    and reader expectations to provide comprehensive story intelligence
    for the adversarial generation system.
    """
    
    current_chapter: int = Field(
        ge=1,
        description="Current chapter number in the story"
    )
    
    total_planned_chapters: Optional[int] = Field(
        default=None,
        ge=1,
        description="Total planned chapters for the complete story"
    )
    
    plot_threads: PlotThreadCollection = Field(
        default_factory=dict,
        description="All plot threads with their current states"
    )
    
    character_arcs: CharacterArcCollection = Field(
        default_factory=dict,
        description="All character arcs with development tracking"
    )
    
    world_state: WorldState = Field(
        description="Complete world state and consistency tracking"
    )
    
    reader_expectations: ReaderExpectations = Field(
        default_factory=ReaderExpectations,
        description="Reader engagement and satisfaction tracking"
    )
    
    narrative_tension: NarrativeTension = Field(
        default=NarrativeTension.BUILDING,
        description="Overall narrative tension level"
    )
    
    story_phase: StoryPhase = Field(
        default=StoryPhase.OPENING,
        description="Current structural phase of the story"
    )
    
    momentum_score: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Calculated story momentum and forward drive"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last state update"
    )
    
    chapter_summaries: List[str] = Field(
        default_factory=list,
        description="Brief summaries of completed chapters"
    )
    
    @field_validator('total_planned_chapters')
    @classmethod
    def validate_planned_chapters(cls, v, info):
        """Ensure planned chapters is reasonable and >= current chapter."""
        if v is not None and 'current_chapter' in info.data:
            current = info.data['current_chapter']
            if v < current:
                raise ValueError(f'Planned chapters {v} cannot be less than current chapter {current}')
            if v > 200:  # Reasonable upper bound
                raise ValueError(f'Planned chapters {v} exceeds reasonable limit of 200')
        return v
    
    @computed_field
    @property
    def story_completion_ratio(self) -> float:
        """Calculate completion ratio of story (0.0-1.0)."""
        if self.total_planned_chapters is None:
            return 0.0
        return min(1.0, self.current_chapter / self.total_planned_chapters)
    
    @computed_field
    @property
    def active_thread_count(self) -> int:
        """Number of currently active plot threads."""
        return len([t for t in self.plot_threads.values() if t.status == ThreadStatus.ACTIVE])
    
    @computed_field
    @property
    def stagnant_character_count(self) -> int:
        """Number of characters needing development attention."""
        return len([c for c in self.character_arcs.values() 
                   if c.needs_development(self.current_chapter)])
    
    @computed_field
    @property
    def satisfaction_risk_level(self) -> str:
        """Current reader satisfaction risk level."""
        debt = self.reader_expectations.satisfaction_debt
        if debt >= 0.8:
            return "critical"
        elif debt >= 0.6:
            return "high"
        elif debt >= 0.4:
            return "medium"
        else:
            return "low"
    
    def calculate_momentum_score(self) -> float:
        """
        Calculate overall story momentum based on multiple factors.
        
        Returns:
            Momentum score from 0.0 (stagnant) to 1.0 (high momentum)
        """
        momentum_factors = []
        
        # Plot thread advancement factor
        active_threads = [t for t in self.plot_threads.values() if t.status == ThreadStatus.ACTIVE]
        if active_threads:
            avg_staleness = sum(t.chapters_since_advancement(self.current_chapter) 
                               for t in active_threads) / len(active_threads)
            thread_momentum = max(0.0, 1.0 - (avg_staleness / 10.0))  # Decay over 10 chapters
            momentum_factors.append(thread_momentum * 0.4)  # 40% weight
        
        # Character development factor
        if self.character_arcs:
            stagnant_ratio = self.stagnant_character_count / len(self.character_arcs)
            char_momentum = 1.0 - stagnant_ratio
            momentum_factors.append(char_momentum * 0.3)  # 30% weight
        
        # Reader engagement factor
        engagement_momentum = self.reader_expectations.engagement_momentum
        momentum_factors.append(engagement_momentum * 0.3)  # 30% weight
        
        # Calculate weighted average
        if momentum_factors:
            self.momentum_score = sum(momentum_factors)
        else:
            self.momentum_score = 0.5  # Default when no data
            
        return self.momentum_score
    
    def get_active_threads(self) -> List[PlotThread]:
        """Get all currently active plot threads."""
        return [t for t in self.plot_threads.values() if t.status == ThreadStatus.ACTIVE]
    
    def get_resolution_ready_threads(self, readiness_threshold: float = 0.8) -> List[PlotThread]:
        """Get plot threads ready for resolution."""
        return [t for t in self.plot_threads.values() 
                if t.is_ready_for_resolution(readiness_threshold)]
    
    def get_stagnant_threads(self, staleness_threshold: int = 8) -> List[PlotThread]:
        """Get plot threads that haven't advanced recently."""
        return [t for t in self.plot_threads.values()
                if t.chapters_since_advancement(self.current_chapter) >= staleness_threshold]
    
    def get_characters_needing_development(self) -> List[CharacterArc]:
        """Get characters that need development attention."""
        return [c for c in self.character_arcs.values() 
                if c.needs_development(self.current_chapter)]
    
    def get_payoff_ready_elements(self) -> Dict[str, Any]:
        """
        Get all story elements ready for payoff/resolution.
        
        Returns:
            Dict containing threads, questions, promises, and surprises ready for payoff
        """
        ready_elements = {
            "threads": [t.thread_id for t in self.get_resolution_ready_threads()],
            "characters": [c.character_id for c in self.get_characters_needing_development()],
            "expectations": self.reader_expectations.calculate_payoff_readiness(self.current_chapter)
        }
        
        return ready_elements
    
    def advance_to_chapter(self, chapter_number: int, chapter_summary: str) -> None:
        """
        Advance story state to new chapter with summary.
        
        Args:
            chapter_number: New chapter number
            chapter_summary: Brief summary of chapter events
        """
        if chapter_number <= self.current_chapter:
            raise ValueError(f"Cannot advance to chapter {chapter_number}, currently at {self.current_chapter}")
        
        # Add summary for completed chapter
        while len(self.chapter_summaries) < self.current_chapter:
            self.chapter_summaries.append("")
        
        if len(self.chapter_summaries) >= self.current_chapter:
            self.chapter_summaries[self.current_chapter - 1] = chapter_summary
        else:
            self.chapter_summaries.append(chapter_summary)
        
        # Update current chapter
        self.current_chapter = chapter_number
        self.last_updated = datetime.now()
        
        # Recalculate momentum
        self.calculate_momentum_score()
        
        # Update story phase based on completion ratio
        self._update_story_phase()
    
    def _update_story_phase(self) -> None:
        """Update story phase based on completion ratio."""
        if self.total_planned_chapters is None:
            return
            
        completion = self.story_completion_ratio
        
        if completion < 0.2:
            self.story_phase = StoryPhase.OPENING
        elif completion < 0.7:
            self.story_phase = StoryPhase.RISING_ACTION
        elif completion < 0.8:
            self.story_phase = StoryPhase.CLIMAX
        elif completion < 0.95:
            self.story_phase = StoryPhase.FALLING_ACTION
        else:
            self.story_phase = StoryPhase.RESOLUTION
    
    def add_plot_thread(self, thread: PlotThread) -> None:
        """Add new plot thread to story state."""
        self.plot_threads[thread.thread_id] = thread
        self.last_updated = datetime.now()
    
    def add_character_arc(self, character: CharacterArc) -> None:
        """Add new character arc to story state."""
        self.character_arcs[character.character_id] = character
        self.last_updated = datetime.now()
    
    def resolve_plot_thread(self, thread_id: str, resolution_chapter: int) -> None:
        """Mark plot thread as resolved."""
        if thread_id not in self.plot_threads:
            raise ValueError(f"Plot thread {thread_id} not found")
        
        self.plot_threads[thread_id].mark_resolved(resolution_chapter)
        self.last_updated = datetime.now()
    
    def advance_plot_thread(self, thread_id: str, advancement_chapter: int, tension_delta: int = 0) -> None:
        """Advance specific plot thread."""
        if thread_id not in self.plot_threads:
            raise ValueError(f"Plot thread {thread_id} not found")
        
        self.plot_threads[thread_id].advance_thread(advancement_chapter, tension_delta)
        self.last_updated = datetime.now()
    
    def get_contextual_summary(self, max_tokens: int = 2000) -> str:
        """
        Generate contextual summary for chapter generation.
        
        Args:
            max_tokens: Approximate maximum tokens for summary
            
        Returns:
            Contextual summary prioritizing most relevant information
        """
        summary_parts = []
        
        # Current state overview
        summary_parts.append(f"Chapter {self.current_chapter} | {self.story_phase.value} | {self.narrative_tension.value}")
        
        # Active plot threads (highest priority)
        active_threads = self.get_active_threads()
        if active_threads:
            thread_summary = "Active Threads: " + "; ".join([
                f"{t.title} (tension: {t.tension_level}, last: ch{t.last_advancement})" 
                for t in active_threads[:3]  # Limit to top 3
            ])
            summary_parts.append(thread_summary)
        
        # Key character states
        main_characters = list(self.character_arcs.values())[:3]  # Focus on first 3 characters
        if main_characters:
            char_summary = "Characters: " + "; ".join([
                f"{c.name} ({c.emotional_state}, {c.narrative_function.value})"
                for c in main_characters
            ])
            summary_parts.append(char_summary)
        
        # Reader expectation pressure
        if self.reader_expectations.needs_payoff():
            expectation_summary = f"Reader Satisfaction: NEEDS PAYOFF (debt: {self.reader_expectations.satisfaction_debt:.2f})"
            summary_parts.append(expectation_summary)
        
        # Recent chapter context
        if self.chapter_summaries and len(self.chapter_summaries) >= self.current_chapter - 1:
            recent_summary = f"Previous: {self.chapter_summaries[-1]}"
            summary_parts.append(recent_summary)
        
        # World state essentials
        if self.world_state.current_setting:
            location_info = f"Setting: {self.world_state.current_setting}"
            if self.world_state.location_states.get(self.world_state.current_setting):
                location = self.world_state.location_states[self.world_state.current_setting]
                location_info += f" ({location.mood_atmosphere.value})"
            summary_parts.append(location_info)
        
        full_summary = " | ".join(summary_parts)
        
        # Rough token estimation (4 chars per token)
        if len(full_summary) > max_tokens * 4:
            # Truncate to fit within token limit
            full_summary = full_summary[:max_tokens * 4 - 3] + "..."
        
        return full_summary
    
    def validate_consistency(self) -> List[str]:
        """
        Validate story state consistency and return list of issues.
        
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check for orphaned references
        thread_chars = set()
        for thread in self.plot_threads.values():
            # Plot threads should reference existing characters
            for char_id in getattr(thread, 'related_characters', []):
                if char_id not in self.character_arcs:
                    issues.append(f"Plot thread {thread.thread_id} references unknown character {char_id}")
                thread_chars.add(char_id)
        
        # Check character relationships reference existing characters
        for character in self.character_arcs.values():
            for related_char_id in character.relationship_dynamics.keys():
                if related_char_id not in self.character_arcs:
                    issues.append(f"Character {character.character_id} has relationship with unknown character {related_char_id}")
        
        # Check world state location references
        for char_id, location_state in self.world_state.location_states.items():
            for char_present in location_state.characters_present:
                if char_present not in self.character_arcs:
                    issues.append(f"Location {char_id} contains unknown character {char_present}")
        
        # Check chapter number consistency
        for thread in self.plot_threads.values():
            if thread.last_advancement > self.current_chapter:
                issues.append(f"Plot thread {thread.thread_id} advanced in future chapter {thread.last_advancement}")
        
        return issues
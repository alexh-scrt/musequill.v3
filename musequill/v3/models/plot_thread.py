"""
Plot Thread Model

Represents individual narrative threads with tension tracking, resolution readiness,
and reader investment metrics for dynamic story state management.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class ThreadStatus(str, Enum):
    """Status of a plot thread in the narrative."""
    ACTIVE = "active"
    DORMANT = "dormant"
    RESOLVED = "resolved"


class InvestmentLevel(str, Enum):
    """Reader investment level for a plot thread."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlotThread(BaseModel):
    """
    Individual plot thread with progression tracking and resolution management.
    
    Attributes:
        thread_id: Unique identifier for the plot thread
        title: Human-readable title for the thread
        status: Current status of the thread (active/dormant/resolved)
        tension_level: Subjective tension level on 1-10 scale
        last_advancement: Chapter number where thread was last advanced
        resolution_readiness: Float 0.0-1.0 indicating how ready for resolution
        reader_investment: Estimated reader investment level
        description: Detailed description of the plot thread
        setup_chapter: Chapter where thread was first introduced
        expected_resolution_chapter: Optional target chapter for resolution
    """
    
    thread_id: str = Field(
        description="Unique identifier for this plot thread"
    )
    
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Human-readable title for the plot thread"
    )
    
    status: ThreadStatus = Field(
        default=ThreadStatus.ACTIVE,
        description="Current status of the plot thread"
    )
    
    tension_level: int = Field(
        ge=1, 
        le=10, 
        description="Subjective tension level on 1-10 scale"
    )
    
    last_advancement: int = Field(
        ge=1, 
        description="Chapter number where thread was last advanced"
    )
    
    resolution_readiness: float = Field(
        ge=0.0, 
        le=1.0,
        description="Float 0.0-1.0 indicating readiness for resolution"
    )
    
    reader_investment: InvestmentLevel = Field(
        description="Estimated reader investment level in this thread"
    )
    
    description: str = Field(
        min_length=10,
        max_length=1000,
        description="Detailed description of what this plot thread represents"
    )
    
    setup_chapter: int = Field(
        ge=1,
        description="Chapter number where this thread was first introduced"
    )
    
    expected_resolution_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional target chapter number for thread resolution"
    )
    
    @field_validator('expected_resolution_chapter')
    @classmethod
    def validate_resolution_chapter(cls, v, info):
        """Ensure resolution chapter comes after setup chapter."""
        if v is not None and 'setup_chapter' in info.data:
            setup_chapter = info.data['setup_chapter']
            if v <= setup_chapter:
                raise ValueError(
                    f'Resolution chapter {v} must be after setup chapter {setup_chapter}'
                )
        return v
    
    @field_validator('last_advancement')
    @classmethod
    def validate_last_advancement(cls, v, info):
        """Ensure last advancement is not before setup chapter."""
        if 'setup_chapter' in info.data:
            setup_chapter = info.data['setup_chapter']
            if v < setup_chapter:
                raise ValueError(
                    f'Last advancement chapter {v} cannot be before setup chapter {setup_chapter}'
                )
        return v
    
    def is_ready_for_resolution(self, threshold: float = 0.8) -> bool:
        """
        Check if thread is ready for resolution based on readiness threshold.
        
        Args:
            threshold: Minimum readiness score (default 0.8)
            
        Returns:
            True if resolution_readiness >= threshold
        """
        return self.resolution_readiness >= threshold
    
    def chapters_since_advancement(self, current_chapter: int) -> int:
        """
        Calculate chapters since last advancement.
        
        Args:
            current_chapter: Current chapter number in story
            
        Returns:
            Number of chapters since last advancement
        """
        return max(0, current_chapter - self.last_advancement)
    
    def advance_thread(self, chapter_number: int, tension_delta: int = 0) -> None:
        """
        Advance the thread to a new chapter with optional tension adjustment.
        
        Args:
            chapter_number: Chapter where advancement occurs
            tension_delta: Optional adjustment to tension level (+/- int)
        """
        if chapter_number < self.last_advancement:
            raise ValueError("Cannot advance thread to earlier chapter")
            
        self.last_advancement = chapter_number
        
        if tension_delta != 0:
            new_tension = self.tension_level + tension_delta
            self.tension_level = max(1, min(10, new_tension))
    
    def mark_resolved(self, resolution_chapter: int) -> None:
        """
        Mark thread as resolved at specified chapter.
        
        Args:
            resolution_chapter: Chapter where thread is resolved
        """
        if resolution_chapter < self.setup_chapter:
            raise ValueError("Cannot resolve thread before it was set up")
            
        self.status = ThreadStatus.RESOLVED
        self.last_advancement = resolution_chapter
        self.resolution_readiness = 1.0
        self.expected_resolution_chapter = resolution_chapter


# Type alias for collections of plot threads
PlotThreadCollection = dict[str, PlotThread]
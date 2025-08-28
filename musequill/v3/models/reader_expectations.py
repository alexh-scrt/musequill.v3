"""
Reader Expectations Model

Tracks planted questions, story promises, payoff readiness, and surprise potential
for managing reader engagement and satisfaction across the narrative arc.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Dict, List, Optional, Set
from enum import Enum


class QuestionType(str, Enum):
    """Types of questions planted for readers."""
    MYSTERY = "mystery"
    CHARACTER_MOTIVATION = "character_motivation"
    RELATIONSHIP = "relationship"
    WORLD_BUILDING = "world_building"
    PLOT_OUTCOME = "plot_outcome"
    BACKSTORY = "backstory"
    IDENTITY = "identity"


class PromiseType(str, Enum):
    """Types of promises made to readers."""
    ROMANCE = "romance"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CHARACTER_GROWTH = "character_growth"
    MYSTERY_SOLUTION = "mystery_solution"
    JUSTICE = "justice"
    REVELATION = "revelation"
    REUNION = "reunion"
    TRANSFORMATION = "transformation"


class UrgencyLevel(str, Enum):
    """Urgency level for question/promise resolution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlantedQuestion(BaseModel):
    """A question planted in the reader's mind that expects resolution."""
    
    question_id: str = Field(
        description="Unique identifier for this question"
    )
    
    question_text: str = Field(
        min_length=10,
        max_length=300,
        description="The actual question or mystery presented"
    )
    
    question_type: QuestionType = Field(
        description="Category of question"
    )
    
    chapter_planted: int = Field(
        ge=1,
        description="Chapter where question was first introduced"
    )
    
    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM,
        description="How urgently this needs resolution"
    )
    
    related_characters: List[str] = Field(
        default_factory=list,
        description="Character IDs related to this question"
    )
    
    related_plot_threads: List[str] = Field(
        default_factory=list,
        description="Plot thread IDs connected to this question"
    )
    
    hints_given: List[str] = Field(
        default_factory=list,
        description="Hints or clues provided toward the answer"
    )
    
    expected_resolution_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target chapter for resolution"
    )
    
    @field_validator('expected_resolution_chapter')
    @classmethod
    def validate_resolution_chapter(cls, v, info):
        """Ensure resolution comes after planting."""
        if v is not None and 'chapter_planted' in info.data:
            if v <= info.data['chapter_planted']:
                raise ValueError(
                    f'Resolution chapter {v} must be after planted chapter {info.data["chapter_planted"]}'
                )
        return v


class StoryPromise(BaseModel):
    """A promise made to readers that requires fulfillment."""
    
    promise_id: str = Field(
        description="Unique identifier for this promise"
    )
    
    promise_description: str = Field(
        min_length=10,
        max_length=500,
        description="What was promised to the reader"
    )
    
    promise_type: PromiseType = Field(
        description="Category of promise"
    )
    
    chapter_made: int = Field(
        ge=1,
        description="Chapter where promise was established"
    )
    
    fulfillment_urgency: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM,
        description="How urgent fulfillment is"
    )
    
    related_characters: List[str] = Field(
        default_factory=list,
        description="Characters involved in promise fulfillment"
    )
    
    setup_requirements: List[str] = Field(
        default_factory=list,
        description="What needs to be established before fulfillment"
    )
    
    payoff_potential: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated reader satisfaction potential (0.0-1.0)"
    )
    
    expected_fulfillment_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target chapter for promise fulfillment"
    )


class SurpriseElement(BaseModel):
    """Potential surprise or twist element held in reserve."""
    
    surprise_id: str = Field(
        description="Unique identifier for this surprise"
    )
    
    surprise_description: str = Field(
        min_length=10,
        max_length=500,
        description="Description of the surprise element"
    )
    
    surprise_impact: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated impact/shock value (0.0-1.0)"
    )
    
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Story elements that must be established first"
    )
    
    affected_characters: List[str] = Field(
        default_factory=list,
        description="Characters affected by this surprise"
    )
    
    affected_plot_threads: List[str] = Field(
        default_factory=list,
        description="Plot threads affected by this surprise"
    )
    
    optimal_timing_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optimal chapter for maximum impact"
    )
    
    used: bool = Field(
        default=False,
        description="Whether this surprise has been deployed"
    )


class ReaderExpectations(BaseModel):
    """
    Comprehensive tracking of reader expectations, questions, promises, and surprises.
    
    Manages the balance between answering questions, fulfilling promises,
    and maintaining engagement through strategic surprise deployment.
    """
    
    planted_questions: Dict[str, PlantedQuestion] = Field(
        default_factory=dict,
        description="Questions planted in reader's mind awaiting answers"
    )
    
    story_promises: Dict[str, StoryPromise] = Field(
        default_factory=dict,
        description="Promises made to readers requiring fulfillment"
    )
    
    surprise_elements: Dict[str, SurpriseElement] = Field(
        default_factory=dict,
        description="Surprise elements available for deployment"
    )
    
    satisfaction_debt: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Accumulated unfulfilled reader expectations (0.0-1.0)"
    )
    
    engagement_momentum: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Current reader engagement momentum (0.0-1.0)"
    )
    
    last_payoff_chapter: Optional[int] = Field(
        default=None,
        ge=1,
        description="Last chapter that provided significant payoff"
    )
    
    @computed_field
    @property
    def total_questions(self) -> int:
        """Total number of planted questions."""
        return len(self.planted_questions)
    
    @computed_field
    @property
    def urgent_questions(self) -> int:
        """Number of high/critical urgency questions."""
        return len([q for q in self.planted_questions.values() 
                   if q.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]])
    
    @computed_field
    @property
    def urgent_promises(self) -> int:
        """Number of high/critical urgency promises."""
        return len([p for p in self.story_promises.values()
                   if p.fulfillment_urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]])
    
    @computed_field
    @property
    def unused_surprises(self) -> int:
        """Number of unused surprise elements."""
        return len([s for s in self.surprise_elements.values() if not s.used])
    
    def plant_question(self, question: PlantedQuestion) -> None:
        """
        Add a new question to reader expectations.
        
        Args:
            question: PlantedQuestion instance to add
        """
        self.planted_questions[question.question_id] = question
        
        # Increase satisfaction debt for new questions
        self.satisfaction_debt = min(1.0, self.satisfaction_debt + 0.1)
    
    def make_promise(self, promise: StoryPromise) -> None:
        """
        Add a new promise to fulfill.
        
        Args:
            promise: StoryPromise instance to add
        """
        self.story_promises[promise.promise_id] = promise
        
        # Increase satisfaction debt for new promises
        debt_increase = promise.payoff_potential * 0.2
        self.satisfaction_debt = min(1.0, self.satisfaction_debt + debt_increase)
    
    def answer_question(self, question_id: str, chapter: int, satisfaction_level: float = 1.0) -> None:
        """
        Mark a question as answered and reduce satisfaction debt.
        
        Args:
            question_id: ID of question being answered
            chapter: Chapter where answer is provided
            satisfaction_level: How satisfying the answer is (0.0-1.0)
        """
        if question_id not in self.planted_questions:
            raise ValueError(f"Question {question_id} not found")
        
        question = self.planted_questions[question_id]
        
        # Remove from planted questions
        del self.planted_questions[question_id]
        
        # Reduce satisfaction debt based on answer quality
        debt_reduction = 0.15 * satisfaction_level
        if question.urgency_level == UrgencyLevel.CRITICAL:
            debt_reduction *= 1.5
        
        self.satisfaction_debt = max(0.0, self.satisfaction_debt - debt_reduction)
        self.last_payoff_chapter = chapter
        
        # Boost engagement momentum for satisfying answers
        momentum_boost = 0.1 * satisfaction_level
        self.engagement_momentum = min(1.0, self.engagement_momentum + momentum_boost)
    
    def fulfill_promise(self, promise_id: str, chapter: int, fulfillment_quality: float = 1.0) -> None:
        """
        Mark a promise as fulfilled and provide reader satisfaction.
        
        Args:
            promise_id: ID of promise being fulfilled
            chapter: Chapter where fulfillment occurs
            fulfillment_quality: How well promise was fulfilled (0.0-1.0)
        """
        if promise_id not in self.story_promises:
            raise ValueError(f"Promise {promise_id} not found")
        
        promise = self.story_promises[promise_id]
        
        # Remove from active promises
        del self.story_promises[promise_id]
        
        # Reduce satisfaction debt based on fulfillment quality
        debt_reduction = promise.payoff_potential * fulfillment_quality * 0.3
        self.satisfaction_debt = max(0.0, self.satisfaction_debt - debt_reduction)
        self.last_payoff_chapter = chapter
        
        # Major boost to engagement momentum for good fulfillment
        momentum_boost = promise.payoff_potential * fulfillment_quality * 0.2
        self.engagement_momentum = min(1.0, self.engagement_momentum + momentum_boost)
    
    def deploy_surprise(self, surprise_id: str, chapter: int) -> None:
        """
        Deploy a surprise element.
        
        Args:
            surprise_id: ID of surprise to deploy
            chapter: Chapter where surprise is revealed
        """
        if surprise_id not in self.surprise_elements:
            raise ValueError(f"Surprise {surprise_id} not found")
        
        surprise = self.surprise_elements[surprise_id]
        surprise.used = True
        
        # Surprises boost engagement momentum
        momentum_boost = surprise.surprise_impact * 0.25
        self.engagement_momentum = min(1.0, self.engagement_momentum + momentum_boost)
    
    def get_overdue_questions(self, current_chapter: int, staleness_threshold: int = 8) -> List[PlantedQuestion]:
        """
        Get questions that have been unanswered too long.
        
        Args:
            current_chapter: Current story chapter
            staleness_threshold: Chapters without answer to consider overdue
            
        Returns:
            List of overdue questions
        """
        overdue = []
        for question in self.planted_questions.values():
            chapters_waiting = current_chapter - question.chapter_planted
            if chapters_waiting >= staleness_threshold:
                overdue.append(question)
        
        return overdue
    
    def get_ready_surprises(self, current_chapter: int) -> List[SurpriseElement]:
        """
        Get surprise elements ready for deployment.
        
        Args:
            current_chapter: Current story chapter
            
        Returns:
            List of surprises ready to deploy
        """
        ready = []
        for surprise in self.surprise_elements.values():
            if not surprise.used:
                if surprise.optimal_timing_chapter is None:
                    ready.append(surprise)
                elif current_chapter >= surprise.optimal_timing_chapter:
                    ready.append(surprise)
        
        return ready
    
    def calculate_payoff_readiness(self, current_chapter: int) -> Dict[str, List[str]]:
        """
        Identify elements ready for payoff.
        
        Args:
            current_chapter: Current story chapter
            
        Returns:
            Dict with 'questions', 'promises', and 'surprises' ready for payoff
        """
        ready = {
            "questions": [],
            "promises": [],
            "surprises": []
        }
        
        # Questions ready for answering
        for q_id, question in self.planted_questions.items():
            if question.expected_resolution_chapter and current_chapter >= question.expected_resolution_chapter:
                ready["questions"].append(q_id)
            elif question.urgency_level == UrgencyLevel.CRITICAL:
                ready["questions"].append(q_id)
        
        # Promises ready for fulfillment
        for p_id, promise in self.story_promises.items():
            if promise.expected_fulfillment_chapter and current_chapter >= promise.expected_fulfillment_chapter:
                ready["promises"].append(p_id)
            elif promise.fulfillment_urgency == UrgencyLevel.CRITICAL:
                ready["promises"].append(p_id)
        
        # Ready surprises
        ready_surprises = self.get_ready_surprises(current_chapter)
        ready["surprises"] = [s.surprise_id for s in ready_surprises]
        
        return ready
    
    def needs_payoff(self, urgency_threshold: float = 0.7) -> bool:
        """
        Check if story urgently needs payoff to maintain reader satisfaction.
        
        Args:
            urgency_threshold: Satisfaction debt level that triggers urgent need
            
        Returns:
            True if payoff is urgently needed
        """
        return (self.satisfaction_debt >= urgency_threshold or 
                self.engagement_momentum <= 0.3 or
                self.urgent_questions > 0 or 
                self.urgent_promises > 0)
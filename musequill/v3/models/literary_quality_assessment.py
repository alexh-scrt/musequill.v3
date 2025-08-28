"""
Literary Quality Assessment Model

Evaluates prose quality, language freshness, character voice consistency,
and pacing for the Literary Quality Critic in the adversarial system.
"""

from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class ClicheSeverity(str, Enum):
    """Severity levels for cliché usage."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    EXCESSIVE = "excessive"


class VoiceInconsistencyType(str, Enum):
    """Types of character voice inconsistencies."""
    VOCABULARY = "vocabulary"
    SYNTAX = "syntax"
    FORMALITY = "formality"
    DIALECT = "dialect"
    EMOTIONAL_EXPRESSION = "emotional_expression"
    SPEECH_PATTERNS = "speech_patterns"


class PacingIssueType(str, Enum):
    """Types of pacing problems."""
    TOO_SLOW = "too_slow"
    TOO_FAST = "too_fast"
    UNEVEN = "uneven"
    POOR_TRANSITIONS = "poor_transitions"
    INFO_DUMP = "info_dump"
    STAGNANT = "stagnant"


class ProseQualityIssue(str, Enum):
    """Types of prose quality issues."""
    REPETITIVE_STRUCTURE = "repetitive_structure"
    WEAK_VERBS = "weak_verbs"
    PASSIVE_VOICE_OVERUSE = "passive_voice_overuse"
    ADVERB_OVERUSE = "adverb_overuse"
    TELLING_NOT_SHOWING = "telling_not_showing"
    UNCLEAR_ANTECEDENTS = "unclear_antecedents"
    AWKWARD_PHRASING = "awkward_phrasing"


class ClicheFlag(BaseModel):
    """A specific cliché or overused phrase identified in the text."""
    
    cliche_id: str = Field(
        description="Unique identifier for this cliché instance"
    )
    
    cliche_text: str = Field(
        min_length=3,
        max_length=200,
        description="The actual clichéd text"
    )
    
    cliche_type: str = Field(
        description="Type of cliché (metaphor, description, dialogue, action)"
    )
    
    severity: ClicheSeverity = Field(
        description="How problematic this cliché is"
    )
    
    context: str = Field(
        max_length=500,
        description="Surrounding context where cliché appears"
    )
    
    replacement_suggestions: List[str] = Field(
        default_factory=list,
        description="Alternative phrasings to replace the cliché"
    )
    
    frequency_in_chapter: int = Field(
        ge=1,
        description="Number of times this or similar cliché appears"
    )


class VoiceConsistencyIssue(BaseModel):
    """Character voice consistency problem."""
    
    character_id: str = Field(
        description="Character whose voice is inconsistent"
    )
    
    inconsistency_type: VoiceInconsistencyType = Field(
        description="Type of voice inconsistency"
    )
    
    description: str = Field(
        min_length=20,
        max_length=300,
        description="Description of the inconsistency"
    )
    
    conflicting_examples: List[str] = Field(
        default_factory=list,
        description="Examples of conflicting voice usage"
    )
    
    expected_voice_pattern: str = Field(
        description="How this character should speak/think based on established voice"
    )
    
    correction_suggestion: str = Field(
        description="How to fix this voice inconsistency"
    )


class PacingAnalysis(BaseModel):
    """Analysis of chapter pacing and rhythm."""
    
    overall_pace: str = Field(
        description="Overall pacing assessment (slow, moderate, fast, varied)"
    )
    
    sentence_rhythm_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Variety and flow of sentence structures"
    )
    
    paragraph_flow_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Smoothness of paragraph transitions"
    )
    
    scene_transition_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of transitions between scenes"
    )
    
    information_density: float = Field(
        ge=0.0,
        le=1.0,
        description="Appropriate balance of information vs. action"
    )
    
    pacing_issues: List[PacingIssueType] = Field(
        default_factory=list,
        description="Specific pacing problems identified"
    )
    
    pacing_strengths: List[str] = Field(
        default_factory=list,
        description="Well-executed pacing elements"
    )


class ProseAnalysis(BaseModel):
    """Detailed analysis of prose quality and style."""
    
    sentence_variety_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Variety in sentence length and structure"
    )
    
    word_choice_sophistication: float = Field(
        ge=0.0,
        le=1.0,
        description="Sophistication and precision of word choices"
    )
    
    show_vs_tell_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Balance of showing vs. telling (higher = more showing)"
    )
    
    sensory_detail_richness: float = Field(
        ge=0.0,
        le=1.0,
        description="Use of sensory details for immersion"
    )
    
    dialogue_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Naturalness and effectiveness of dialogue"
    )
    
    prose_issues: List[ProseQualityIssue] = Field(
        default_factory=list,
        description="Specific prose quality issues"
    )
    
    stylistic_strengths: List[str] = Field(
        default_factory=list,
        description="Notable prose strengths"
    )


class LanguageFreshnessAnalysis(BaseModel):
    """Analysis of language originality and freshness."""
    
    metaphor_originality: float = Field(
        ge=0.0,
        le=1.0,
        description="Originality of metaphors and comparisons"
    )
    
    descriptive_creativity: float = Field(
        ge=0.0,
        le=1.0,
        description="Creativity in descriptive language"
    )
    
    phrase_uniqueness: float = Field(
        ge=0.0,
        le=1.0,
        description="Uniqueness of phrases and expressions"
    )
    
    cliche_density: float = Field(
        ge=0.0,
        le=1.0,
        description="Density of clichéd expressions (lower is better)"
    )
    
    vocabulary_richness: float = Field(
        ge=0.0,
        le=1.0,
        description="Richness and variety of vocabulary"
    )
    
    innovation_examples: List[str] = Field(
        default_factory=list,
        description="Examples of particularly fresh or creative language"
    )


class LiteraryQualityAssessment(BaseModel):
    """
    Comprehensive assessment of literary quality and prose excellence.
    
    Evaluates language freshness, character voice consistency, pacing,
    and overall prose quality for the Literary Quality Critic.
    """
    
    chapter_number: int = Field(
        ge=1,
        description="Chapter being assessed"
    )
    
    language_freshness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall originality and freshness of language"
    )
    
    character_voice_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Consistency and authenticity of character voices"
    )
    
    pacing_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Effectiveness of pacing and rhythm"
    )
    
    prose_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Technical prose quality and craftsmanship"
    )
    
    cliche_flags: List[ClicheFlag] = Field(
        default_factory=list,
        description="Clichéd or overused expressions identified"
    )
    
    voice_consistency_issues: List[VoiceConsistencyIssue] = Field(
        default_factory=list,
        description="Character voice inconsistency problems"
    )
    
    pacing_analysis: PacingAnalysis = Field(
        description="Detailed pacing assessment"
    )
    
    prose_analysis: ProseAnalysis = Field(
        description="Detailed prose quality analysis"
    )
    
    language_freshness_analysis: LanguageFreshnessAnalysis = Field(
        description="Analysis of language originality"
    )
    
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific suggestions for literary quality improvement"
    )
    
    notable_strengths: List[str] = Field(
        default_factory=list,
        description="Particularly strong literary elements"
    )
    
    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall literary quality score."""
        return (
            self.language_freshness_score * 0.25 +
            self.character_voice_score * 0.25 +
            self.pacing_score * 0.25 +
            self.prose_quality_score * 0.25
        )
    
    @computed_field
    @property
    def cliche_severity_breakdown(self) -> Dict[str, int]:
        """Count clichés by severity level."""
        breakdown = {severity.value: 0 for severity in ClicheSeverity}
        for cliche in self.cliche_flags:
            breakdown[cliche.severity.value] += 1
        return breakdown
    
    @computed_field
    @property
    def voice_issue_count(self) -> int:
        """Total number of voice consistency issues."""
        return len(self.voice_consistency_issues)
    
    @computed_field
    @property
    def prose_issue_count(self) -> int:
        """Total number of prose quality issues."""
        return len(self.prose_analysis.prose_issues)
    
    def get_cliches_by_severity(self, severity: ClicheSeverity) -> List[ClicheFlag]:
        """Get all clichés of specified severity level."""
        return [cliche for cliche in self.cliche_flags 
                if cliche.severity == severity]
    
    def get_voice_issues_by_character(self, character_id: str) -> List[VoiceConsistencyIssue]:
        """Get voice consistency issues for specific character."""
        return [issue for issue in self.voice_consistency_issues 
                if issue.character_id == character_id]
    
    def calculate_readability_score(self) -> float:
        """
        Calculate overall readability based on multiple factors.
        
        Returns:
            Readability score from 0.0 (difficult) to 1.0 (very readable)
        """
        readability_factors = []
        
        # Sentence variety contributes to readability
        readability_factors.append(self.prose_analysis.sentence_variety_score)
        
        # Balanced pacing improves readability
        readability_factors.append(self.pacing_score)
        
        # Clear, unclichéd language is more readable
        cliche_penalty = min(1.0, len(self.cliche_flags) / 10.0)  # Penalty for many clichés
        language_clarity = max(0.0, 1.0 - cliche_penalty)
        readability_factors.append(language_clarity)
        
        # Good dialogue quality helps readability
        readability_factors.append(self.prose_analysis.dialogue_quality)
        
        return sum(readability_factors) / len(readability_factors)
    
    def identify_priority_improvements(self) -> List[str]:
        """Generate prioritized list of most important improvements."""
        priorities = []
        
        # Critical clichés first
        critical_cliches = self.get_cliches_by_severity(ClicheSeverity.EXCESSIVE)
        if critical_cliches:
            priorities.append(f"Remove excessive clichés: {len(critical_cliches)} instances found")
        
        # Voice consistency issues
        if self.voice_issue_count > 0:
            priorities.append(f"Fix character voice inconsistencies: {self.voice_issue_count} issues")
        
        # Major prose issues
        major_prose_issues = [issue for issue in self.prose_analysis.prose_issues
                             if issue in [ProseQualityIssue.TELLING_NOT_SHOWING,
                                        ProseQualityIssue.REPETITIVE_STRUCTURE,
                                        ProseQualityIssue.AWKWARD_PHRASING]]
        if major_prose_issues:
            priorities.append(f"Address major prose issues: {', '.join(issue.value for issue in major_prose_issues)}")
        
        # Pacing problems
        if PacingIssueType.INFO_DUMP in self.pacing_analysis.pacing_issues:
            priorities.append("Break up information dumps for better pacing")
        
        # Overall quality thresholds
        if self.overall_score < 0.6:
            priorities.append("Overall literary quality needs significant improvement")
        
        return priorities[:5]
    
    def assess_commercial_viability(self) -> Tuple[float, List[str]]:
        """
        Assess commercial viability from literary quality perspective.
        
        Returns:
            Tuple of (viability_score, concerns_list)
        """
        viability_score = 0.0
        concerns = []
        
        # Readability is crucial for commercial success
        readability = self.calculate_readability_score()
        viability_score += readability * 0.4
        
        if readability < 0.6:
            concerns.append("Poor readability may limit commercial appeal")
        
        # Voice consistency important for reader engagement
        voice_penalty = min(0.3, self.voice_issue_count * 0.05)
        viability_score += max(0.0, 0.3 - voice_penalty)
        
        if self.voice_issue_count > 3:
            concerns.append("Character voice inconsistencies may confuse readers")
        
        # Excessive clichés hurt commercial prospects
        excessive_cliches = len(self.get_cliches_by_severity(ClicheSeverity.EXCESSIVE))
        cliche_penalty = min(0.2, excessive_cliches * 0.05)
        viability_score += max(0.0, 0.2 - cliche_penalty)
        
        if excessive_cliches > 2:
            concerns.append("Excessive clichés may be perceived as low-quality writing")
        
        # Good pacing essential for page-turning quality
        viability_score += self.pacing_score * 0.1
        
        if self.pacing_score < 0.5:
            concerns.append("Poor pacing may reduce reader engagement")
        
        return (min(1.0, viability_score), concerns)
    
    def get_genre_specific_feedback(self, genre: str) -> List[str]:
        """
        Provide genre-specific literary quality feedback.
        
        Args:
            genre: Target genre for the work
            
        Returns:
            List of genre-specific recommendations
        """
        feedback = []
        
        genre_lower = genre.lower()
        
        if "romance" in genre_lower:
            if self.prose_analysis.sensory_detail_richness < 0.6:
                feedback.append("Romance readers expect rich sensory details for emotional connection")
            if self.character_voice_score < 0.7:
                feedback.append("Strong character voices crucial for romance reader investment")
        
        elif "thriller" in genre_lower or "suspense" in genre_lower:
            if self.pacing_score < 0.7:
                feedback.append("Thriller/suspense requires tight pacing to maintain tension")
            if PacingIssueType.TOO_SLOW in self.pacing_analysis.pacing_issues:
                feedback.append("Slow pacing undermines thriller/suspense effectiveness")
        
        elif "literary" in genre_lower:
            if self.language_freshness_score < 0.8:
                feedback.append("Literary fiction demands high language originality and freshness")
            if self.prose_quality_score < 0.8:
                feedback.append("Literary fiction requires exceptional prose craftsmanship")
        
        elif "fantasy" in genre_lower or "sci-fi" in genre_lower or "science fiction" in genre_lower:
            if self.prose_analysis.sensory_detail_richness < 0.7:
                feedback.append("SFF readers expect immersive world-building through rich description")
            if len(self.cliche_flags) > 5:
                feedback.append("SFF benefits from fresh, original imagery to support world-building")
        
        return feedback
    
    def is_publication_ready(self, genre: str = "general", threshold: float = 0.7) -> bool:
        """
        Determine if literary quality meets publication standards.
        
        Args:
            genre: Target genre for assessment
            threshold: Minimum acceptable overall score
            
        Returns:
            True if quality meets publication standards
        """
        # Must meet overall threshold
        if self.overall_score < threshold:
            return False
        
        # Cannot have excessive clichés
        if len(self.get_cliches_by_severity(ClicheSeverity.EXCESSIVE)) > 0:
            return False
        
        # Cannot have too many voice consistency issues
        if self.voice_issue_count > 2:
            return False
        
        # Genre-specific requirements
        genre_lower = genre.lower()
        if "literary" in genre_lower and self.language_freshness_score < 0.75:
            return False
        
        if ("thriller" in genre_lower or "suspense" in genre_lower) and self.pacing_score < 0.65:
            return False
        
        return True
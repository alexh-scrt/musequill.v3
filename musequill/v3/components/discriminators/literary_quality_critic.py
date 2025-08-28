"""
Literary Quality Critic Component

Implements prose quality evaluation, originality assessment, character voice analysis,
and pacing evaluation for the adversarial system discriminator layer.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState


class LiteraryQualityCriticConfig(BaseModel):
    """Configuration for Literary Quality Critic component."""
    
    prose_quality_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for acceptable prose quality"
    )
    
    originality_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for acceptable originality"
    )
    
    voice_authenticity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for character voice authenticity"
    )
    
    cliche_detection_sensitivity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Sensitivity for cliché detection"
    )
    
    max_analysis_time_seconds: int = Field(
        default=45,
        ge=5,
        le=300,
        description="Maximum time to spend analyzing a chapter"
    )
    
    enable_style_analysis: bool = Field(
        default=True,
        description="Whether to perform detailed style analysis"
    )
    
    enable_vocabulary_analysis: bool = Field(
        default=True,
        description="Whether to analyze vocabulary richness"
    )
    
    minimum_word_count: int = Field(
        default=100,
        ge=50,
        description="Minimum word count for meaningful analysis"
    )


class StyleIssue(BaseModel):
    """Represents a style issue found in the text."""
    
    issue_type: str = Field(description="Type of style issue")
    description: str = Field(description="Description of the issue")
    severity: str = Field(description="Severity: low, medium, high, critical")
    location: str = Field(description="Where the issue was found")
    suggestion: Optional[str] = Field(default=None, description="Improvement suggestion")


class VoiceAnalysis(BaseModel):
    """Analysis of character voice authenticity."""
    
    character_id: str = Field(description="Character identifier")
    voice_consistency_score: float = Field(ge=0.0, le=1.0, description="Voice consistency score")
    dialogue_authenticity: float = Field(ge=0.0, le=1.0, description="Dialogue authenticity")
    internal_thoughts_consistency: float = Field(ge=0.0, le=1.0, description="Internal thoughts consistency")
    issues: List[str] = Field(default_factory=list, description="Voice inconsistency issues")


class LiteraryQualityAssessment(BaseModel):
    """Assessment of literary quality for a chapter variant."""
    
    chapter_number: int = Field(description="Chapter number analyzed")
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall literary quality score")
    
    # Component scores
    prose_quality_score: float = Field(ge=0.0, le=1.0, description="Prose quality score")
    originality_score: float = Field(ge=0.0, le=1.0, description="Originality score")
    voice_authenticity_score: float = Field(ge=0.0, le=1.0, description="Character voice score")
    pacing_score: float = Field(ge=0.0, le=1.0, description="Pacing quality score")
    vocabulary_richness_score: float = Field(ge=0.0, le=1.0, description="Vocabulary richness")
    
    # Detailed analysis
    style_issues: List[StyleIssue] = Field(default_factory=list, description="Style issues found")
    voice_analysis: List[VoiceAnalysis] = Field(default_factory=list, description="Voice analysis per character")
    cliches_detected: List[str] = Field(default_factory=list, description="Clichéd phrases detected")
    
    # Strengths and suggestions
    strengths: List[str] = Field(default_factory=list, description="Literary strengths identified")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")
    
    # Metrics
    word_count: int = Field(description="Total word count analyzed")
    unique_word_ratio: float = Field(description="Ratio of unique words to total words")
    average_sentence_length: float = Field(description="Average sentence length")
    reading_level_score: float = Field(description="Estimated reading level difficulty")


class LiteraryQualityCriticInput(BaseModel):
    """Input data for Literary Quality Critic."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for voice consistency checking"
    )
    
    previous_assessments: List[LiteraryQualityAssessment] = Field(
        default_factory=list,
        description="Previous chapter assessments for consistency"
    )


class LiteraryQualityCritic(BaseComponent[LiteraryQualityCriticInput, LiteraryQualityAssessment, LiteraryQualityCriticConfig]):
    """
    Literary Quality Critic component for prose and style evaluation.
    
    Analyzes chapter variants for prose quality, originality, character voice
    authenticity, pacing, and overall literary merit.
    """
    
    def __init__(self, config: ComponentConfiguration[LiteraryQualityCriticConfig]):
        super().__init__(config)
        self._cliche_database: Set[str] = set()
        self._character_voice_patterns: Dict[str, Dict[str, Any]] = {}
        self._style_analysis_tools: Dict[str, Any] = {}
        self._vocabulary_analyzer: Optional[Any] = None
    
    async def initialize(self) -> bool:
        """Initialize the literary quality analysis systems."""
        try:
            # Initialize cliché database
            await self._load_cliche_database()
            
            # Initialize character voice tracking
            self._character_voice_patterns = {}
            
            # Initialize style analysis tools
            await self._initialize_style_analysis_tools()
            
            # Initialize vocabulary analyzer
            await self._initialize_vocabulary_analyzer()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Literary quality critic initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: LiteraryQualityCriticInput) -> LiteraryQualityAssessment:
        """
        Analyze chapter variant for literary quality.
        
        Args:
            input_data: Chapter variant and context for analysis
            
        Returns:
            Comprehensive literary quality assessment
        """
        start_time = datetime.now()
        chapter = input_data.chapter_variant
        story_state = input_data.story_state
        
        try:
            # Extract text content
            text_content = await self._extract_text_content(chapter)
            
            if len(text_content.split()) < self.config.specific_config.minimum_word_count:
                raise ComponentError("Chapter too short for meaningful literary analysis")
            
            # Update character voice patterns from story state
            await self._update_character_voice_patterns(story_state)
            
            # Perform parallel analysis
            analysis_tasks = [
                self._analyze_prose_quality(text_content),
                self._analyze_originality(text_content),
                self._analyze_character_voices(text_content, chapter, story_state),
                self._analyze_pacing(text_content),
                self._analyze_vocabulary_richness(text_content),
                self._detect_cliches(text_content),
                self._identify_style_issues(text_content)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            prose_result = results[0]
            originality_result = results[1] 
            voice_result = results[2]
            pacing_result = results[3]
            vocabulary_result = results[4]
            cliche_result = results[5]
            style_result = results[6]
            
            # Calculate overall score with weights
            overall_score = self._calculate_overall_score(
                prose_result['score'],
                originality_result['score'],
                voice_result['overall_score'],
                pacing_result['score'],
                vocabulary_result['score']
            )
            
            # Compile assessment
            assessment = LiteraryQualityAssessment(
                chapter_number=chapter.chapter_number,
                overall_score=overall_score,
                prose_quality_score=prose_result['score'],
                originality_score=originality_result['score'],
                voice_authenticity_score=voice_result['overall_score'],
                pacing_score=pacing_result['score'],
                vocabulary_richness_score=vocabulary_result['score'],
                style_issues=style_result['issues'],
                voice_analysis=voice_result['character_analyses'],
                cliches_detected=cliche_result['cliches'],
                strengths=self._identify_strengths(results),
                improvement_suggestions=self._generate_improvement_suggestions(results),
                word_count=vocabulary_result['word_count'],
                unique_word_ratio=vocabulary_result['unique_ratio'],
                average_sentence_length=vocabulary_result['avg_sentence_length'],
                reading_level_score=vocabulary_result['reading_level']
            )
            
            return assessment
            
        except Exception as e:
            raise ComponentError(f"Literary quality analysis failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on literary quality analysis systems."""
        try:
            # Check cliché database is loaded
            if not self._cliche_database:
                return False
            
            # Test style analysis tools
            if self.config.specific_config.enable_style_analysis:
                test_result = await self._test_style_analysis()
                if not test_result:
                    return False
            
            # Test vocabulary analyzer
            if self.config.specific_config.enable_vocabulary_analysis:
                test_result = await self._test_vocabulary_analyzer()
                if not test_result:
                    return False
            
            # Check component metrics
            if self.state.metrics.failure_rate > 0.2:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup literary quality analysis resources."""
        try:
            # Clear databases and caches
            self._cliche_database.clear()
            self._character_voice_patterns.clear()
            self._style_analysis_tools.clear()
            self._vocabulary_analyzer = None
            
            return True
            
        except Exception:
            return False
    
    # Analysis Implementation Methods
    
    async def _load_cliche_database(self) -> None:
        """Load database of common clichés and overused phrases."""
        # Common literary clichés
        common_cliches = {
            "it was a dark and stormy night",
            "all hell broke loose",
            "avoid like the plague",
            "crystal clear",
            "dead as a doornail",
            "easier said than done",
            "few and far between",
            "last but not least",
            "the calm before the storm",
            "time will tell",
            "when all is said and done",
            "you can't judge a book by its cover",
            "at the end of the day",
            "thinking outside the box",
            "it goes without saying",
            "needless to say"
        }
        
        self._cliche_database.update(common_cliches)
    
    async def _initialize_style_analysis_tools(self) -> None:
        """Initialize tools for style analysis."""
        # Placeholder for actual NLP tool initialization
        self._style_analysis_tools = {
            'sentence_complexity_analyzer': True,
            'readability_calculator': True,
            'tone_analyzer': True,
            'flow_analyzer': True
        }
    
    async def _initialize_vocabulary_analyzer(self) -> None:
        """Initialize vocabulary richness analyzer."""
        # Placeholder for vocabulary analysis tools
        self._vocabulary_analyzer = {
            'word_frequency_analyzer': True,
            'lexical_diversity_calculator': True,
            'reading_level_estimator': True
        }
    
    async def _extract_text_content(self, chapter: ChapterVariant) -> str:
        """Extract text content from chapter for analysis."""
        # Combine all text elements
        text_parts = []
        
        if hasattr(chapter, 'content') and chapter.content:
            text_parts.append(chapter.content)
        
        if hasattr(chapter, 'scenes'):
            for scene in chapter.scenes:
                if hasattr(scene, 'narrative_text') and scene.narrative_text:
                    text_parts.append(scene.narrative_text)
                if hasattr(scene, 'dialogue') and scene.dialogue:
                    text_parts.extend(scene.dialogue)
        
        return " ".join(text_parts)
    
    async def _analyze_prose_quality(self, text: str) -> Dict[str, Any]:
        """Analyze prose quality including sentence structure and flow."""
        # Placeholder implementation
        # In real implementation, this would use NLP tools to analyze:
        # - Sentence variety and structure
        # - Flow and rhythm
        # - Word choice precision
        # - Descriptive language effectiveness
        
        sentences = text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Simple scoring based on sentence variety
        score = min(1.0, max(0.0, 0.8 - abs(avg_length - 15) * 0.02))
        
        return {
            'score': score,
            'avg_sentence_length': avg_length,
            'sentence_variety': 0.7,  # Placeholder
            'flow_quality': 0.8       # Placeholder
        }
    
    async def _analyze_originality(self, text: str) -> Dict[str, Any]:
        """Analyze text originality and uniqueness."""
        # Placeholder implementation
        # In real implementation, this would:
        # - Check against known text databases
        # - Analyze phrase uniqueness
        # - Detect formulaic patterns
        # - Assess creative language use
        
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        uniqueness_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Base score on word uniqueness
        score = min(1.0, uniqueness_ratio * 2)
        
        return {
            'score': score,
            'uniqueness_ratio': uniqueness_ratio,
            'formulaic_patterns': [],
            'creative_phrases': []
        }
    
    async def _analyze_character_voices(self, text: str, chapter: ChapterVariant, 
                                      story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze character voice authenticity and consistency."""
        # Placeholder implementation
        character_analyses = []
        
        # Extract characters mentioned in the story state
        characters = getattr(story_state, 'characters', [])
        
        for character in characters[:3]:  # Analyze up to 3 main characters
            char_id = getattr(character, 'character_id', 'unknown')
            
            # Analyze voice consistency for this character
            voice_analysis = VoiceAnalysis(
                character_id=char_id,
                voice_consistency_score=0.8,  # Placeholder
                dialogue_authenticity=0.75,   # Placeholder
                internal_thoughts_consistency=0.85,  # Placeholder
                issues=[]
            )
            character_analyses.append(voice_analysis)
        
        overall_score = sum(va.voice_consistency_score for va in character_analyses) / max(len(character_analyses), 1)
        
        return {
            'overall_score': overall_score,
            'character_analyses': character_analyses
        }
    
    async def _analyze_pacing(self, text: str) -> Dict[str, Any]:
        """Analyze chapter pacing and rhythm."""
        # Placeholder implementation
        # Real implementation would analyze:
        # - Action vs. description balance
        # - Dialogue density
        # - Scene transitions
        # - Tension building/release
        
        paragraphs = text.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        
        # Simple pacing score based on paragraph variety
        score = 0.75  # Placeholder
        
        return {
            'score': score,
            'avg_paragraph_length': avg_paragraph_length,
            'action_description_balance': 0.6,
            'dialogue_density': 0.4
        }
    
    async def _analyze_vocabulary_richness(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary richness and reading level."""
        words = text.split()
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        unique_ratio = unique_words / max(word_count, 1)
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        # Simple reading level estimate (Flesch-like)
        reading_level = min(100, max(0, 206.835 - 1.015 * avg_sentence_length - 84.6 * (unique_ratio * 2)))
        
        score = min(1.0, unique_ratio * 2)
        
        return {
            'score': score,
            'word_count': word_count,
            'unique_ratio': unique_ratio,
            'avg_sentence_length': avg_sentence_length,
            'reading_level': reading_level / 100.0
        }
    
    async def _detect_cliches(self, text: str) -> Dict[str, Any]:
        """Detect clichéd phrases and expressions."""
        text_lower = text.lower()
        detected_cliches = []
        
        for cliche in self._cliche_database:
            if cliche in text_lower:
                detected_cliches.append(cliche)
        
        return {
            'cliches': detected_cliches,
            'cliche_count': len(detected_cliches)
        }
    
    async def _identify_style_issues(self, text: str) -> Dict[str, Any]:
        """Identify specific style issues in the text."""
        issues = []
        
        # Check for common style issues
        if text.count('very ') > 5:
            issues.append(StyleIssue(
                issue_type="overuse",
                description="Overuse of the word 'very' - consider stronger adjectives",
                severity="low",
                location="throughout text",
                suggestion="Replace 'very + adjective' with stronger alternatives"
            ))
        
        if text.count(' and ') > len(text.split()) * 0.05:
            issues.append(StyleIssue(
                issue_type="repetitive_structure",
                description="Frequent use of 'and' may indicate repetitive sentence structure",
                severity="medium",
                location="sentence structure",
                suggestion="Vary sentence structure and use different conjunctions"
            ))
        
        return {'issues': issues}
    
    async def _update_character_voice_patterns(self, story_state: DynamicStoryState) -> None:
        """Update character voice patterns from story state."""
        # Extract character information from story state
        characters = getattr(story_state, 'characters', [])
        
        for character in characters:
            char_id = getattr(character, 'character_id', 'unknown')
            if char_id not in self._character_voice_patterns:
                self._character_voice_patterns[char_id] = {
                    'dialogue_patterns': [],
                    'vocabulary_preferences': set(),
                    'speech_mannerisms': []
                }
    
    def _calculate_overall_score(self, prose_score: float, originality_score: float,
                               voice_score: float, pacing_score: float, vocab_score: float) -> float:
        """Calculate weighted overall literary quality score."""
        weights = {
            'prose': 0.3,
            'originality': 0.25,
            'voice': 0.25,
            'pacing': 0.1,
            'vocabulary': 0.1
        }
        
        return (prose_score * weights['prose'] +
                originality_score * weights['originality'] +
                voice_score * weights['voice'] +
                pacing_score * weights['pacing'] +
                vocab_score * weights['vocabulary'])
    
    def _identify_strengths(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """Identify literary strengths from analysis results."""
        strengths = []
        
        prose_result = analysis_results[0]
        if prose_result['score'] > 0.8:
            strengths.append("Excellent prose quality with varied sentence structure")
        
        originality_result = analysis_results[1]
        if originality_result['score'] > 0.7:
            strengths.append("Strong originality with unique phrasing")
        
        return strengths
    
    def _generate_improvement_suggestions(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        prose_result = analysis_results[0]
        if prose_result['score'] < 0.6:
            suggestions.append("Vary sentence structure and improve flow between sentences")
        
        vocab_result = analysis_results[4]
        if vocab_result['unique_ratio'] < 0.4:
            suggestions.append("Expand vocabulary variety to avoid repetitive word choice")
        
        return suggestions
    
    # Test methods for health checks
    
    async def _test_style_analysis(self) -> bool:
        """Test style analysis functionality."""
        try:
            test_text = "This is a test sentence. This is another test sentence."
            result = await self._analyze_prose_quality(test_text)
            return 'score' in result
        except Exception:
            return False
    
    async def _test_vocabulary_analyzer(self) -> bool:
        """Test vocabulary analysis functionality."""
        try:
            test_text = "The quick brown fox jumps over the lazy dog."
            result = await self._analyze_vocabulary_richness(test_text)
            return 'score' in result
        except Exception:
            return False
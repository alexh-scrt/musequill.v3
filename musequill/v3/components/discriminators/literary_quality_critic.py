"""
Literary Quality Critic Component

Implements prose quality evaluation, language freshness analysis, character voice
validation, and pacing assessment for the adversarial system discriminator layer.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.literary_quality_assessment import (
    LiteraryQualityAssessment, ClicheFlag, VoiceConsistencyIssue,
    PacingAnalysis, ProseAnalysis, LanguageFreshnessAnalysis,
    ClicheSeverity, VoiceInconsistencyType, PacingIssueType, ProseQualityIssue
)


class LiteraryQualityCriticConfig(BaseModel):
    """Configuration for Literary Quality Critic component."""
    
    cliche_detection_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging potential clichés"
    )
    
    voice_consistency_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum voice consistency score before flagging issues"
    )
    
    enable_advanced_prose_analysis: bool = Field(
        default=True,
        description="Whether to perform detailed prose structure analysis"
    )
    
    enable_readability_scoring: bool = Field(
        default=True,
        description="Whether to calculate readability metrics"
    )
    
    target_reading_level: str = Field(
        default="general_adult",
        description="Target reading level for assessment"
    )
    
    max_analysis_time_seconds: int = Field(
        default=45,
        ge=10,
        le=300,
        description="Maximum time for complete literary analysis"
    )


class LiteraryQualityCriticInput(BaseModel):
    """Input data for Literary Quality Critic."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate for literary quality"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for character voice validation"
    )
    
    genre_context: Optional[str] = Field(
        default=None,
        description="Genre context for genre-specific quality standards"
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience for appropriate style assessment"
    )


class LiteraryQualityCritic(BaseComponent[LiteraryQualityCriticInput, LiteraryQualityAssessment, LiteraryQualityCriticConfig]):
    """
    Literary Quality Critic component for prose and style evaluation.
    
    Analyzes language freshness, character voice consistency, pacing effectiveness,
    and technical prose quality with genre-aware assessment criteria.
    """
    
    def __init__(self, config: ComponentConfiguration[LiteraryQualityCriticConfig]):
        super().__init__(config)
        self._cliche_database: Set[str] = set()
        self._character_voice_patterns: Dict[str, Dict[str, Any]] = {}
        self._prose_quality_rules: List[Dict[str, Any]] = []
        self._readability_calculator = None
    
    async def initialize(self) -> bool:
        """Initialize literary analysis tools and databases."""
        try:
            # Load cliché database
            await self._load_cliche_database()
            
            # Initialize prose quality rules
            await self._initialize_prose_rules()
            
            # Initialize readability calculator
            if self.config.specific_config.enable_readability_scoring:
                await self._initialize_readability_tools()
            
            # Initialize NLP components for advanced analysis
            if self.config.specific_config.enable_advanced_prose_analysis:
                await self._initialize_advanced_analysis_tools()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Literary critic initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: LiteraryQualityCriticInput) -> LiteraryQualityAssessment:
        """
        Perform comprehensive literary quality analysis.
        
        Args:
            input_data: Chapter variant and context for analysis
            
        Returns:
            Detailed literary quality assessment
        """
        start_time = datetime.now()
        
        try:
            # Update character voice patterns from story state
            await self._update_character_voice_patterns(input_data.story_state)
            
            # Perform parallel analysis across quality dimensions
            analysis_tasks = [
                self._analyze_language_freshness(input_data.chapter_variant),
                self._analyze_character_voice_consistency(input_data.chapter_variant, input_data.story_state),
                self._analyze_pacing_quality(input_data.chapter_variant),
                self._analyze_prose_technical_quality(input_data.chapter_variant)
            ]
            
            # Execute analysis with timeout
            timeout_seconds = self.config.specific_config.max_analysis_time_seconds
            results = await asyncio.wait_for(
                asyncio.gather(*analysis_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            
            # Handle exceptions in analysis tasks
            freshness_result, voice_result, pacing_result, prose_result = results
            for result in results:
                if isinstance(result, Exception):
                    raise ComponentError(f"Literary analysis subtask failed: {str(result)}")
            
            # Compile comprehensive assessment
            assessment = await self._compile_literary_assessment(
                input_data.chapter_variant,
                freshness_result,
                voice_result,
                pacing_result,
                prose_result,
                input_data.genre_context
            )
            
            return assessment
            
        except asyncio.TimeoutError:
            raise ComponentError(f"Literary analysis exceeded {timeout_seconds}s timeout")
        except Exception as e:
            raise ComponentError(f"Literary quality analysis failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on literary analysis systems."""
        try:
            # Test cliché detection
            test_text = "It was a dark and stormy night, and time was running out."
            cliches = await self._detect_cliches(test_text)
            if len(cliches) == 0:  # Should detect at least one cliché
                return False
            
            # Test prose analysis
            prose_metrics = await self._calculate_prose_metrics(test_text)
            if not prose_metrics:
                return False
            
            # Check component error rates
            if self.state.metrics.failure_rate > 0.15:
                return False
                
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup literary analysis resources."""
        try:
            self._character_voice_patterns.clear()
            self._cliche_database.clear()
            self._prose_quality_rules.clear()
            
            # Cleanup NLP models if loaded
            self._readability_calculator = None
            
            return True
            
        except Exception:
            return False
    
    async def _load_cliche_database(self) -> None:
        """Load database of common clichés and overused phrases."""
        # Common clichés database (would be loaded from external source)
        common_cliches = {
            "it was a dark and stormy night",
            "time was running out",
            "her heart skipped a beat",
            "his eyes sparkled with",
            "she felt a shiver down her spine",
            "the weight of the world",
            "a bitter taste in her mouth",
            "his blood ran cold",
            "time stood still",
            "her mind raced",
            "dawn was breaking",
            "silence was deafening",
            "crystal clear",
            "diamond in the rough",
            "needle in a haystack"
        }
        
        self._cliche_database.update(common_cliches)
        
        # Would also load genre-specific and contextual clichés
    
    async def _initialize_prose_rules(self) -> None:
        """Initialize prose quality evaluation rules."""
        self._prose_quality_rules = [
            {
                'name': 'passive_voice_detection',
                'pattern': r'\b(was|were|is|are|am|be|been|being)\s+\w+ed\b',
                'severity': 'moderate',
                'issue_type': ProseQualityIssue.PASSIVE_VOICE_OVERUSE
            },
            {
                'name': 'weak_verbs',
                'pattern': r'\b(went|got|had|was|were|did|make|take|give|put)\b',
                'severity': 'minor',
                'issue_type': ProseQualityIssue.WEAK_VERBS
            },
            {
                'name': 'adverb_overuse',
                'pattern': r'\w+ly\b',
                'severity': 'minor', 
                'issue_type': ProseQualityIssue.ADVERB_OVERUSE
            },
            {
                'name': 'repetitive_sentence_starts',
                'pattern': r'^(He|She|I|They|The)\s',
                'severity': 'moderate',
                'issue_type': ProseQualityIssue.REPETITIVE_STRUCTURE
            }
        ]
    
    async def _initialize_readability_tools(self) -> None:
        """Initialize readability calculation tools."""
        # Placeholder for readability calculator initialization
        # Would initialize Flesch-Kincaid, SMOG, or other readability metrics
        self._readability_calculator = "initialized"
    
    async def _initialize_advanced_analysis_tools(self) -> None:
        """Initialize advanced NLP analysis tools."""
        # Placeholder for advanced NLP tool initialization
        # Would initialize sentence parsing, semantic analysis, etc.
        pass
    
    async def _update_character_voice_patterns(self, story_state: DynamicStoryState) -> None:
        """Update character voice patterns from story state."""
        for char_id, character in story_state.character_arcs.items():
            self._character_voice_patterns[char_id] = {
                'vocabulary_level': character.voice_characteristics.vocabulary_level,
                'speech_patterns': character.voice_characteristics.speech_patterns,
                'formality_level': character.voice_characteristics.formality_level,
                'dialogue_quirks': character.voice_characteristics.dialogue_quirks,
                'personality_traits': character.personality_traits
            }
    
    async def _analyze_language_freshness(self, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze originality and freshness of language use."""
        text = chapter.chapter_text
        
        # Detect clichés
        cliche_flags = await self._detect_cliches(text)
        
        # Calculate freshness metrics
        metaphor_originality = await self._assess_metaphor_originality(text)
        descriptive_creativity = await self._assess_descriptive_creativity(text)
        phrase_uniqueness = await self._assess_phrase_uniqueness(text)
        
        # Calculate cliché density
        word_count = len(text.split())
        cliche_density = len(cliche_flags) / max(1, word_count / 100)  # Per 100 words
        cliche_density = min(1.0, cliche_density / 5)  # Normalize to 0-1
        
        # Calculate vocabulary richness
        vocabulary_richness = await self._calculate_vocabulary_richness(text)
        
        # Find innovation examples
        innovation_examples = await self._find_innovative_language(text)
        
        freshness_analysis = LanguageFreshnessAnalysis(
            metaphor_originality=metaphor_originality,
            descriptive_creativity=descriptive_creativity,
            phrase_uniqueness=phrase_uniqueness,
            cliche_density=cliche_density,
            vocabulary_richness=vocabulary_richness,
            innovation_examples=innovation_examples
        )
        
        # Calculate overall freshness score
        freshness_score = (
            metaphor_originality * 0.25 +
            descriptive_creativity * 0.25 +
            phrase_uniqueness * 0.2 +
            (1.0 - cliche_density) * 0.2 +
            vocabulary_richness * 0.1
        )
        
        return {
            'language_freshness_score': freshness_score,
            'freshness_analysis': freshness_analysis,
            'cliche_flags': cliche_flags
        }
    
    async def _analyze_character_voice_consistency(self, chapter: ChapterVariant, 
                                                  story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze consistency of character voices in dialogue and thoughts."""
        voice_issues = []
        voice_scores = []
        
        # Extract dialogue and character thoughts from text
        character_speech = await self._extract_character_speech(chapter.chapter_text, chapter.characters_featured)
        
        # Analyze each character's voice consistency
        for char_id in chapter.characters_featured:
            if char_id in self._character_voice_patterns and char_id in character_speech:
                expected_patterns = self._character_voice_patterns[char_id]
                actual_speech = character_speech[char_id]
                
                # Check voice consistency across different dimensions
                voice_analysis = await self._analyze_individual_voice_consistency(
                    char_id, expected_patterns, actual_speech
                )
                
                voice_scores.append(voice_analysis['consistency_score'])
                voice_issues.extend(voice_analysis['issues'])
        
        # Calculate overall voice consistency score
        if voice_scores:
            overall_voice_score = sum(voice_scores) / len(voice_scores)
        else:
            overall_voice_score = 1.0  # No characters to evaluate
        
        return {
            'character_voice_score': overall_voice_score,
            'voice_consistency_issues': voice_issues
        }
    
    async def _analyze_pacing_quality(self, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze chapter pacing and rhythm quality."""
        text = chapter.chapter_text
        
        # Analyze sentence structure variety
        sentence_rhythm_score = await self._analyze_sentence_rhythm(text)
        
        # Analyze paragraph flow
        paragraph_flow_score = await self._analyze_paragraph_flow(text)
        
        # Analyze scene transitions (if multiple scenes)
        scene_transition_quality = await self._analyze_scene_transitions(chapter.scene_structure)
        
        # Analyze information density
        information_density = await self._calculate_information_density(text)
        
        # Identify pacing issues
        pacing_issues = await self._identify_pacing_issues(text, chapter)
        
        # Identify pacing strengths
        pacing_strengths = await self._identify_pacing_strengths(text, chapter)
        
        # Determine overall pace
        overall_pace = await self._classify_overall_pace(text, pacing_issues)
        
        pacing_analysis = PacingAnalysis(
            overall_pace=overall_pace,
            sentence_rhythm_score=sentence_rhythm_score,
            paragraph_flow_score=paragraph_flow_score,
            scene_transition_quality=scene_transition_quality,
            information_density=information_density,
            pacing_issues=pacing_issues,
            pacing_strengths=pacing_strengths
        )
        
        # Calculate overall pacing score
        pacing_score = (
            sentence_rhythm_score * 0.3 +
            paragraph_flow_score * 0.3 +
            scene_transition_quality * 0.2 +
            information_density * 0.2
        )
        
        return {
            'pacing_score': pacing_score,
            'pacing_analysis': pacing_analysis
        }
    
    async def _analyze_prose_technical_quality(self, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze technical aspects of prose quality."""
        text = chapter.chapter_text
        
        # Analyze sentence variety
        sentence_variety_score = await self._calculate_sentence_variety(text)
        
        # Assess word choice sophistication
        word_choice_score = await self._assess_word_choice_sophistication(text)
        
        # Calculate show vs tell ratio
        show_tell_ratio = await self._calculate_show_tell_ratio(text)
        
        # Assess sensory detail richness
        sensory_detail_score = await self._assess_sensory_details(text)
        
        # Evaluate dialogue quality
        dialogue_quality_score = await self._evaluate_dialogue_quality(text)
        
        # Identify prose issues
        prose_issues = await self._identify_prose_issues(text)
        
        # Identify stylistic strengths
        stylistic_strengths = await self._identify_stylistic_strengths(text)
        
        prose_analysis = ProseAnalysis(
            sentence_variety_score=sentence_variety_score,
            word_choice_sophistication=word_choice_score,
            show_vs_tell_ratio=show_tell_ratio,
            sensory_detail_richness=sensory_detail_score,
            dialogue_quality=dialogue_quality_score,
            prose_issues=prose_issues,
            stylistic_strengths=stylistic_strengths
        )
        
        # Calculate overall prose quality score
        prose_score = (
            sentence_variety_score * 0.2 +
            word_choice_score * 0.2 +
            show_tell_ratio * 0.2 +
            sensory_detail_score * 0.2 +
            dialogue_quality_score * 0.2
        )
        
        return {
            'prose_quality_score': prose_score,
            'prose_analysis': prose_analysis
        }
    
    async def _compile_literary_assessment(self, chapter: ChapterVariant,
                                          freshness_result: Dict, voice_result: Dict,
                                          pacing_result: Dict, prose_result: Dict,
                                          genre_context: Optional[str]) -> LiteraryQualityAssessment:
        """Compile comprehensive literary quality assessment."""
        
        # Generate improvement suggestions
        suggestions = []
        
        if freshness_result['language_freshness_score'] < 0.6:
            suggestions.append("Reduce clichéd expressions and develop more original metaphors")
        
        if voice_result['character_voice_score'] < 0.7:
            suggestions.append("Improve character voice consistency in dialogue and internal thoughts")
        
        if pacing_result['pacing_score'] < 0.6:
            suggestions.append("Address pacing issues to improve reader engagement")
        
        if prose_result['prose_quality_score'] < 0.7:
            suggestions.append("Strengthen technical prose quality and sentence variety")
        
        # Identify notable strengths
        notable_strengths = []
        
        if freshness_result['language_freshness_score'] >= 0.8:
            notable_strengths.append("Excellent language originality and creative expression")
        
        if voice_result['character_voice_score'] >= 0.8:
            notable_strengths.append("Strong character voice differentiation and consistency")
        
        if pacing_result['pacing_score'] >= 0.8:
            notable_strengths.append("Effective pacing and rhythm management")
        
        # Add prose strengths
        notable_strengths.extend(prose_result['prose_analysis'].stylistic_strengths)
        
        return LiteraryQualityAssessment(
            chapter_number=chapter.chapter_number,
            language_freshness_score=freshness_result['language_freshness_score'],
            character_voice_score=voice_result['character_voice_score'],
            pacing_score=pacing_result['pacing_score'],
            prose_quality_score=prose_result['prose_quality_score'],
            cliche_flags=freshness_result['cliche_flags'],
            voice_consistency_issues=voice_result['voice_consistency_issues'],
            pacing_analysis=pacing_result['pacing_analysis'],
            prose_analysis=prose_result['prose_analysis'],
            language_freshness_analysis=freshness_result['freshness_analysis'],
            improvement_suggestions=suggestions,
            notable_strengths=notable_strengths
        )
    
    # Placeholder methods for actual analysis implementation
    async def _detect_cliches(self, text: str) -> List[ClicheFlag]:
        """Detect clichéd expressions in text."""
        cliches = []
        text_lower = text.lower()
        
        for cliche in self._cliche_database:
            if cliche in text_lower:
                count = text_lower.count(cliche)
                severity = ClicheSeverity.EXCESSIVE if count > 2 else ClicheSeverity.MODERATE
                
                cliche_flag = ClicheFlag(
                    cliche_id=f"cliche_{len(cliches)}",
                    cliche_text=cliche,
                    cliche_type="expression",
                    severity=severity,
                    context=f"Found {count} times in chapter",
                    replacement_suggestions=[f"Consider rephrasing '{cliche}' with original language"],
                    frequency_in_chapter=count
                )
                cliches.append(cliche_flag)
        
        return cliches
    
    async def _assess_metaphor_originality(self, text: str) -> float:
        """Assess originality of metaphors and comparisons."""
        # Placeholder - would use NLP to identify and evaluate metaphors
        return 0.75
    
    async def _assess_descriptive_creativity(self, text: str) -> float:
        """Assess creativity in descriptive language."""
        # Placeholder - would analyze descriptive passages for creativity
        return 0.7
    
    async def _assess_phrase_uniqueness(self, text: str) -> float:
        """Assess uniqueness of phrases and expressions."""
        # Placeholder - would check against common phrase databases
        return 0.8
    
    async def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary diversity and richness."""
        words = text.lower().split()
        unique_words = set(words)
        if len(words) == 0:
            return 0.0
        return min(1.0, len(unique_words) / len(words) * 2)  # Simple diversity metric
    
    async def _find_innovative_language(self, text: str) -> List[str]:
        """Find examples of particularly creative language."""
        # Placeholder - would identify creative expressions
        return ["Example of creative language use"]
    
    async def _extract_character_speech(self, text: str, character_ids: List[str]) -> Dict[str, List[str]]:
        """Extract dialogue and thoughts for each character."""
        # Placeholder - would use NLP to extract character-specific speech
        return {char_id: [f"Sample dialogue for {char_id}"] for char_id in character_ids}
    
    async def _analyze_individual_voice_consistency(self, char_id: str, expected_patterns: Dict, 
                                                   actual_speech: List[str]) -> Dict[str, Any]:
        """Analyze voice consistency for individual character."""
        # Placeholder - would analyze speech patterns, vocabulary, formality
        return {
            'consistency_score': 0.8,
            'issues': []
        }
    
    async def _analyze_sentence_rhythm(self, text: str) -> float:
        """Analyze variety and flow of sentence structures."""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
        
        # Simple analysis of sentence length variety
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        # Calculate coefficient of variation as rhythm measure
        mean_length = sum(lengths) / len(lengths)
        if mean_length == 0:
            return 0.0
        
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_length
        
        # Normalize to 0-1 where moderate variation (cv ~0.3-0.7) is ideal
        ideal_cv = 0.5
        rhythm_score = 1.0 - abs(cv - ideal_cv)
        return max(0.0, min(1.0, rhythm_score))
    
    async def _analyze_paragraph_flow(self, text: str) -> float:
        """Analyze smoothness of paragraph transitions."""
        # Placeholder - would analyze paragraph connections and transitions
        return 0.75
    
    async def _analyze_scene_transitions(self, scene_structure: List) -> float:
        """Analyze quality of scene transitions."""
        # Placeholder - would analyze scene break effectiveness
        return 0.8
    
    async def _calculate_information_density(self, text: str) -> float:
        """Calculate appropriate balance of information vs action."""
        # Placeholder - would measure information load per paragraph
        return 0.7
    
    async def _identify_pacing_issues(self, text: str, chapter) -> List[PacingIssueType]:
        """Identify specific pacing problems."""
        issues = []
        
        # Simple heuristics for common pacing issues
        if len(text.split()) < 1000:  # Very short chapter
            issues.append(PacingIssueType.TOO_FAST)
        elif len(text.split()) > 5000:  # Very long chapter
            issues.append(PacingIssueType.TOO_SLOW)
        
        return issues
    
    async def _identify_pacing_strengths(self, text: str, chapter) -> List[str]:
        """Identify effective pacing elements."""
        # Placeholder - would identify well-executed pacing techniques
        return ["Good chapter length balance"]
    
    async def _classify_overall_pace(self, text: str, issues: List[PacingIssueType]) -> str:
        """Classify overall pacing of chapter."""
        if PacingIssueType.TOO_SLOW in issues:
            return "slow"
        elif PacingIssueType.TOO_FAST in issues:
            return "fast"
        else:
            return "moderate"
    
    async def _calculate_sentence_variety(self, text: str) -> float:
        """Calculate variety in sentence length and structure."""
        # Already implemented in _analyze_sentence_rhythm
        return await self._analyze_sentence_rhythm(text)
    
    async def _assess_word_choice_sophistication(self, text: str) -> float:
        """Assess sophistication of vocabulary choices."""
        # Placeholder - would analyze vocabulary complexity
        return 0.7
    
    async def _calculate_show_tell_ratio(self, text: str) -> float:
        """Calculate balance of showing vs telling."""
        # Placeholder - would identify narrative vs descriptive passages
        return 0.75
    
    async def _assess_sensory_details(self, text: str) -> float:
        """Assess richness of sensory descriptions."""
        # Simple check for sensory words
        sensory_words = ['see', 'saw', 'hear', 'heard', 'feel', 'felt', 'taste', 'smell', 'touch']
        word_count = len(text.split())
        sensory_count = sum(1 for word in sensory_words if word in text.lower())
        
        if word_count == 0:
            return 0.0
        
        sensory_density = sensory_count / word_count * 100  # Per 100 words
        return min(1.0, sensory_density)
    
    async def _evaluate_dialogue_quality(self, text: str) -> float:
        """Evaluate naturalness and effectiveness of dialogue."""
        # Placeholder - would analyze dialogue patterns and naturalness
        return 0.8
    
    async def _identify_prose_issues(self, text: str) -> List[ProseQualityIssue]:
        """Identify technical prose quality issues."""
        issues = []
        
        # Check for passive voice overuse
        passive_matches = re.findall(r'\b(was|were|is|are|am|be|been|being)\s+\w+ed\b', text)
        if len(passive_matches) > len(text.split()) * 0.1:  # More than 10% passive
            issues.append(ProseQualityIssue.PASSIVE_VOICE_OVERUSE)
        
        # Check for adverb overuse
        adverb_matches = re.findall(r'\w+ly\b', text)
        if len(adverb_matches) > len(text.split()) * 0.05:  # More than 5% adverbs
            issues.append(ProseQualityIssue.ADVERB_OVERUSE)
        
        return issues
    
    async def _identify_stylistic_strengths(self, text: str) -> List[str]:
        """Identify notable prose strengths."""
        strengths = []
        
        # Check for good sentence variety (already calculated)
        sentence_rhythm = await self._analyze_sentence_rhythm(text)
        if sentence_rhythm >= 0.8:
            strengths.append("Excellent sentence rhythm and variety")
        
        return strengths
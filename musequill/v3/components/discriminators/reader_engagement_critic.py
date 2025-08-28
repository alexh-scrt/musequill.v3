"""
Reader Engagement Critic Component

Implements reader engagement evaluation, commercial viability assessment,
and reader satisfaction prediction for the adversarial system discriminator layer.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.market_intelligence import MarketIntelligence


class EngagementFactor(str, Enum):
    """Types of engagement factors."""
    EMOTIONAL_JOURNEY = "emotional_journey"
    CURIOSITY_HOOKS = "curiosity_hooks"
    CLIFFHANGER_EFFECTIVENESS = "cliffhanger_effectiveness"
    CHARACTER_RELATABILITY = "character_relatability"
    CONFLICT_TENSION = "conflict_tension"
    PACING_MOMENTUM = "pacing_momentum"
    DIALOGUE_APPEAL = "dialogue_appeal"
    SCENE_IMMERSION = "scene_immersion"


class ReaderEngagementCriticConfig(BaseModel):
    """Configuration for Reader Engagement Critic component."""
    
    engagement_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable reader engagement score"
    )
    
    commercial_viability_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for commercial viability in overall score"
    )
    
    emotional_impact_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for emotional impact in overall score"
    )
    
    curiosity_factor_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for curiosity/hooks in overall score"
    )
    
    max_analysis_time_seconds: int = Field(
        default=40,
        ge=5,
        le=300,
        description="Maximum time to spend analyzing engagement"
    )
    
    enable_market_alignment: bool = Field(
        default=True,
        description="Whether to consider market trends in assessment"
    )
    
    target_demographics: List[str] = Field(
        default_factory=lambda: ["young_adult", "adult", "general_fiction"],
        description="Target reader demographics for assessment"
    )
    
    minimum_emotional_variety: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum variety of emotions that should be evoked"
    )


class EmotionalBeat(BaseModel):
    """Represents an emotional moment in the chapter."""
    
    emotion_type: str = Field(description="Type of emotion evoked")
    intensity: float = Field(ge=0.0, le=1.0, description="Emotional intensity")
    location: str = Field(description="Where in the chapter this occurs")
    effectiveness: float = Field(ge=0.0, le=1.0, description="How effective this beat is")


class CuriosityHook(BaseModel):
    """Represents a curiosity-inducing element."""
    
    hook_type: str = Field(description="Type of curiosity hook")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of the hook")
    resolution_promise: bool = Field(description="Whether it promises future resolution")
    description: str = Field(description="Description of the hook")


class EngagementAnalysis(BaseModel):
    """Analysis of a specific engagement factor."""
    
    factor: EngagementFactor = Field(description="The engagement factor analyzed")
    score: float = Field(ge=0.0, le=1.0, description="Score for this factor")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class ReaderEngagementAssessment(BaseModel):
    """Assessment of reader engagement for a chapter variant."""
    
    chapter_number: int = Field(description="Chapter number analyzed")
    overall_engagement_score: float = Field(ge=0.0, le=1.0, description="Overall engagement score")
    
    # Component scores
    emotional_impact_score: float = Field(ge=0.0, le=1.0, description="Emotional impact score")
    curiosity_factor_score: float = Field(ge=0.0, le=1.0, description="Curiosity and hooks score")
    commercial_viability_score: float = Field(ge=0.0, le=1.0, description="Commercial appeal score")
    character_connection_score: float = Field(ge=0.0, le=1.0, description="Character connection score")
    pacing_momentum_score: float = Field(ge=0.0, le=1.0, description="Pacing and momentum score")
    
    # Detailed analysis
    emotional_beats: List[EmotionalBeat] = Field(default_factory=list, description="Emotional beats identified")
    curiosity_hooks: List[CuriosityHook] = Field(default_factory=list, description="Curiosity hooks found")
    engagement_factors: List[EngagementAnalysis] = Field(default_factory=list, description="Detailed factor analysis")
    
    # Reader experience prediction
    predicted_reader_retention: float = Field(ge=0.0, le=1.0, description="Predicted reader retention")
    cliffhanger_effectiveness: float = Field(ge=0.0, le=1.0, description="Chapter ending effectiveness")
    page_turner_quality: float = Field(ge=0.0, le=1.0, description="Page-turning quality")
    
    # Market alignment
    market_alignment_score: Optional[float] = Field(default=None, description="Market trend alignment")
    demographic_appeal: Dict[str, float] = Field(default_factory=dict, description="Appeal to demographics")
    
    # Recommendations
    engagement_strengths: List[str] = Field(default_factory=list, description="Engagement strengths")
    improvement_priorities: List[str] = Field(default_factory=list, description="Priority improvements")
    reader_experience_prediction: str = Field(description="Predicted reader experience")


class ReaderEngagementCriticInput(BaseModel):
    """Input data for Reader Engagement Critic."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for engagement context"
    )
    
    market_intelligence: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence for commercial viability assessment"
    )
    
    previous_assessments: List[ReaderEngagementAssessment] = Field(
        default_factory=list,
        description="Previous engagement assessments for trend analysis"
    )


class ReaderEngagementCritic(BaseComponent[ReaderEngagementCriticInput, ReaderEngagementAssessment, ReaderEngagementCriticConfig]):
    """
    Reader Engagement Critic component for commercial viability and engagement evaluation.
    
    Analyzes chapter variants for reader engagement factors, emotional journey effectiveness,
    commercial appeal, and reader satisfaction prediction.
    """
    
    def __init__(self, config: ComponentConfiguration[ReaderEngagementCriticConfig]):
        super().__init__(config)
        self._engagement_patterns: Dict[str, Dict[str, Any]] = {}
        self._market_trends_cache: Dict[str, Any] = {}
        self._emotional_response_models: Dict[str, Any] = {}
        self._engagement_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize the reader engagement analysis systems."""
        try:
            # Initialize engagement pattern recognition
            await self._initialize_engagement_patterns()
            
            # Initialize emotional response models
            await self._initialize_emotional_models()
            
            # Initialize market trend analysis
            await self._initialize_market_analysis()
            
            # Initialize demographic models
            await self._initialize_demographic_models()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Reader engagement critic initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: ReaderEngagementCriticInput) -> ReaderEngagementAssessment:
        """
        Analyze chapter variant for reader engagement and commercial viability.
        
        Args:
            input_data: Chapter variant and context for engagement analysis
            
        Returns:
            Comprehensive reader engagement assessment
        """
        start_time = datetime.now()
        chapter = input_data.chapter_variant
        story_state = input_data.story_state
        market_intelligence = input_data.market_intelligence
        
        try:
            # Extract content for analysis
            text_content = await self._extract_chapter_content(chapter)
            
            # Perform parallel engagement analysis
            analysis_tasks = [
                self._analyze_emotional_journey(text_content, chapter, story_state),
                self._analyze_curiosity_hooks(text_content, chapter),
                self._analyze_character_connection(text_content, chapter, story_state),
                self._analyze_pacing_momentum(text_content, chapter),
                self._analyze_cliffhanger_effectiveness(text_content, chapter),
                self._analyze_commercial_viability(text_content, chapter, market_intelligence)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            emotional_result = results[0]
            curiosity_result = results[1]
            character_result = results[2]
            pacing_result = results[3]
            cliffhanger_result = results[4]
            commercial_result = results[5]
            
            # Calculate component scores
            emotional_impact_score = emotional_result['score']
            curiosity_factor_score = curiosity_result['score']
            character_connection_score = character_result['score']
            pacing_momentum_score = pacing_result['score']
            commercial_viability_score = commercial_result['score']
            
            # Calculate overall engagement score
            overall_score = self._calculate_overall_engagement_score(
                emotional_impact_score,
                curiosity_factor_score,
                character_connection_score,
                pacing_momentum_score,
                commercial_viability_score
            )
            
            # Predict reader experience
            reader_retention = await self._predict_reader_retention(
                overall_score, emotional_result, curiosity_result
            )
            
            page_turner_quality = await self._assess_page_turner_quality(
                pacing_result, curiosity_result, cliffhanger_result
            )
            
            # Generate engagement factors analysis
            engagement_factors = await self._generate_engagement_factors_analysis(
                emotional_result, curiosity_result, character_result, 
                pacing_result, commercial_result
            )
            
            # Market alignment analysis
            market_alignment_score = None
            demographic_appeal = {}
            if market_intelligence and self.config.specific_config.enable_market_alignment:
                market_alignment_score = commercial_result['market_alignment']
                demographic_appeal = await self._analyze_demographic_appeal(
                    text_content, chapter, market_intelligence
                )
            
            # Generate recommendations
            strengths = self._identify_engagement_strengths(results)
            improvement_priorities = self._generate_improvement_priorities(results)
            reader_prediction = self._predict_reader_experience(overall_score, results)
            
            # Compile comprehensive assessment
            assessment = ReaderEngagementAssessment(
                chapter_number=chapter.chapter_number,
                overall_engagement_score=overall_score,
                emotional_impact_score=emotional_impact_score,
                curiosity_factor_score=curiosity_factor_score,
                commercial_viability_score=commercial_viability_score,
                character_connection_score=character_connection_score,
                pacing_momentum_score=pacing_momentum_score,
                emotional_beats=emotional_result['beats'],
                curiosity_hooks=curiosity_result['hooks'],
                engagement_factors=engagement_factors,
                predicted_reader_retention=reader_retention,
                cliffhanger_effectiveness=cliffhanger_result['score'],
                page_turner_quality=page_turner_quality,
                market_alignment_score=market_alignment_score,
                demographic_appeal=demographic_appeal,
                engagement_strengths=strengths,
                improvement_priorities=improvement_priorities,
                reader_experience_prediction=reader_prediction
            )
            
            # Update engagement history for learning
            await self._update_engagement_history(assessment, results)
            
            return assessment
            
        except Exception as e:
            raise ComponentError(f"Reader engagement analysis failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on reader engagement analysis systems."""
        try:
            # Check engagement pattern models are loaded
            if not self._engagement_patterns:
                return False
            
            # Test emotional response models
            if not self._emotional_response_models:
                return False
            
            # Test analysis functionality
            test_result = await self._test_engagement_analysis()
            if not test_result:
                return False
            
            # Check component performance metrics
            if self.state.metrics.failure_rate > 0.15:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup reader engagement analysis resources."""
        try:
            # Clear caches and models
            self._engagement_patterns.clear()
            self._market_trends_cache.clear()
            self._emotional_response_models.clear()
            
            # Preserve recent history for analysis but limit size
            if len(self._engagement_history) > 50:
                self._engagement_history = self._engagement_history[-25:]
            
            return True
            
        except Exception:
            return False
    
    # Analysis Implementation Methods
    
    async def _initialize_engagement_patterns(self) -> None:
        """Initialize engagement pattern recognition models."""
        # Placeholder for engagement pattern models
        self._engagement_patterns = {
            'emotional_peaks': {'model': 'loaded'},
            'tension_curves': {'model': 'loaded'},
            'curiosity_triggers': {'model': 'loaded'},
            'cliffhanger_patterns': {'model': 'loaded'}
        }
    
    async def _initialize_emotional_models(self) -> None:
        """Initialize emotional response prediction models."""
        self._emotional_response_models = {
            'emotion_classifier': {'model': 'loaded'},
            'intensity_predictor': {'model': 'loaded'},
            'emotional_arc_analyzer': {'model': 'loaded'}
        }
    
    async def _initialize_market_analysis(self) -> None:
        """Initialize market trend analysis tools."""
        self._market_trends_cache = {
            'current_trends': {},
            'reader_preferences': {},
            'commercial_patterns': {}
        }
    
    async def _initialize_demographic_models(self) -> None:
        """Initialize demographic appeal models."""
        # Placeholder for demographic analysis models
        pass
    
    async def _extract_chapter_content(self, chapter: ChapterVariant) -> str:
        """Extract text content from chapter for engagement analysis."""
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
    
    async def _analyze_emotional_journey(self, text: str, chapter: ChapterVariant, 
                                       story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze the emotional journey and beats in the chapter."""
        # Placeholder implementation for emotional analysis
        
        # Identify emotional beats (simplified)
        emotional_beats = []
        
        # Look for emotional indicators
        emotion_indicators = {
            'joy': ['smiled', 'laughed', 'happy', 'delighted', 'pleased'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
            'anger': ['angry', 'furious', 'rage', 'mad', 'annoyed'],
            'sadness': ['sad', 'crying', 'tears', 'grief', 'sorrow'],
            'surprise': ['surprised', 'shocked', 'amazed', 'stunned'],
            'tension': ['tense', 'nervous', 'edge', 'suspense', 'anticipation']
        }
        
        text_lower = text.lower()
        for emotion, indicators in emotion_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    beat = EmotionalBeat(
                        emotion_type=emotion,
                        intensity=0.7,  # Placeholder
                        location=f"Contains '{indicator}'",
                        effectiveness=0.8  # Placeholder
                    )
                    emotional_beats.append(beat)
                    break
        
        # Calculate emotional variety and impact
        unique_emotions = len(set(beat.emotion_type for beat in emotional_beats))
        emotional_variety_score = min(1.0, unique_emotions / self.config.specific_config.minimum_emotional_variety)
        
        avg_intensity = sum(beat.intensity for beat in emotional_beats) / max(len(emotional_beats), 1)
        
        score = (emotional_variety_score * 0.4 + avg_intensity * 0.6)
        
        return {
            'score': score,
            'beats': emotional_beats,
            'emotional_variety': unique_emotions,
            'average_intensity': avg_intensity
        }
    
    async def _analyze_curiosity_hooks(self, text: str, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze curiosity hooks and mystery elements."""
        hooks = []
        
        # Look for question-based hooks
        question_count = text.count('?')
        if question_count > 0:
            hooks.append(CuriosityHook(
                hook_type="questions",
                strength=min(1.0, question_count * 0.1),
                resolution_promise=True,
                description=f"Chapter contains {question_count} questions that engage curiosity"
            ))
        
        # Look for mystery indicators
        mystery_words = ['mystery', 'secret', 'hidden', 'unknown', 'wonder', 'curious', 'strange']
        mystery_count = sum(1 for word in mystery_words if word in text.lower())
        
        if mystery_count > 0:
            hooks.append(CuriosityHook(
                hook_type="mystery_elements",
                strength=min(1.0, mystery_count * 0.15),
                resolution_promise=False,
                description=f"Contains {mystery_count} mystery-inducing elements"
            ))
        
        # Calculate overall curiosity score
        if hooks:
            avg_strength = sum(hook.strength for hook in hooks) / len(hooks)
            score = min(1.0, avg_strength)
        else:
            score = 0.3  # Default low score if no hooks found
        
        return {
            'score': score,
            'hooks': hooks,
            'question_density': question_count / max(len(text.split()), 1)
        }
    
    async def _analyze_character_connection(self, text: str, chapter: ChapterVariant,
                                          story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze character relatability and connection potential."""
        # Placeholder implementation
        
        # Look for character development indicators
        character_indicators = ['thought', 'felt', 'remembered', 'realized', 'understood']
        character_development_count = sum(1 for indicator in character_indicators if indicator in text.lower())
        
        # Look for relatable situations
        relatable_situations = ['family', 'friend', 'work', 'home', 'love', 'fear', 'hope', 'dream']
        relatability_count = sum(1 for situation in relatable_situations if situation in text.lower())
        
        # Calculate connection score
        development_score = min(1.0, character_development_count * 0.1)
        relatability_score = min(1.0, relatability_count * 0.1)
        
        score = (development_score + relatability_score) / 2
        
        return {
            'score': score,
            'character_development_indicators': character_development_count,
            'relatability_factors': relatability_count
        }
    
    async def _analyze_pacing_momentum(self, text: str, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze pacing and momentum for reader engagement."""
        # Simple pacing analysis based on text structure
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Good pacing has varied paragraph and sentence lengths
        paragraph_variety = len(set(len(p.split()) // 10 for p in paragraphs)) / max(len(paragraphs), 1)
        sentence_variety = len(set(len(s.split()) // 5 for s in sentences)) / max(len(sentences), 1)
        
        variety_score = (paragraph_variety + sentence_variety) / 2
        
        # Action words indicate good momentum
        action_words = ['ran', 'jumped', 'rushed', 'hurried', 'moved', 'went', 'came', 'turned']
        action_count = sum(1 for word in action_words if word in text.lower())
        momentum_score = min(1.0, action_count * 0.05)
        
        score = (variety_score * 0.6 + momentum_score * 0.4)
        
        return {
            'score': score,
            'avg_paragraph_length': avg_paragraph_length,
            'avg_sentence_length': avg_sentence_length,
            'variety_score': variety_score,
            'momentum_indicators': action_count
        }
    
    async def _analyze_cliffhanger_effectiveness(self, text: str, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze chapter ending for cliffhanger effectiveness."""
        # Get last few sentences for cliffhanger analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        last_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
        ending_text = '. '.join(last_sentences).lower()
        
        # Look for cliffhanger indicators
        cliffhanger_indicators = [
            'suddenly', 'but then', 'however', 'until', 'when suddenly',
            'just as', 'but', 'then', 'unexpectedly', 'to be continued'
        ]
        
        cliffhanger_score = 0.0
        for indicator in cliffhanger_indicators:
            if indicator in ending_text:
                cliffhanger_score += 0.2
        
        # Questions at the end are strong cliffhangers
        if '?' in ending_text:
            cliffhanger_score += 0.3
        
        # Incomplete actions indicate cliffhangers
        incomplete_indicators = ['began to', 'started to', 'was about to', 'almost']
        for indicator in incomplete_indicators:
            if indicator in ending_text:
                cliffhanger_score += 0.25
        
        return {
            'score': min(1.0, cliffhanger_score),
            'ending_analysis': ending_text[:100] + "..." if len(ending_text) > 100 else ending_text
        }
    
    async def _analyze_commercial_viability(self, text: str, chapter: ChapterVariant,
                                          market_intelligence: Optional[MarketIntelligence]) -> Dict[str, Any]:
        """Analyze commercial viability and market appeal."""
        # Base commercial score on engagement factors
        base_score = 0.6
        
        market_alignment = 0.7  # Placeholder
        
        if market_intelligence and self.config.specific_config.enable_market_alignment:
            # Analyze alignment with market trends
            # This would involve checking against current market preferences
            market_alignment = 0.8  # Placeholder for actual market analysis
        
        score = (base_score + market_alignment) / 2
        
        return {
            'score': score,
            'market_alignment': market_alignment,
            'commercial_factors': {
                'genre_appeal': 0.7,
                'demographic_fit': 0.8,
                'trend_alignment': market_alignment
            }
        }
    
    def _calculate_overall_engagement_score(self, emotional_score: float, curiosity_score: float,
                                          character_score: float, pacing_score: float,
                                          commercial_score: float) -> float:
        """Calculate weighted overall engagement score."""
        config = self.config.specific_config
        
        weights = {
            'emotional': config.emotional_impact_weight,
            'curiosity': config.curiosity_factor_weight,
            'commercial': config.commercial_viability_weight,
            'character': 0.15,
            'pacing': 0.1
        }
        
        return (emotional_score * weights['emotional'] +
                curiosity_score * weights['curiosity'] +
                commercial_score * weights['commercial'] +
                character_score * weights['character'] +
                pacing_score * weights['pacing'])
    
    async def _predict_reader_retention(self, overall_score: float,
                                       emotional_result: Dict[str, Any],
                                       curiosity_result: Dict[str, Any]) -> float:
        """Predict likelihood of reader retention."""
        # Simple retention prediction based on engagement factors
        base_retention = overall_score
        
        # Boost for strong emotional variety
        if emotional_result['emotional_variety'] >= 3:
            base_retention += 0.1
        
        # Boost for strong curiosity hooks
        if len(curiosity_result['hooks']) >= 2:
            base_retention += 0.1
        
        return min(1.0, base_retention)
    
    async def _assess_page_turner_quality(self, pacing_result: Dict[str, Any],
                                        curiosity_result: Dict[str, Any],
                                        cliffhanger_result: Dict[str, Any]) -> float:
        """Assess page-turning quality of the chapter."""
        return (pacing_result['score'] * 0.4 +
                curiosity_result['score'] * 0.4 +
                cliffhanger_result['score'] * 0.2)
    
    async def _generate_engagement_factors_analysis(self, *results) -> List[EngagementAnalysis]:
        """Generate detailed analysis for each engagement factor."""
        # Placeholder implementation
        return []
    
    async def _analyze_demographic_appeal(self, text: str, chapter: ChapterVariant,
                                        market_intelligence: MarketIntelligence) -> Dict[str, float]:
        """Analyze appeal to different demographic groups."""
        # Placeholder demographic analysis
        return {
            'young_adult': 0.7,
            'adult': 0.8,
            'general_fiction': 0.75
        }
    
    def _identify_engagement_strengths(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify engagement strengths from analysis results."""
        strengths = []
        
        emotional_result, curiosity_result, character_result, pacing_result = results[:4]
        
        if emotional_result['score'] > 0.8:
            strengths.append(f"Strong emotional impact with {emotional_result['emotional_variety']} emotion types")
        
        if curiosity_result['score'] > 0.7:
            strengths.append(f"Effective curiosity hooks with {len(curiosity_result['hooks'])} engagement elements")
        
        if character_result['score'] > 0.7:
            strengths.append("Strong character connection and relatability")
        
        return strengths
    
    def _generate_improvement_priorities(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized improvement suggestions."""
        priorities = []
        
        emotional_result, curiosity_result, character_result, pacing_result = results[:4]
        
        if emotional_result['score'] < 0.6:
            priorities.append("Increase emotional variety and intensity to improve reader connection")
        
        if curiosity_result['score'] < 0.5:
            priorities.append("Add more curiosity hooks and questions to maintain reader interest")
        
        if pacing_result['score'] < 0.6:
            priorities.append("Improve pacing variation and momentum to maintain engagement")
        
        return priorities
    
    def _predict_reader_experience(self, overall_score: float, results: List[Dict[str, Any]]) -> str:
        """Predict overall reader experience."""
        if overall_score >= 0.8:
            return "Highly engaging - readers likely to be captivated and eager to continue"
        elif overall_score >= 0.6:
            return "Moderately engaging - most readers will continue with interest"
        elif overall_score >= 0.4:
            return "Somewhat engaging - may lose some readers, needs improvement"
        else:
            return "Low engagement - significant improvements needed to retain readers"
    
    async def _update_engagement_history(self, assessment: ReaderEngagementAssessment,
                                       results: List[Dict[str, Any]]) -> None:
        """Update engagement analysis history for learning."""
        history_entry = {
            'timestamp': datetime.now(),
            'chapter_number': assessment.chapter_number,
            'overall_score': assessment.overall_engagement_score,
            'component_scores': {
                'emotional': assessment.emotional_impact_score,
                'curiosity': assessment.curiosity_factor_score,
                'character': assessment.character_connection_score,
                'pacing': assessment.pacing_momentum_score,
                'commercial': assessment.commercial_viability_score
            }
        }
        
        self._engagement_history.append(history_entry)
        
        # Keep history manageable
        if len(self._engagement_history) > 100:
            self._engagement_history = self._engagement_history[-50:]
    
    # Test methods for health checks
    
    async def _test_engagement_analysis(self) -> bool:
        """Test engagement analysis functionality."""
        try:
            test_text = "She was afraid, but curious about what lay behind the mysterious door. What secrets awaited her?"
            emotional_result = await self._analyze_emotional_journey(test_text, None, None)
            curiosity_result = await self._analyze_curiosity_hooks(test_text, None)
            
            return ('score' in emotional_result and 'score' in curiosity_result)
        except Exception:
            return False
"""
Reader Engagement Critic Component

Implements commercial viability assessment, emotional journey analysis, satisfaction
prediction, and market appeal evaluation for the adversarial system discriminator layer.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.market_intelligence import MarketIntelligence
from musequill.v3.models.reader_engagement_assessment import (
    ReaderEngagementAssessment, EmotionalBeat, QuestionAnswerAnalysis,
    SatisfactionPrediction, CliffhangerAnalysis, EmotionalResponse,
    EngagementRisk, CliffhangerType
)


class ReaderEngagementCriticConfig(BaseModel):
    """Configuration for Reader Engagement Critic component."""
    
    emotional_detection_sensitivity: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Sensitivity threshold for emotional beat detection"
    )
    
    question_tracking_enabled: bool = Field(
        default=True,
        description="Whether to track question planting and resolution"
    )
    
    cliffhanger_analysis_enabled: bool = Field(
        default=True,
        description="Whether to analyze chapter ending effectiveness"
    )
    
    commercial_focus_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight given to commercial appeal vs artistic merit"
    )
    
    target_demographic: str = Field(
        default="general_adult",
        description="Target reader demographic for assessment"
    )
    
    binge_reading_optimization: bool = Field(
        default=True,
        description="Whether to optimize for binge-reading behavior"
    )
    
    max_analysis_time_seconds: int = Field(
        default=40,
        ge=10,
        le=300,
        description="Maximum time for engagement analysis"
    )


class ReaderEngagementCriticInput(BaseModel):
    """Input data for Reader Engagement Critic."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate for reader engagement"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for context and progression analysis"
    )
    
    market_intelligence: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence for commercial assessment"
    )
    
    previous_engagement_scores: List[float] = Field(
        default_factory=list,
        description="Previous chapter engagement scores for trend analysis"
    )
    
    target_genre: Optional[str] = Field(
        default=None,
        description="Target genre for genre-specific engagement assessment"
    )


class ReaderEngagementCritic(BaseComponent[ReaderEngagementCriticInput, ReaderEngagementAssessment, ReaderEngagementCriticConfig]):
    """
    Reader Engagement Critic component for commercial viability assessment.
    
    Evaluates emotional journey effectiveness, question/answer balance,
    satisfaction potential, and market appeal with genre-aware criteria.
    """
    
    def __init__(self, config: ComponentConfiguration[ReaderEngagementCriticConfig]):
        super().__init__(config)
        self._emotional_patterns: Dict[str, List[str]] = {}
        self._engagement_history: List[Dict[str, Any]] = []
        self._market_benchmarks: Dict[str, float] = {}
        self._question_tracking_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize reader engagement analysis systems."""
        try:
            # Initialize emotional pattern recognition
            await self._load_emotional_patterns()
            
            # Initialize market benchmarks
            await self._load_market_benchmarks()
            
            # Initialize question tracking system
            if self.config.specific_config.question_tracking_enabled:
                await self._initialize_question_tracking()
            
            # Initialize cliffhanger analysis
            if self.config.specific_config.cliffhanger_analysis_enabled:
                await self._initialize_cliffhanger_analysis()
            
            # Initialize commercial assessment tools
            await self._initialize_commercial_analysis()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Reader engagement critic initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: ReaderEngagementCriticInput) -> ReaderEngagementAssessment:
        """
        Perform comprehensive reader engagement analysis.
        
        Args:
            input_data: Chapter variant and context for engagement assessment
            
        Returns:
            Detailed reader engagement assessment with commercial viability metrics
        """
        start_time = datetime.now()
        
        try:
            # Update market context if available
            if input_data.market_intelligence:
                await self._update_market_context(input_data.market_intelligence)
            
            # Perform parallel engagement analysis
            analysis_tasks = [
                self._analyze_emotional_journey(input_data.chapter_variant, input_data.story_state),
                self._analyze_question_answer_balance(input_data.chapter_variant, input_data.story_state),
                self._predict_reader_satisfaction(input_data.chapter_variant, input_data.story_state),
                self._analyze_cliffhanger_effectiveness(input_data.chapter_variant)
            ]
            
            # Execute analysis with timeout
            timeout_seconds = self.config.specific_config.max_analysis_time_seconds
            results = await asyncio.wait_for(
                asyncio.gather(*analysis_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            
            # Handle exceptions in analysis tasks
            emotional_result, qa_result, satisfaction_result, cliffhanger_result = results
            for result in results:
                if isinstance(result, Exception):
                    raise ComponentError(f"Engagement analysis subtask failed: {str(result)}")
            
            # Identify engagement risks and strengths
            engagement_risks = await self._identify_engagement_risks(
                input_data.chapter_variant, emotional_result, qa_result, satisfaction_result
            )
            
            engagement_strengths = await self._identify_engagement_strengths(
                emotional_result, qa_result, satisfaction_result, cliffhanger_result
            )
            
            # Generate predicted reader response
            predicted_response = await self._predict_reader_response(
                emotional_result, satisfaction_result, input_data.target_genre
            )
            
            # Generate engagement recommendations
            recommendations = await self._generate_engagement_recommendations(
                engagement_risks, emotional_result, qa_result, input_data.target_genre
            )
            
            # Compile comprehensive assessment
            assessment = await self._compile_engagement_assessment(
                input_data.chapter_variant,
                emotional_result,
                qa_result,
                satisfaction_result,
                cliffhanger_result,
                engagement_risks,
                engagement_strengths,
                predicted_response,
                recommendations
            )
            
            # Store engagement history for trend analysis
            self._engagement_history.append({
                'chapter_number': input_data.chapter_variant.chapter_number,
                'overall_score': assessment.overall_engagement_score,
                'timestamp': datetime.now()
            })
            
            return assessment
            
        except asyncio.TimeoutError:
            raise ComponentError(f"Reader engagement analysis exceeded {timeout_seconds}s timeout")
        except Exception as e:
            raise ComponentError(f"Reader engagement analysis failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on engagement analysis systems."""
        try:
            # Test emotional pattern recognition
            test_text = "She felt her heart racing as the door creaked open, revealing nothing but darkness."
            emotions = await self._detect_emotional_beats(test_text)
            if not emotions:
                return False
            
            # Test question detection
            test_questions = await self._detect_questions_planted("Who was behind the mysterious letter?")
            if len(test_questions) == 0:
                return False
            
            # Check component performance
            if self.state.metrics.failure_rate > 0.1:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup engagement analysis resources."""
        try:
            self._emotional_patterns.clear()
            self._engagement_history.clear()
            self._market_benchmarks.clear()
            self._question_tracking_cache.clear()
            
            return True
            
        except Exception:
            return False
    
    async def _load_emotional_patterns(self) -> None:
        """Load emotional pattern recognition data."""
        # Common emotional trigger patterns
        self._emotional_patterns = {
            'tension': [
                r'\b(nervous|anxious|worried|tense|afraid|scared)\b',
                r'\b(heart\s+(racing|pounding|skipped))\b',
                r'\b(breath\s+(caught|held|shallow))\b'
            ],
            'curiosity': [
                r'\b(wondered|curious|mystery|secret|hidden)\b',
                r'\?[^"]*$',  # Questions
                r'\b(what\s+if|how\s+could|why\s+would)\b'
            ],
            'satisfaction': [
                r'\b(relief|satisfied|content|pleased|happy)\b',
                r'\b(smiled|grinned|laughed)\b',
                r'\b(finally|at\s+last|success)\b'
            ],
            'surprise': [
                r'\b(suddenly|unexpected|shock|stunned|amazed)\b',
                r'\!',
                r'\b(never\s+thought|couldn\'t\s+believe)\b'
            ]
        }
    
    async def _load_market_benchmarks(self) -> None:
        """Load market performance benchmarks for comparison."""
        # Industry benchmarks for engagement metrics
        self._market_benchmarks = {
            'minimum_engagement_score': 0.65,
            'excellent_engagement_threshold': 0.85,
            'cliffhanger_effectiveness_target': 0.7,
            'emotional_variety_target': 4,  # Different emotion types per chapter
            'question_answer_ratio_optimal': 1.2,  # Slightly more questions than answers
            'binge_reading_score_target': 0.8
        }
    
    async def _initialize_question_tracking(self) -> None:
        """Initialize question and mystery tracking systems."""
        self._question_tracking_cache = {
            'planted_patterns': [
                r'\b(who\s+(?:is|was|will|could|would))\b',
                r'\b(what\s+(?:is|was|will|could|would|happened))\b',
                r'\b(why\s+(?:did|does|would|could))\b',
                r'\b(how\s+(?:did|does|will|could))\b',
                r'\b(where\s+(?:is|was|will|could))\b',
                r'\b(when\s+(?:will|did|does))\b'
            ],
            'mystery_indicators': [
                r'\b(mystery|secret|hidden|unknown|unexplained)\b',
                r'\b(strange|odd|peculiar|unusual|mysterious)\b',
                r'\b(disappeared|vanished|missing|gone)\b'
            ]
        }
    
    async def _initialize_cliffhanger_analysis(self) -> None:
        """Initialize cliffhanger effectiveness analysis."""
        # Cliffhanger pattern recognition would be initialized here
        pass
    
    async def _initialize_commercial_analysis(self) -> None:
        """Initialize commercial viability analysis tools."""
        # Commercial appeal metrics and genre expectations
        pass
    
    async def _update_market_context(self, market_intelligence: MarketIntelligence) -> None:
        """Update analysis context with current market intelligence."""
        # Update benchmarks based on current market trends
        for trend in market_intelligence.current_trends:
            if trend.trend_type.value == "pacing" and trend.popularity_score > 0.7:
                # Adjust expectations based on popular pacing trends
                pass
    
    async def _analyze_emotional_journey(self, chapter: ChapterVariant, 
                                        story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze emotional beats and journey effectiveness."""
        text = chapter.chapter_text
        
        # Detect emotional beats throughout chapter
        emotional_beats = []
        
        # Analyze text in chunks to track emotional progression
        text_chunks = self._chunk_text_for_analysis(text, chunk_size=500)
        
        for i, chunk in enumerate(text_chunks):
            chunk_emotions = await self._detect_emotional_beats(chunk)
            
            # Calculate placement appropriateness (emotional arc)
            placement_score = await self._calculate_emotional_placement_score(i, len(text_chunks))
            
            for emotion_type, effectiveness in chunk_emotions.items():
                if effectiveness > self.config.specific_config.emotional_detection_sensitivity:
                    emotional_beat = EmotionalBeat(
                        beat_type=emotion_type,
                        effectiveness=effectiveness,
                        placement_appropriateness=placement_score,
                        supporting_elements=[f"Detected in text chunk {i+1}"],
                        evidence_text=chunk[:100] + "..." if len(chunk) > 100 else chunk
                    )
                    emotional_beats.append(emotional_beat)
        
        # Calculate overall emotional journey score
        if emotional_beats:
            avg_effectiveness = sum(beat.effectiveness for beat in emotional_beats) / len(emotional_beats)
            avg_placement = sum(beat.placement_appropriateness for beat in emotional_beats) / len(emotional_beats)
            emotional_journey_score = (avg_effectiveness + avg_placement) / 2
        else:
            emotional_journey_score = 0.3  # Low score for no emotional engagement
        
        return {
            'emotional_journey_score': emotional_journey_score,
            'emotional_beats': emotional_beats
        }
    
    async def _analyze_question_answer_balance(self, chapter: ChapterVariant, 
                                              story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze balance of questions planted vs answers provided."""
        text = chapter.chapter_text
        
        # Detect new questions planted
        questions_planted = await self._detect_questions_planted(text)
        new_questions_count = len(questions_planted)
        
        # Estimate questions answered based on story state
        questions_answered = await self._estimate_questions_answered(text, story_state)
        answered_count = len(questions_answered)
        
        # Calculate quality scores
        question_quality = await self._assess_question_quality(questions_planted)
        answer_satisfaction = await self._assess_answer_satisfaction(questions_answered)
        
        # Assess balance appropriateness based on story position
        balance_appropriateness = await self._assess_qa_balance_appropriateness(
            new_questions_count, answered_count, story_state.story_completion_ratio
        )
        
        # Calculate outstanding question pressure
        total_outstanding = len(story_state.reader_expectations.planted_questions)
        outstanding_pressure = min(1.0, total_outstanding / 10.0)  # Normalize to 0-1
        
        qa_analysis = QuestionAnswerAnalysis(
            new_questions_planted=new_questions_count,
            existing_questions_answered=answered_count,
            question_quality_score=question_quality,
            answer_satisfaction_score=answer_satisfaction,
            balance_appropriateness=balance_appropriateness,
            outstanding_question_pressure=outstanding_pressure
        )
        
        # Calculate overall Q&A balance score
        qa_balance_score = (
            balance_appropriateness * 0.4 +
            question_quality * 0.3 +
            answer_satisfaction * 0.3
        )
        
        return {
            'question_answer_balance_score': qa_balance_score,
            'question_answer_analysis': qa_analysis
        }
    
    async def _predict_reader_satisfaction(self, chapter: ChapterVariant, 
                                          story_state: DynamicStoryState) -> Dict[str, Any]:
        """Predict reader satisfaction with chapter."""
        
        # Calculate immediate satisfaction based on chapter content
        immediate_factors = [
            await self._assess_pacing_satisfaction(chapter),
            await self._assess_content_satisfaction(chapter),
            await self._assess_emotional_satisfaction(chapter)
        ]
        immediate_satisfaction = sum(immediate_factors) / len(immediate_factors)
        
        # Calculate long-term satisfaction based on story progression
        long_term_factors = [
            await self._assess_plot_progression_satisfaction(chapter, story_state),
            await self._assess_character_development_satisfaction(chapter, story_state),
            await self._assess_world_building_satisfaction(chapter, story_state)
        ]
        long_term_satisfaction = sum(long_term_factors) / len(long_term_factors)
        
        # Assess reread value
        reread_value = await self._assess_reread_potential(chapter, story_state)
        
        # Assess recommendation likelihood
        recommendation_likelihood = await self._assess_recommendation_potential(
            immediate_satisfaction, long_term_satisfaction
        )
        
        # Identify satisfaction factors and risks
        satisfaction_factors = await self._identify_satisfaction_factors(chapter, story_state)
        dissatisfaction_risks = await self._identify_dissatisfaction_risks(chapter, story_state)
        
        satisfaction_prediction = SatisfactionPrediction(
            immediate_satisfaction=immediate_satisfaction,
            long_term_satisfaction=long_term_satisfaction,
            reread_value=reread_value,
            recommendation_likelihood=recommendation_likelihood,
            satisfaction_factors=satisfaction_factors,
            dissatisfaction_risks=dissatisfaction_risks
        )
        
        # Calculate overall satisfaction potential
        satisfaction_score = (immediate_satisfaction + long_term_satisfaction) / 2
        
        return {
            'satisfaction_potential_score': satisfaction_score,
            'satisfaction_prediction': satisfaction_prediction
        }
    
    async def _analyze_cliffhanger_effectiveness(self, chapter: ChapterVariant) -> Dict[str, Any]:
        """Analyze effectiveness of chapter ending."""
        
        # Extract chapter ending (last 200 words)
        words = chapter.chapter_text.split()
        ending_text = " ".join(words[-200:]) if len(words) > 200 else chapter.chapter_text
        
        # Classify cliffhanger type
        cliffhanger_type = await self._classify_cliffhanger_type(ending_text)
        
        # Assess effectiveness dimensions
        effectiveness_score = await self._assess_cliffhanger_effectiveness(ending_text, cliffhanger_type)
        stakes_clarity = await self._assess_stakes_clarity(ending_text)
        emotional_impact = await self._assess_ending_emotional_impact(ending_text)
        next_chapter_compulsion = await self._assess_next_chapter_compulsion(ending_text, cliffhanger_type)
        
        # Determine resolution timeline expectation
        resolution_timeline = await self._predict_resolution_timeline(ending_text, cliffhanger_type)
        
        cliffhanger_analysis = CliffhangerAnalysis(
            cliffhanger_type=cliffhanger_type,
            effectiveness_score=effectiveness_score,
            stakes_clarity=stakes_clarity,
            emotional_impact=emotional_impact,
            next_chapter_compulsion=next_chapter_compulsion,
            resolution_timeline=resolution_timeline
        )
        
        return {
            'cliffhanger_effectiveness_score': effectiveness_score,
            'cliffhanger_analysis': cliffhanger_analysis
        }
    
    async def _compile_engagement_assessment(self, chapter: ChapterVariant,
                                           emotional_result: Dict, qa_result: Dict,
                                           satisfaction_result: Dict, cliffhanger_result: Dict,
                                           engagement_risks: List[EngagementRisk],
                                           engagement_strengths: List[str],
                                           predicted_response: str,
                                           recommendations: List[str]) -> ReaderEngagementAssessment:
        """Compile comprehensive reader engagement assessment."""
        
        return ReaderEngagementAssessment(
            chapter_number=chapter.chapter_number,
            emotional_journey_score=emotional_result['emotional_journey_score'],
            question_answer_balance_score=qa_result['question_answer_balance_score'],
            satisfaction_potential_score=satisfaction_result['satisfaction_potential_score'],
            cliffhanger_effectiveness_score=cliffhanger_result['cliffhanger_effectiveness_score'],
            emotional_beats=emotional_result['emotional_beats'],
            question_answer_analysis=qa_result['question_answer_analysis'],
            satisfaction_prediction=satisfaction_result['satisfaction_prediction'],
            cliffhanger_analysis=cliffhanger_result['cliffhanger_analysis'],
            engagement_risks=engagement_risks,
            engagement_strengths=engagement_strengths,
            predicted_reader_response=predicted_response,
            engagement_recommendations=recommendations
        )
    
    # Implementation helper methods (many would be placeholders for actual NLP/ML models)
    
    def _chunk_text_for_analysis(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for sequential emotional analysis."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks or [text]
    
    async def _detect_emotional_beats(self, text: str) -> Dict[EmotionalResponse, float]:
        """Detect emotional beats in text chunk."""
        emotions = {}
        text_lower = text.lower()
        
        for emotion_name, patterns in self._emotional_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.2  # Simple scoring
            
            # Map emotion names to EmotionalResponse enum
            emotion_mapping = {
                'tension': EmotionalResponse.TENSION,
                'curiosity': EmotionalResponse.CURIOSITY,
                'satisfaction': EmotionalResponse.SATISFACTION,
                'surprise': EmotionalResponse.SURPRISE
            }
            
            if emotion_name in emotion_mapping and score > 0:
                emotions[emotion_mapping[emotion_name]] = min(1.0, score)
        
        return emotions
    
    async def _calculate_emotional_placement_score(self, chunk_index: int, total_chunks: int) -> float:
        """Calculate appropriateness of emotional beat placement."""
        if total_chunks == 1:
            return 0.8  # Single chunk gets neutral score
        
        # Simple scoring based on position (tension should build toward end)
        position_ratio = chunk_index / (total_chunks - 1) if total_chunks > 1 else 0
        
        # Assume most emotional beats are appropriately placed
        return 0.7 + (position_ratio * 0.2)
    
    async def _detect_questions_planted(self, text: str) -> List[str]:
        """Detect new questions or mysteries introduced."""
        questions = []
        
        # Look for explicit questions
        question_sentences = re.findall(r'[^.!?]*\?', text)
        questions.extend(question_sentences)
        
        # Look for mystery indicators
        for pattern in self._question_tracking_cache['mystery_indicators']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            questions.extend([f"Mystery: {match}" for match in matches])
        
        return questions
    
    async def _estimate_questions_answered(self, text: str, story_state: DynamicStoryState) -> List[str]:
        """Estimate which existing questions were answered."""
        # Placeholder - would analyze text against outstanding questions
        # and determine if any were resolved
        return []
    
    async def _assess_question_quality(self, questions: List[str]) -> float:
        """Assess intrigue level of planted questions."""
        if not questions:
            return 0.5
        
        # Simple heuristic - questions with specific details are more intriguing
        quality_scores = []
        for question in questions:
            if len(question.split()) > 5:  # More detailed questions
                quality_scores.append(0.8)
            else:
                quality_scores.append(0.6)
        
        return sum(quality_scores) / len(quality_scores)
    
    async def _assess_answer_satisfaction(self, answers: List[str]) -> float:
        """Assess satisfaction level of answers provided."""
        if not answers:
            return 0.5  # Neutral if no answers
        
        # Placeholder - would assess answer completeness and satisfaction
        return 0.7
    
    async def _assess_qa_balance_appropriateness(self, questions: int, answers: int, 
                                               story_position: float) -> float:
        """Assess appropriateness of question/answer balance for story position."""
        # Early in story: more questions than answers is good
        # Late in story: more answers than questions is expected
        
        if story_position < 0.3:  # Early story
            if questions >= answers:
                return 0.9
            else:
                return 0.6
        elif story_position > 0.8:  # Late story  
            if answers >= questions:
                return 0.9
            else:
                return 0.5
        else:  # Middle story
            ratio = questions / max(1, answers)
            if 0.8 <= ratio <= 1.5:  # Balanced
                return 0.8
            else:
                return 0.6
    
    async def _assess_pacing_satisfaction(self, chapter: ChapterVariant) -> float:
        """Assess reader satisfaction with pacing."""
        # Simple heuristic based on word count and scene count
        words_per_scene = chapter.word_count / max(1, chapter.scene_count)
        
        if 800 <= words_per_scene <= 1500:  # Good pacing
            return 0.8
        elif words_per_scene < 500:  # Too fast
            return 0.5
        elif words_per_scene > 2000:  # Too slow
            return 0.4
        else:
            return 0.7
    
    async def _assess_content_satisfaction(self, chapter: ChapterVariant) -> float:
        """Assess satisfaction with chapter content quality."""
        # Based on objective fulfillment from chapter variant
        return chapter.objective_fulfillment_score
    
    async def _assess_emotional_satisfaction(self, chapter: ChapterVariant) -> float:
        """Assess emotional satisfaction of chapter."""
        # Simple heuristic based on emotional beats achieved
        emotional_beats_count = len(chapter.emotional_beats_achieved)
        
        if emotional_beats_count >= 3:
            return 0.8
        elif emotional_beats_count >= 2:
            return 0.7
        elif emotional_beats_count >= 1:
            return 0.6
        else:
            return 0.3
    
    async def _assess_plot_progression_satisfaction(self, chapter: ChapterVariant, 
                                                  story_state: DynamicStoryState) -> float:
        """Assess satisfaction with plot progression."""
        threads_advanced = len(chapter.plot_threads_advanced)
        total_active_threads = len([t for t in story_state.plot_threads.values() 
                                  if t.status.value == "active"])
        
        if total_active_threads == 0:
            return 0.5
        
        advancement_ratio = threads_advanced / total_active_threads
        return min(1.0, advancement_ratio * 2)  # Scale to reasonable satisfaction
    
    async def _assess_character_development_satisfaction(self, chapter: ChapterVariant, 
                                                       story_state: DynamicStoryState) -> float:
        """Assess satisfaction with character development."""
        characters_featured = len(chapter.characters_featured)
        
        if characters_featured >= 2:
            return 0.8
        elif characters_featured >= 1:
            return 0.7
        else:
            return 0.4
    
    async def _assess_world_building_satisfaction(self, chapter: ChapterVariant, 
                                                story_state: DynamicStoryState) -> float:
        """Assess satisfaction with world building elements."""
        new_elements = len(chapter.new_world_elements)
        
        if new_elements >= 2:
            return 0.8
        elif new_elements >= 1:
            return 0.7
        else:
            return 0.6  # Not all chapters need new world building
    
    async def _assess_reread_potential(self, chapter: ChapterVariant, 
                                     story_state: DynamicStoryState) -> float:
        """Assess likelihood of chapter being reread."""
        # Chapters with high emotional impact and complexity have higher reread value
        complexity_score = chapter.calculate_narrative_density()
        emotional_count = len(chapter.emotional_beats_achieved)
        
        reread_score = (complexity_score + (emotional_count / 5)) / 2
        return min(1.0, reread_score)
    
    async def _assess_recommendation_potential(self, immediate_sat: float, 
                                             long_term_sat: float) -> float:
        """Assess likelihood reader would recommend based on satisfaction."""
        combined_satisfaction = (immediate_sat + long_term_sat) / 2
        
        # Recommendation threshold is higher than satisfaction
        if combined_satisfaction >= 0.8:
            return 0.9
        elif combined_satisfaction >= 0.7:
            return 0.7
        elif combined_satisfaction >= 0.6:
            return 0.5
        else:
            return 0.3
    
    async def _identify_satisfaction_factors(self, chapter: ChapterVariant, 
                                           story_state: DynamicStoryState) -> List[str]:
        """Identify elements contributing to reader satisfaction."""
        factors = []
        
        if chapter.objective_fulfillment_score >= 0.8:
            factors.append("Strong chapter objective fulfillment")
        
        if len(chapter.plot_threads_advanced) >= 2:
            factors.append("Good plot progression across multiple threads")
        
        if len(chapter.emotional_beats_achieved) >= 3:
            factors.append("Rich emotional journey")
        
        return factors
    
    async def _identify_dissatisfaction_risks(self, chapter: ChapterVariant, 
                                            story_state: DynamicStoryState) -> List[str]:
        """Identify elements that might reduce satisfaction."""
        risks = []
        
        if chapter.word_count < 1000:
            risks.append("Chapter may feel too short for reader satisfaction")
        
        if len(chapter.plot_threads_advanced) == 0:
            risks.append("No plot advancement may disappoint readers")
        
        if chapter.objective_fulfillment_score < 0.5:
            risks.append("Poor objective fulfillment may reduce satisfaction")
        
        return risks
    
    async def _classify_cliffhanger_type(self, ending_text: str) -> CliffhangerType:
        """Classify type of chapter ending."""
        ending_lower = ending_text.lower()
        
        if any(word in ending_lower for word in ['revealed', 'discovered', 'realized']):
            return CliffhangerType.REVELATION
        elif any(word in ending_lower for word in ['danger', 'threat', 'attack']):
            return CliffhangerType.DANGER
        elif any(word in ending_lower for word in ['decide', 'choice', 'decision']):
            return CliffhangerType.DECISION
        elif any(word in ending_lower for word in ['arrived', 'reached', 'entered']):
            return CliffhangerType.ARRIVAL
        elif '?' in ending_text:
            return CliffhangerType.DISCOVERY
        else:
            return CliffhangerType.NONE
    
    async def _assess_cliffhanger_effectiveness(self, ending_text: str, 
                                              cliffhanger_type: CliffhangerType) -> float:
        """Assess effectiveness of cliffhanger."""
        if cliffhanger_type == CliffhangerType.NONE:
            return 0.3
        
        # Simple heuristic based on cliffhanger type and text analysis
        if cliffhanger_type in [CliffhangerType.DANGER, CliffhangerType.REVELATION]:
            return 0.8
        elif cliffhanger_type in [CliffhangerType.DECISION, CliffhangerType.DISCOVERY]:
            return 0.7
        else:
            return 0.6
    
    async def _assess_stakes_clarity(self, ending_text: str) -> float:
        """Assess how clear the stakes are in the cliffhanger."""
        # Placeholder - would analyze text for clear consequences
        return 0.7
    
    async def _assess_ending_emotional_impact(self, ending_text: str) -> float:
        """Assess emotional impact of chapter ending."""
        # Look for emotional indicators in ending
        emotional_words = ['shocked', 'surprised', 'terrified', 'excited', 'devastated']
        
        ending_lower = ending_text.lower()
        emotional_count = sum(1 for word in emotional_words if word in ending_lower)
        
        return min(1.0, emotional_count * 0.3)
    
    async def _assess_next_chapter_compulsion(self, ending_text: str, 
                                            cliffhanger_type: CliffhangerType) -> float:
        """Assess how much ending compels reading next chapter."""
        base_compulsion = {
            CliffhangerType.DANGER: 0.9,
            CliffhangerType.REVELATION: 0.8,
            CliffhangerType.DISCOVERY: 0.8,
            CliffhangerType.DECISION: 0.7,
            CliffhangerType.ARRIVAL: 0.6,
            CliffhangerType.EMOTIONAL_PEAK: 0.7,
            CliffhangerType.CONFRONTATION: 0.8,
            CliffhangerType.NONE: 0.3
        }
        
        return base_compulsion.get(cliffhanger_type, 0.5)
    
    async def _predict_resolution_timeline(self, ending_text: str, 
                                         cliffhanger_type: CliffhangerType) -> str:
        """Predict when cliffhanger will likely be resolved."""
        if cliffhanger_type in [CliffhangerType.DANGER, CliffhangerType.CONFRONTATION]:
            return "immediate"
        elif cliffhanger_type in [CliffhangerType.DECISION, CliffhangerType.DISCOVERY]:
            return "short-term"
        else:
            return "long-term"
    
    async def _identify_engagement_risks(self, chapter: ChapterVariant,
                                       emotional_result: Dict, qa_result: Dict,
                                       satisfaction_result: Dict) -> List[EngagementRisk]:
        """Identify specific engagement risks."""
        risks = []
        
        if chapter.word_count > 4000:
            risks.append(EngagementRisk.PACING_TOO_SLOW)
        
        if len(emotional_result['emotional_beats']) == 0:
            risks.append(EngagementRisk.LACK_OF_STAKES)
        
        if qa_result['question_answer_analysis'].new_questions_planted == 0:
            risks.append(EngagementRisk.PREDICTABILITY)
        
        return risks
    
    async def _identify_engagement_strengths(self, emotional_result: Dict, qa_result: Dict,
                                           satisfaction_result: Dict, cliffhanger_result: Dict) -> List[str]:
        """Identify engagement strengths."""
        strengths = []
        
        if emotional_result['emotional_journey_score'] >= 0.8:
            strengths.append("Strong emotional engagement")
        
        if cliffhanger_result['cliffhanger_effectiveness_score'] >= 0.8:
            strengths.append("Highly effective chapter ending")
        
        if satisfaction_result['satisfaction_potential_score'] >= 0.8:
            strengths.append("High reader satisfaction potential")
        
        return strengths
    
    async def _predict_reader_response(self, emotional_result: Dict, 
                                     satisfaction_result: Dict, genre: Optional[str]) -> str:
        """Predict overall reader response to chapter."""
        emotional_score = emotional_result['emotional_journey_score']
        satisfaction_score = satisfaction_result['satisfaction_potential_score']
        
        combined_score = (emotional_score + satisfaction_score) / 2
        
        if combined_score >= 0.8:
            return "Readers likely to respond very positively with strong emotional engagement and high satisfaction."
        elif combined_score >= 0.6:
            return "Readers likely to respond positively with moderate engagement and satisfaction."
        else:
            return "Readers may respond lukewarmly due to limited emotional engagement or satisfaction."
    
    async def _generate_engagement_recommendations(self, risks: List[EngagementRisk],
                                                 emotional_result: Dict, qa_result: Dict,
                                                 genre: Optional[str]) -> List[str]:
        """Generate specific recommendations for improving engagement."""
        recommendations = []
        
        if EngagementRisk.PACING_TOO_SLOW in risks:
            recommendations.append("Tighten pacing by reducing word count or increasing action density")
        
        if EngagementRisk.LACK_OF_STAKES in risks:
            recommendations.append("Raise stakes and create more tension to engage readers")
        
        if emotional_result['emotional_journey_score'] < 0.6:
            recommendations.append("Strengthen emotional beats and character reactions")
        
        return recommendations
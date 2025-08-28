"""
LLM Discriminator Component

Implements LLM-based content discrimination and critique for the adversarial system.
Uses Large Language Model to provide comprehensive literary analysis and feedback.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import Field, BaseModel

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.market_intelligence import MarketIntelligence
from musequill.v3.llm.ollama.ollama_client import LLMService, create_llm_service


class LLMDiscriminatorConfig(BaseModel):
    """Configuration for LLM Discriminator component."""
    
    llm_model_name: str = Field(
        default="llama3.3:70b",
        description="LLM model to use for discrimination"
    )
    
    analysis_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM analysis (lower for more consistent critique)"
    )
    
    max_analysis_tokens: int = Field(
        default=2000,
        ge=500,
        le=8000,
        description="Maximum tokens for analysis response"
    )
    
    critique_depth: str = Field(
        default="comprehensive",
        description="Depth of critique: 'basic', 'detailed', 'comprehensive'"
    )
    
    focus_areas: List[str] = Field(
        default_factory=lambda: [
            "plot_coherence",
            "character_development", 
            "prose_quality",
            "pacing",
            "dialogue_authenticity",
            "emotional_resonance",
            "market_appeal"
        ],
        description="Areas to focus critique on"
    )
    
    scoring_strictness: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="How strict the scoring should be (1.0 = very strict)"
    )
    
    include_suggestions: bool = Field(
        default=True,
        description="Whether to include specific improvement suggestions"
    )
    
    max_analysis_time_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum time to spend on LLM analysis"
    )


class LLMDiscriminatorInput(BaseModel):
    """Input data for LLM Discriminator."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to critique"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for context"
    )
    
    market_intelligence: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence for commercial viability assessment"
    )
    
    previous_chapters_summary: Optional[str] = Field(
        default=None,
        description="Summary of previous chapters for continuity analysis"
    )
    
    critique_focus: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus the critique on"
    )


class LLMDiscriminatorOutput(BaseModel):
    """Output from LLM Discriminator."""
    
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score from LLM analysis"
    )
    
    dimension_scores: Dict[str, float] = Field(
        description="Scores for different critique dimensions"
    )
    
    strengths: List[str] = Field(
        description="Identified strengths in the chapter"
    )
    
    weaknesses: List[str] = Field(
        description="Identified weaknesses and issues"
    )
    
    specific_feedback: Dict[str, Any] = Field(
        description="Detailed feedback organized by category"
    )
    
    improvement_suggestions: List[str] = Field(
        description="Specific suggestions for improvement"
    )
    
    market_viability_assessment: Optional[str] = Field(
        default=None,
        description="Assessment of commercial/market appeal"
    )
    
    critique_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM's confidence in its critique"
    )
    
    analysis_metadata: Dict[str, Any] = Field(
        description="Metadata about the analysis process"
    )


class LLMDiscriminator(BaseComponent[LLMDiscriminatorInput, LLMDiscriminatorOutput, LLMDiscriminatorConfig]):
    """
    LLM-powered discriminator for comprehensive content critique.
    
    Uses Large Language Model to provide sophisticated literary analysis,
    combining multiple evaluation dimensions in a single AI-powered assessment.
    """
    
    def __init__(self, config: ComponentConfiguration[LLMDiscriminatorConfig]):
        super().__init__(config)
        self._llm_client: Optional[LLMService] = None
        self._analysis_history: List[Dict[str, Any]] = []
        self._critique_templates: Dict[str, str] = {}
        self._learned_patterns: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize the LLM discriminator with LLM client and critique templates."""
        try:
            # Initialize LLM client
            await self._initialize_llm_client()
            
            # Load critique templates
            await self._load_critique_templates()
            
            # Initialize learned patterns from previous critiques
            await self._load_learned_patterns()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"LLM discriminator initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: LLMDiscriminatorInput) -> LLMDiscriminatorOutput:
        """
        Analyze chapter variant using LLM-powered critique.
        
        Args:
            input_data: Chapter variant and context for analysis
            
        Returns:
            Comprehensive LLM-based critique and assessment
        """
        start_time = datetime.now()
        
        try:
            # Build analysis prompt based on configuration and input
            analysis_prompt = await self._build_analysis_prompt(input_data)
            
            # Get LLM critique
            llm_response = await self._get_llm_critique(analysis_prompt)
            
            # Parse and structure the LLM response
            structured_critique = await self._parse_llm_response(llm_response, input_data)
            
            # Apply scoring adjustments based on configuration
            adjusted_scores = await self._apply_scoring_adjustments(
                structured_critique, input_data
            )
            
            # Record analysis for learning
            await self._record_analysis(input_data, structured_critique, start_time)
            
            return structured_critique
            
        except Exception as e:
            self.state.last_error = f"LLM discrimination failed: {str(e)}"
            # Return default failure response
            return self._create_failure_response()
    
    async def health_check(self) -> bool:
        """Check if LLM discriminator is functioning correctly."""
        try:
            # Test LLM connectivity
            if not await self._test_llm_connection():
                return False
            
            # Check recent analysis success rate
            if self.state.metrics.failure_rate > 0.2:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup LLM discriminator resources."""
        try:
            if self._llm_client:
                await self._cleanup_llm_client()
            
            self._analysis_history.clear()
            
            return True
            
        except Exception:
            return False
    
    # Private Methods
    
    async def _initialize_llm_client(self) -> None:
        """Initialize LLM client for critique generation."""
        self._llm_client = create_llm_service()
        await self._llm_client.update_default_parameters({
            'model': self.config.specific_config.llm_model_name,
            'temperature': self.config.specific_config.analysis_temperature,
            'max_tokens': self.config.specific_config.max_analysis_tokens
        })
    
    async def _load_critique_templates(self) -> None:
        """Load templates for different types of critique."""
        self._critique_templates = {
            'comprehensive': self._get_comprehensive_template(),
            'detailed': self._get_detailed_template(),
            'basic': self._get_basic_template()
        }
    
    async def _load_learned_patterns(self) -> None:
        """Load patterns learned from previous successful critiques."""
        self._learned_patterns = {
            'effective_critique_approaches': [],
            'successful_suggestion_patterns': [],
            'reliable_scoring_indicators': {},
            'market_correlation_factors': []
        }
    
    async def _build_analysis_prompt(self, input_data: LLMDiscriminatorInput) -> str:
        """Build comprehensive analysis prompt for the LLM."""
        
        # Get base template based on critique depth
        template = self._critique_templates.get(
            self.config.specific_config.critique_depth,
            self._critique_templates['comprehensive']
        )
        
        # Determine focus areas
        focus_areas = (input_data.critique_focus or 
                      self.config.specific_config.focus_areas)
        
        # Build context information
        context_info = await self._build_context_section(input_data)
        
        # Build chapter information
        chapter_info = await self._build_chapter_section(input_data.chapter_variant)
        
        # Build market context if available
        market_context = ""
        if input_data.market_intelligence:
            market_context = await self._build_market_context(input_data.market_intelligence)
        
        # Assemble full prompt
        prompt = template.format(
            context_info=context_info,
            chapter_info=chapter_info,
            market_context=market_context,
            focus_areas=", ".join(focus_areas),
            scoring_strictness=self.config.specific_config.scoring_strictness,
            include_suggestions="Please include specific improvement suggestions." 
                              if self.config.specific_config.include_suggestions else ""
        )
        
        return prompt
    
    async def _get_llm_critique(self, prompt: str) -> str:
        """Get critique response from LLM."""
        
        try:
            response = await self._llm_client.generate(
                prompt=prompt,
                temperature=self.config.specific_config.analysis_temperature,
                max_tokens=self.config.specific_config.max_analysis_tokens
            )
            return response['response']
        
        except Exception as e:
            raise ComponentError(f"LLM critique generation failed: {str(e)}")
    
    async def _parse_llm_response(self, llm_response: str, 
                                 input_data: LLMDiscriminatorInput) -> LLMDiscriminatorOutput:
        """Parse and structure the LLM response into standardized output."""
        
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('{'):
                parsed_response = json.loads(llm_response)
                return await self._structure_json_response(parsed_response, input_data)
            else:
                return await self._structure_text_response(llm_response, input_data)
        
        except Exception as e:
            # Fallback to text parsing if JSON parsing fails
            return await self._structure_text_response(llm_response, input_data)
    
    async def _structure_json_response(self, parsed_response: Dict[str, Any], 
                                     input_data: LLMDiscriminatorInput) -> LLMDiscriminatorOutput:
        """Structure JSON response from LLM."""
        
        return LLMDiscriminatorOutput(
            overall_score=min(1.0, max(0.0, parsed_response.get('overall_score', 0.5))),
            dimension_scores=parsed_response.get('dimension_scores', {}),
            strengths=parsed_response.get('strengths', []),
            weaknesses=parsed_response.get('weaknesses', []),
            specific_feedback=parsed_response.get('specific_feedback', {}),
            improvement_suggestions=parsed_response.get('improvement_suggestions', []),
            market_viability_assessment=parsed_response.get('market_viability_assessment'),
            critique_confidence=min(1.0, max(0.0, parsed_response.get('confidence', 0.8))),
            analysis_metadata={
                'response_format': 'json',
                'analysis_timestamp': datetime.now(),
                'chapter_id': input_data.chapter_variant.variant_id
            }
        )
    
    async def _structure_text_response(self, text_response: str,
                                     input_data: LLMDiscriminatorInput) -> LLMDiscriminatorOutput:
        """Structure text response from LLM using pattern matching."""
        
        # Extract overall score using patterns
        overall_score = self._extract_overall_score(text_response)
        
        # Extract strengths and weaknesses
        strengths = self._extract_section_items(text_response, ['STRENGTHS', 'STRONG POINTS'])
        weaknesses = self._extract_section_items(text_response, ['WEAKNESSES', 'ISSUES', 'PROBLEMS'])
        
        # Extract suggestions
        suggestions = self._extract_section_items(text_response, ['SUGGESTIONS', 'RECOMMENDATIONS', 'IMPROVEMENTS'])
        
        # Extract dimension scores
        dimension_scores = self._extract_dimension_scores(text_response)
        
        return LLMDiscriminatorOutput(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            specific_feedback={'raw_analysis': text_response},
            improvement_suggestions=suggestions,
            market_viability_assessment=self._extract_market_assessment(text_response),
            critique_confidence=0.8,  # Default confidence for text parsing
            analysis_metadata={
                'response_format': 'text',
                'analysis_timestamp': datetime.now(),
                'chapter_id': input_data.chapter_variant.variant_id
            }
        )
    
    async def _apply_scoring_adjustments(self, critique: LLMDiscriminatorOutput,
                                       input_data: LLMDiscriminatorInput) -> LLMDiscriminatorOutput:
        """Apply configuration-based scoring adjustments."""
        
        # Apply strictness adjustment
        strictness = self.config.specific_config.scoring_strictness
        if strictness != 0.7:  # Default strictness
            adjustment_factor = strictness / 0.7
            critique.overall_score = min(1.0, max(0.0, critique.overall_score * adjustment_factor))
            
            # Adjust dimension scores too
            for key, score in critique.dimension_scores.items():
                critique.dimension_scores[key] = min(1.0, max(0.0, score * adjustment_factor))
        
        return critique
    
    async def _record_analysis(self, input_data: LLMDiscriminatorInput, 
                             critique: LLMDiscriminatorOutput, start_time: datetime) -> None:
        """Record analysis for learning and improvement."""
        
        analysis_record = {
            'timestamp': start_time,
            'chapter_id': input_data.chapter_variant.variant_id,
            'chapter_approach': input_data.chapter_variant.approach.value,
            'overall_score': critique.overall_score,
            'dimension_scores': critique.dimension_scores,
            'critique_confidence': critique.critique_confidence,
            'analysis_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        self._analysis_history.append(analysis_record)
        
        # Keep history manageable
        if len(self._analysis_history) > 100:
            self._analysis_history = self._analysis_history[-50:]
    
    def _create_failure_response(self) -> LLMDiscriminatorOutput:
        """Create a default failure response when analysis fails."""
        return LLMDiscriminatorOutput(
            overall_score=0.0,
            dimension_scores={},
            strengths=[],
            weaknesses=["Analysis failed - technical error occurred"],
            specific_feedback={'error': 'LLM analysis failed'},
            improvement_suggestions=["Unable to provide suggestions due to analysis failure"],
            critique_confidence=0.0,
            analysis_metadata={
                'response_format': 'error',
                'analysis_timestamp': datetime.now(),
                'error': 'Analysis failed'
            }
        )
    
    # Template Methods
    
    def _get_comprehensive_template(self) -> str:
        """Get comprehensive critique template."""
        return """
You are an expert literary critic and editor analyzing a chapter from a novel. Provide a comprehensive critique analyzing the following aspects:

CONTEXT INFORMATION:
{context_info}

CHAPTER TO ANALYZE:
{chapter_info}

MARKET CONTEXT:
{market_context}

ANALYSIS FOCUS: {focus_areas}

Please provide your analysis in the following JSON format:
{{
    "overall_score": [0.0-1.0 score],
    "dimension_scores": {{
        "plot_coherence": [0.0-1.0],
        "character_development": [0.0-1.0],
        "prose_quality": [0.0-1.0],
        "pacing": [0.0-1.0],
        "dialogue_authenticity": [0.0-1.0],
        "emotional_resonance": [0.0-1.0],
        "market_appeal": [0.0-1.0]
    }},
    "strengths": ["list of specific strengths"],
    "weaknesses": ["list of specific issues"],
    "specific_feedback": {{
        "plot": "detailed plot analysis",
        "characters": "character analysis",
        "prose": "prose quality assessment",
        "pacing": "pacing analysis",
        "dialogue": "dialogue assessment",
        "market": "commercial viability assessment"
    }},
    "improvement_suggestions": ["specific actionable suggestions"],
    "market_viability_assessment": "assessment of commercial appeal and market fit",
    "confidence": [0.0-1.0 confidence in analysis]
}}

Scoring Guidelines (strictness factor: {scoring_strictness}):
- 0.9-1.0: Exceptional, publishable quality
- 0.8-0.9: Strong, needs minor polish
- 0.7-0.8: Good foundation, needs revision
- 0.6-0.7: Decent but significant issues
- 0.5-0.6: Mediocre, major problems
- 0.0-0.5: Poor, substantial rewrite needed

{include_suggestions}
"""
    
    def _get_detailed_template(self) -> str:
        """Get detailed critique template."""
        return """
Analyze this chapter focusing on: {focus_areas}

CONTEXT: {context_info}
CHAPTER: {chapter_info}
MARKET: {market_context}

Provide scores (0.0-1.0) and specific feedback for each focus area.
Include 3-5 strengths, 3-5 weaknesses, and 5-8 improvement suggestions.

Strictness level: {scoring_strictness}
{include_suggestions}
"""
    
    def _get_basic_template(self) -> str:
        """Get basic critique template."""
        return """
Quick analysis of this chapter:

{chapter_info}

Focus on: {focus_areas}

Provide:
- Overall score (0.0-1.0)
- Top 3 strengths
- Top 3 issues
- 3 improvement suggestions

{include_suggestions}
"""
    
    # Utility Methods
    
    async def _build_context_section(self, input_data: LLMDiscriminatorInput) -> str:
        """Build context information section."""
        context_parts = []
        
        # Story state information
        story_state = input_data.story_state
        context_parts.append(f"Chapter Number: {input_data.chapter_variant.chapter_number}")
        
        if hasattr(story_state, 'characters') and story_state.characters:
            char_names = [getattr(char, 'name', 'Unknown') for char in story_state.characters[:5]]
            context_parts.append(f"Main Characters: {', '.join(char_names)}")
        
        if hasattr(story_state, 'current_location'):
            context_parts.append(f"Current Setting: {story_state.current_location}")
        
        if input_data.previous_chapters_summary:
            context_parts.append(f"Previous Context: {input_data.previous_chapters_summary}")
        
        return "\n".join(context_parts)
    
    async def _build_chapter_section(self, chapter: ChapterVariant) -> str:
        """Build chapter information section."""
        chapter_parts = [
            f"Title: {chapter.chapter_title or 'Untitled'}",
            f"Approach: {chapter.approach.value}",
            f"Word Count: {chapter.word_count}",
            f"Content:\n{chapter.chapter_text}"
        ]
        
        return "\n".join(chapter_parts)
    
    async def _build_market_context(self, market_intel: MarketIntelligence) -> str:
        """Build market intelligence context."""
        if not market_intel:
            return "No market context available."
        
        return f"Market trends and reader preferences to consider in evaluation."
    
    def _extract_overall_score(self, text: str) -> float:
        """Extract overall score from text response."""
        # Look for patterns like "Overall: 0.8", "Score: 7/10", etc.
        import re
        
        patterns = [
            r'overall[:\s]+(\d*\.?\d+)',
            r'score[:\s]+(\d*\.?\d+)',
            r'rating[:\s]+(\d*\.?\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                score = float(match.group(1))
                # Normalize if necessary
                if score > 1.0:
                    score = score / 10.0  # Assume 0-10 scale
                return min(1.0, max(0.0, score))
        
        return 0.5  # Default score
    
    def _extract_section_items(self, text: str, section_headers: List[str]) -> List[str]:
        """Extract items from specific sections."""
        items = []
        
        for header in section_headers:
            pattern = f"{header}:?\n(.*?)(?:\n\n|$)"
            import re
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_content = match.group(1)
                # Split by bullet points or line breaks
                lines = [line.strip() for line in section_content.split('\n') 
                        if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))]
                items.extend([line[1:].strip() for line in lines])
        
        return items[:10]  # Limit to reasonable number
    
    def _extract_dimension_scores(self, text: str) -> Dict[str, float]:
        """Extract dimension scores from text."""
        scores = {}
        dimensions = ['plot', 'character', 'prose', 'pacing', 'dialogue', 'emotion', 'market']
        
        import re
        for dim in dimensions:
            pattern = f"{dim}[:\s]+(\d*\.?\d+)"
            match = re.search(pattern, text.lower())
            if match:
                score = float(match.group(1))
                if score > 1.0:
                    score = score / 10.0
                scores[dim] = min(1.0, max(0.0, score))
        
        return scores
    
    def _extract_market_assessment(self, text: str) -> Optional[str]:
        """Extract market viability assessment."""
        import re
        pattern = r'market[^.]*?([^.]*\.)'
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).capitalize()
        return None
    
    # Test Methods
    
    async def _test_llm_connection(self) -> bool:
        """Test LLM client connectivity."""
        if not self._llm_client:
            return False
        
        try:
            test_response = await self._llm_client.generate(
                "Test connection. Respond with: OK",
                temperature=0.1,
                max_tokens=10
            )
            return "ok" in test_response['response'].lower()
        except:
            return False
    
    async def _cleanup_llm_client(self) -> None:
        """Cleanup LLM client resources."""
        self._llm_client = None
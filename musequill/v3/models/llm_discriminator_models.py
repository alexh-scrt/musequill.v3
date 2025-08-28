"""
Enhanced model definitions to support LLM discriminator integration.
Extends existing models and adds new ones for comprehensive LLM-based critique.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# pylint: disable=locally-disabled, fixme, line-too-long, no-member

# =============================================================================
# Enhanced Critique Enums
# =============================================================================

class CritiqueDimension(str, Enum):
    """Dimensions that can be evaluated by the LLM discriminator."""
    PLOT_COHERENCE = "plot_coherence"
    CHARACTER_DEVELOPMENT = "character_development"
    PROSE_QUALITY = "prose_quality"
    PACING = "pacing"
    DIALOGUE_AUTHENTICITY = "dialogue_authenticity"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    MARKET_APPEAL = "market_appeal"
    ORIGINALITY = "originality"
    GENRE_CONVENTIONS = "genre_conventions"
    READER_ENGAGEMENT = "reader_engagement"
    WORLD_BUILDING = "world_building"
    THEMATIC_DEPTH = "thematic_depth"


class CritiqueConfidenceLevel(str, Enum):
    """Confidence levels for LLM critique assessments."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


class SuggestionType(str, Enum):
    """Types of improvement suggestions."""
    STRUCTURAL = "structural"
    STYLISTIC = "stylistic"
    CHARACTER_RELATED = "character_related"
    PLOT_RELATED = "plot_related"
    DIALOGUE = "dialogue"
    PACING = "pacing"
    MARKET_OPTIMIZATION = "market_optimization"
    TECHNICAL = "technical"


# =============================================================================
# LLM Critique Models
# =============================================================================

class ImprovementSuggestion(BaseModel):
    """Detailed improvement suggestion from LLM analysis."""
    
    suggestion_id: str = Field(
        description="Unique identifier for this suggestion"
    )
    
    type: SuggestionType = Field(
        description="Category of the suggestion"
    )
    
    priority: str = Field(
        description="Priority level: 'high', 'medium', 'low'"
    )
    
    description: str = Field(
        description="Detailed description of the suggested improvement"
    )
    
    rationale: str = Field(
        description="Explanation of why this improvement is needed"
    )
    
    specific_location: Optional[str] = Field(
        default=None,
        description="Specific text or section this applies to"
    )
    
    expected_impact: str = Field(
        description="Expected impact of implementing this suggestion"
    )
    
    difficulty: str = Field(
        description="Implementation difficulty: 'easy', 'moderate', 'hard'"
    )


class DimensionAnalysis(BaseModel):
    """Detailed analysis of a specific critique dimension."""
    
    dimension: CritiqueDimension = Field(
        description="The dimension being analyzed"
    )
    
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numerical score for this dimension"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM's confidence in this assessment"
    )
    
    analysis_text: str = Field(
        description="Detailed textual analysis of this dimension"
    )
    
    strengths: List[str] = Field(
        description="Specific strengths in this dimension"
    )
    
    weaknesses: List[str] = Field(
        description="Specific weaknesses in this dimension"
    )
    
    suggestions: List[ImprovementSuggestion] = Field(
        description="Targeted suggestions for this dimension"
    )


class MarketViabilityAssessment(BaseModel):
    """Comprehensive market viability analysis from LLM."""
    
    overall_appeal_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall commercial appeal score"
    )
    
    target_audience_alignment: float = Field(
        ge=0.0,
        le=1.0,
        description="How well content aligns with target audience"
    )
    
    genre_expectations_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well content meets genre expectations"
    )
    
    market_positioning: str = Field(
        description="Suggested market positioning"
    )
    
    competitive_analysis: str = Field(
        description="Analysis relative to market competition"
    )
    
    reader_hook_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of reader engagement hooks"
    )
    
    commercial_potential: str = Field(
        description="Assessment of commercial potential"
    )
    
    marketing_angles: List[str] = Field(
        description="Suggested marketing angles"
    )


class LLMCritiqueMetadata(BaseModel):
    """Metadata about the LLM critique process."""
    
    llm_model: str = Field(
        description="LLM model used for critique"
    )
    
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the analysis was performed"
    )
    
    analysis_duration_seconds: float = Field(
        ge=0.0,
        description="Time taken for analysis"
    )
    
    prompt_template_version: str = Field(
        description="Version of prompt template used"
    )
    
    temperature_used: float = Field(
        description="Temperature parameter used for LLM"
    )
    
    tokens_consumed: Optional[int] = Field(
        default=None,
        description="Total tokens consumed by analysis"
    )
    
    critique_depth: str = Field(
        description="Depth of critique: 'basic', 'detailed', 'comprehensive'"
    )
    
    focus_areas: List[str] = Field(
        description="Areas the critique focused on"
    )


# =============================================================================
# Enhanced LLM Discriminator Output Model
# =============================================================================

class ComprehensiveLLMCritique(BaseModel):
    """Comprehensive critique output from LLM discriminator."""
    
    # Core Assessment
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score"
    )
    
    confidence_level: CritiqueConfidenceLevel = Field(
        description="LLM's confidence in the critique"
    )
    
    # Dimensional Analysis
    dimension_analyses: Dict[CritiqueDimension, DimensionAnalysis] = Field(
        description="Detailed analysis for each dimension"
    )
    
    # High-Level Assessment
    executive_summary: str = Field(
        description="Executive summary of the critique"
    )
    
    strengths_summary: List[str] = Field(
        description="Key strengths identified"
    )
    
    critical_issues: List[str] = Field(
        description="Critical issues that must be addressed"
    )
    
    # Improvement Guidance
    priority_suggestions: List[ImprovementSuggestion] = Field(
        description="Prioritized improvement suggestions"
    )
    
    revision_strategy: str = Field(
        description="Recommended overall revision strategy"
    )
    
    # Market Analysis
    market_assessment: Optional[MarketViabilityAssessment] = Field(
        default=None,
        description="Market viability assessment if requested"
    )
    
    # Comparative Analysis
    approach_effectiveness: str = Field(
        description="Assessment of the chapter approach used"
    )
    
    alternative_approaches: List[str] = Field(
        description="Suggested alternative approaches"
    )
    
    # Quality Gates
    publication_readiness: float = Field(
        ge=0.0,
        le=1.0,
        description="Assessment of publication readiness"
    )
    
    revision_urgency: str = Field(
        description="Urgency of needed revisions: 'low', 'medium', 'high', 'critical'"
    )
    
    # Metadata
    critique_metadata: LLMCritiqueMetadata = Field(
        description="Metadata about the critique process"
    )


# =============================================================================
# Comparative Analysis Models
# =============================================================================

class CritiqueComparison(BaseModel):
    """Comparison between traditional and LLM-based critiques."""
    
    chapter_variant_id: str = Field(
        description="ID of the chapter variant analyzed"
    )
    
    traditional_critics_summary: Dict[str, Any] = Field(
        description="Summary of traditional rule-based critics"
    )
    
    llm_critique: ComprehensiveLLMCritique = Field(
        description="LLM-based comprehensive critique"
    )
    
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Agreement between traditional and LLM critiques"
    )
    
    divergent_assessments: List[str] = Field(
        description="Areas where traditional and LLM critiques disagree"
    )
    
    consensus_strengths: List[str] = Field(
        description="Strengths identified by both approaches"
    )
    
    consensus_weaknesses: List[str] = Field(
        description="Weaknesses identified by both approaches"
    )
    
    final_recommendation: str = Field(
        description="Final recommendation based on all critiques"
    )


class AdversarialFeedbackLoop(BaseModel):
    """Model for adversarial feedback between generator and discriminator."""
    
    iteration_number: int = Field(
        ge=1,
        description="Current iteration in the adversarial loop"
    )
    
    generator_approach: str = Field(
        description="Approach used by the generator"
    )
    
    discriminator_feedback: ComprehensiveLLMCritique = Field(
        description="Feedback from the discriminator"
    )
    
    adaptation_signals: Dict[str, Any] = Field(
        description="Signals for generator adaptation"
    )
    
    learning_insights: List[str] = Field(
        description="Insights learned from this iteration"
    )
    
    convergence_indicators: Dict[str, float] = Field(
        description="Indicators of generator-discriminator convergence"
    )
    
    next_iteration_strategy: str = Field(
        description="Strategy for the next iteration"
    )


# =============================================================================
# Learning and Adaptation Models
# =============================================================================

class LLMCritiquePattern(BaseModel):
    """Pattern identified from LLM critiques for learning."""
    
    pattern_id: str = Field(
        description="Unique identifier for this pattern"
    )
    
    pattern_type: str = Field(
        description="Type of pattern: 'success', 'failure', 'preference'"
    )
    
    description: str = Field(
        description="Description of the pattern"
    )
    
    frequency: int = Field(
        ge=1,
        description="How often this pattern has been observed"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this pattern"
    )
    
    applicable_contexts: List[str] = Field(
        description="Contexts where this pattern applies"
    )
    
    impact_on_scores: Dict[str, float] = Field(
        description="Impact of this pattern on different dimension scores"
    )
    
    first_observed: datetime = Field(
        description="When this pattern was first observed"
    )
    
    last_observed: datetime = Field(
        description="When this pattern was last observed"
    )


class AdaptiveCritiqueConfiguration(BaseModel):
    """Configuration that adapts based on LLM feedback patterns."""
    
    base_scoring_strictness: float = Field(
        ge=0.1,
        le=1.0,
        description="Base strictness level for scoring"
    )
    
    dimension_weights: Dict[CritiqueDimension, float] = Field(
        description="Weights for different critique dimensions"
    )
    
    learned_patterns: List[LLMCritiquePattern] = Field(
        description="Patterns learned from previous critiques"
    )
    
    adaptation_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate at which configuration adapts to new patterns"
    )
    
    context_sensitivity: Dict[str, Any] = Field(
        description="Context-sensitive adjustments"
    )
    
    market_alignment_factor: float = Field(
        ge=0.0,
        le=1.0,
        description="How much to weight market considerations"
    )


# =============================================================================
# Integration Models
# =============================================================================

class EnhancedQualityGate(BaseModel):
    """Enhanced quality gate that incorporates LLM discriminator results."""
    
    gate_id: str = Field(
        description="Unique identifier for this quality gate"
    )
    
    traditional_critics_pass: bool = Field(
        description="Whether traditional critics passed the chapter"
    )
    
    llm_discriminator_pass: bool = Field(
        description="Whether LLM discriminator passed the chapter"
    )
    
    combined_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined score from all critics"
    )
    
    minimum_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum threshold for passing"
    )
    
    gate_result: str = Field(
        description="Final gate result: 'pass', 'conditional_pass', 'fail'"
    )
    
    revision_required: bool = Field(
        description="Whether revision is required before proceeding"
    )
    
    critical_blockers: List[str] = Field(
        description="Critical issues that must be resolved"
    )
    
    recommendation: str = Field(
        description="Overall recommendation for this chapter"
    )


# =============================================================================
# Utility Functions
# =============================================================================

def create_comprehensive_critique_template() -> Dict[str, Any]:
    """Create a template for comprehensive LLM critique."""
    return {
        "analysis_prompt": """
        You are an expert literary critic and editor with deep knowledge of:
        - Commercial fiction market trends
        - Literary craft and technique
        - Reader psychology and engagement
        - Genre conventions and expectations
        - Publishing industry standards
        
        Analyze the provided chapter with professional expertise and provide:
        1. Overall quality assessment (0.0-1.0 score)
        2. Dimension-specific analysis for each requested area
        3. Specific strengths and weaknesses
        4. Actionable improvement suggestions
        5. Market viability assessment
        6. Publication readiness evaluation
        
        Be thorough, specific, and constructive in your analysis.
        """,
        
        "scoring_guidelines": {
            "0.9-1.0": "Exceptional quality, ready for publication",
            "0.8-0.9": "Strong quality, minor polish needed",
            "0.7-0.8": "Good foundation, needs targeted revision",
            "0.6-0.7": "Decent work, significant improvement needed",
            "0.5-0.6": "Below average, major revision required",
            "0.0-0.5": "Poor quality, substantial rewrite needed"
        },
        
        "focus_dimensions": [dim.value for dim in CritiqueDimension],
        
        "suggestion_categories": [stype.value for stype in SuggestionType]
    }


def calculate_confidence_level(score: float) -> CritiqueConfidenceLevel:
    """Convert numerical confidence score to confidence level enum."""
    if score >= 0.8:
        return CritiqueConfidenceLevel.VERY_HIGH
    elif score >= 0.6:
        return CritiqueConfidenceLevel.HIGH
    elif score >= 0.4:
        return CritiqueConfidenceLevel.MODERATE
    elif score >= 0.2:
        return CritiqueConfidenceLevel.LOW
    else:
        return CritiqueConfidenceLevel.VERY_LOW


# =============================================================================
# Example Usage
# =============================================================================

def create_sample_llm_critique() -> ComprehensiveLLMCritique:
    """Create a sample LLM critique for testing purposes."""
    
    # Create sample dimension analysis
    plot_analysis = DimensionAnalysis(
        dimension=CritiqueDimension.PLOT_COHERENCE,
        score=0.85,
        confidence=0.9,
        analysis_text="The plot progression is logical and well-paced, with clear cause-and-effect relationships.",
        strengths=["Strong causality", "Good pacing", "Clear stakes"],
        weaknesses=["Minor plot hole in timeline", "Predictable twist"],
        suggestions=[
            ImprovementSuggestion(
                suggestion_id="plot_001",
                type=SuggestionType.STRUCTURAL,
                priority="medium",
                description="Address the timeline inconsistency in the flashback sequence",
                rationale="The inconsistency may confuse readers about the chronology",
                expected_impact="Improved reader understanding and immersion",
                difficulty="easy"
            )
        ]
    )
    
    # Create sample market assessment
    market_assessment = MarketViabilityAssessment(
        overall_appeal_score=0.78,
        target_audience_alignment=0.82,
        genre_expectations_score=0.75,
        market_positioning="Contemporary thriller with strong character development",
        competitive_analysis="Compares favorably to recent bestsellers in the psychological thriller genre",
        reader_hook_strength=0.80,
        commercial_potential="High commercial potential with broad appeal",
        marketing_angles=["Character-driven thriller", "Unreliable narrator", "Psychological suspense"]
    )
    
    # Create critique metadata
    metadata = LLMCritiqueMetadata(
        llm_model="llama3.3:70b",
        analysis_duration_seconds=45.2,
        prompt_template_version="v1.0",
        temperature_used=0.2,
        tokens_consumed=1850,
        critique_depth="comprehensive",
        focus_areas=["plot_coherence", "character_development", "market_appeal"]
    )
    
    return ComprehensiveLLMCritique(
        overall_score=0.82,
        confidence_level=CritiqueConfidenceLevel.HIGH,
        dimension_analyses={CritiqueDimension.PLOT_COHERENCE: plot_analysis},
        executive_summary="A well-crafted chapter with strong character development and engaging plot progression.",
        strengths_summary=["Compelling character voice", "Effective tension building", "Strong emotional resonance"],
        critical_issues=["Timeline inconsistency needs addressing"],
        priority_suggestions=[plot_analysis.suggestions[0]],
        revision_strategy="Focus on tightening plot consistency while maintaining emotional impact",
        market_assessment=market_assessment,
        approach_effectiveness="Character-focused approach works well for this story type",
        alternative_approaches=["Could benefit from more action-oriented scenes"],
        publication_readiness=0.78,
        revision_urgency="medium",
        critique_metadata=metadata
    )


if __name__ == "__main__":
    # Create and display sample critique
    sample_critique = create_sample_llm_critique()
    
    print("🎭 Sample LLM Discriminator Critique")
    print("=" * 50)
    print(f"Overall Score: {sample_critique.overall_score:.2f}")
    print(f"Confidence Level: {sample_critique.confidence_level.value}")
    print(f"Publication Readiness: {sample_critique.publication_readiness:.2f}")
    print(f"Revision Urgency: {sample_critique.revision_urgency}")
    
    print(f"\nExecutive Summary: {sample_critique.executive_summary}")
    
    print(f"\nMarket Assessment Score: {sample_critique.market_assessment.overall_appeal_score:.2f}")
    print(f"Commercial Potential: {sample_critique.market_assessment.commercial_potential}")
    
    print("\nDimension Analysis:")
    for dimension, analysis in sample_critique.dimension_analyses.items():
        print(f"  {dimension.value}: {analysis.score:.2f} (confidence: {analysis.confidence:.2f})")
    
    print(f"\nAnalysis completed in {sample_critique.critique_metadata.analysis_duration_seconds:.1f} seconds")
    print(f"Tokens consumed: {sample_critique.critique_metadata.tokens_consumed}")
    
    # Demonstrate adaptive configuration
    print("\n📊 Creating Adaptive Configuration...")
    adaptive_config = AdaptiveCritiqueConfiguration(
        base_scoring_strictness=0.75,
        dimension_weights={
            CritiqueDimension.PLOT_COHERENCE: 0.2,
            CritiqueDimension.CHARACTER_DEVELOPMENT: 0.25,
            CritiqueDimension.PROSE_QUALITY: 0.2,
            CritiqueDimension.MARKET_APPEAL: 0.15,
            CritiqueDimension.PACING: 0.2
        },
        learned_patterns=[],
        adaptation_rate=0.1,
        context_sensitivity={"genre": "thriller", "target_audience": "adult"},
        market_alignment_factor=0.3
    )
    
    print("✅ Enhanced models and configurations ready for integration!")
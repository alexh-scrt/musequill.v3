"""
Example integration of LLMDiscriminator into the existing pipeline system.
Shows how to register, configure, and use the LLM-based discriminator.
"""

import asyncio
from typing import Dict, Any
from musequill.v3.components.base.component_interface import ComponentConfiguration, ComponentType
from musequill.v3.components.discriminators.llm_discriminator import (
    LLMDiscriminator, LLMDiscriminatorConfig, LLMDiscriminatorInput
)
from musequill.v3.models.chapter_variant import ChapterVariant, ChapterApproach
from musequill.v3.models.dynamic_story_state import DynamicStoryState


# =============================================================================
# Component Registration
# =============================================================================

def register_llm_discriminator():
    """Register LLMDiscriminator with the component registry."""
    from musequill.v3.components.component_registry import component_registry
    
    # Register the LLM discriminator component type
    component_registry.register_component_type(
        component_class=LLMDiscriminator,
        component_type=ComponentType.DISCRIMINATOR,
    )
    
    print("✅ LLMDiscriminator registered successfully")


# =============================================================================
# Configuration Examples
# =============================================================================

def create_comprehensive_llm_config() -> LLMDiscriminatorConfig:
    """Create configuration for comprehensive LLM critique."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.2,  # Low temperature for consistent analysis
        max_analysis_tokens=2500,
        critique_depth="comprehensive",
        focus_areas=[
            "plot_coherence",
            "character_development",
            "prose_quality", 
            "pacing",
            "dialogue_authenticity",
            "emotional_resonance",
            "market_appeal",
            "originality"
        ],
        scoring_strictness=0.8,  # Fairly strict scoring
        include_suggestions=True,
        max_analysis_time_seconds=90
    )


def create_fast_llm_config() -> LLMDiscriminatorConfig:
    """Create configuration for fast LLM critique."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.3,
        max_analysis_tokens=1000,
        critique_depth="basic",
        focus_areas=[
            "overall_quality",
            "readability",
            "engagement"
        ],
        scoring_strictness=0.6,  # More lenient for speed
        include_suggestions=False,
        max_analysis_time_seconds=30
    )


def create_market_focused_config() -> LLMDiscriminatorConfig:
    """Create configuration focused on market appeal."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.25,
        max_analysis_tokens=1500,
        critique_depth="detailed",
        focus_areas=[
            "market_appeal",
            "reader_engagement",
            "commercial_viability",
            "genre_expectations",
            "pacing"
        ],
        scoring_strictness=0.7,
        include_suggestions=True,
        max_analysis_time_seconds=60
    )


# =============================================================================
# Integration with Pipeline Orchestrator
# =============================================================================

async def integrate_llm_discriminator_into_pipeline():
    """Example of integrating LLM discriminator into the existing pipeline."""
    
    # Register the component
    register_llm_discriminator()
    
    # Create component configuration
    llm_config = create_comprehensive_llm_config()
    component_config = ComponentConfiguration(
        component_type=ComponentType.DISCRIMINATOR,
        component_name="LLM Literary Critic",
        specific_config=llm_config
    )
    
    # Create and initialize the discriminator
    llm_discriminator = LLMDiscriminator(component_config)
    
    # Initialize
    if await llm_discriminator.initialize():
        print("✅ LLM Discriminator initialized successfully")
    else:
        print("❌ Failed to initialize LLM Discriminator")
        return None
    
    return llm_discriminator


# =============================================================================
# Enhanced Pipeline Orchestrator with LLM Discriminator
# =============================================================================

class EnhancedPipelineOrchestrator:
    """Enhanced pipeline orchestrator that includes LLM discriminator."""
    
    def __init__(self):
        self._chapter_generator = None
        self._plot_coherence_critic = None
        self._literary_quality_critic = None
        self._reader_engagement_critic = None
        self._llm_discriminator = None  # New LLM-based critic
        self._quality_controller = None
    
    async def initialize_all_components(self) -> bool:
        """Initialize all components including the LLM discriminator."""
        try:
            # Register LLM discriminator
            register_llm_discriminator()
            
            # Create configurations
            component_configs = await self._create_enhanced_component_configurations()
            
            # Initialize existing components (chapter generator and other critics)
            # ... existing component initialization code ...
            
            # Initialize LLM Discriminator
            if 'llm_discriminator' in component_configs:
                from musequill.v3.components.component_registry import component_registry
                
                llm_critic_id = component_registry.create_component(
                    'llm_discriminator',
                    component_configs['llm_discriminator']
                )
                self._llm_discriminator = component_registry.get_component(llm_critic_id)
                
                if not await self._llm_discriminator.start():
                    print("❌ Failed to start LLM Discriminator")
                    return False
                
                print("✅ LLM Discriminator started successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced pipeline initialization failed: {e}")
            return False
    
    async def _create_enhanced_component_configurations(self) -> Dict[str, ComponentConfiguration]:
        """Create configurations including LLM discriminator."""
        configs = {
            # Existing component configurations...
            'chapter_generator': ComponentConfiguration(
                component_type=ComponentType.GENERATOR,
                component_name="Chapter Generator",
                specific_config={}
            ),
            
            # Traditional rule-based critics
            'plot_coherence_critic': ComponentConfiguration(
                component_type=ComponentType.DISCRIMINATOR,
                component_name="Plot Coherence Critic",
                specific_config={}
            ),
            'literary_quality_critic': ComponentConfiguration(
                component_type=ComponentType.DISCRIMINATOR,
                component_name="Literary Quality Critic", 
                specific_config={}
            ),
            'reader_engagement_critic': ComponentConfiguration(
                component_type=ComponentType.DISCRIMINATOR,
                component_name="Reader Engagement Critic",
                specific_config={}
            ),
            
            # New LLM-based discriminator
            'llm_discriminator': ComponentConfiguration(
                component_type=ComponentType.DISCRIMINATOR,
                component_name="LLM Literary Discriminator",
                specific_config=create_comprehensive_llm_config()
            ),
            
            'quality_controller': ComponentConfiguration(
                component_type=ComponentType.QUALITY_CONTROLLER,
                component_name="Enhanced Quality Controller",
                specific_config={}
            )
        }
        
        return configs
    
    async def process_chapter_with_enhanced_critique(self, chapter_input) -> Dict[str, Any]:
        """Process chapter with both traditional and LLM-based critique."""
        
        # Generate chapter variants
        generation_output = await self._chapter_generator.process(chapter_input)
        
        results = {}
        
        for variant in generation_output.chapter_variants:
            # Traditional rule-based critiques
            plot_critique = await self._plot_coherence_critic.process({
                'chapter_variant': variant,
                'story_state': chapter_input.story_state
            })
            
            literary_critique = await self._literary_quality_critic.process({
                'chapter_variant': variant,
                'story_state': chapter_input.story_state
            })
            
            engagement_critique = await self._reader_engagement_critic.process({
                'chapter_variant': variant,
                'story_state': chapter_input.story_state
            })
            
            # LLM-based comprehensive critique
            llm_input = LLMDiscriminatorInput(
                chapter_variant=variant,
                story_state=chapter_input.story_state,
                market_intelligence=getattr(chapter_input, 'market_intelligence', None),
                previous_chapters_summary=getattr(chapter_input, 'previous_chapters_summary', None)
            )
            
            llm_critique = await self._llm_discriminator.process(llm_input)
            
            # Combine all critiques
            combined_assessment = {
                'variant_id': variant.variant_id,
                'approach': variant.approach.value,
                'traditional_critiques': {
                    'plot_coherence': {
                        'score': plot_critique.continuity_score,
                        'issues': len(plot_critique.flagged_inconsistencies)
                    },
                    'literary_quality': {
                        'score': getattr(literary_critique, 'overall_score', 0.7)
                    },
                    'reader_engagement': {
                        'score': getattr(engagement_critique, 'engagement_score', 0.7)
                    }
                },
                'llm_critique': {
                    'overall_score': llm_critique.overall_score,
                    'dimension_scores': llm_critique.dimension_scores,
                    'strengths': llm_critique.strengths,
                    'weaknesses': llm_critique.weaknesses,
                    'suggestions': llm_critique.improvement_suggestions,
                    'confidence': llm_critique.critique_confidence
                },
                'combined_score': self._calculate_combined_score(
                    plot_critique, literary_critique, engagement_critique, llm_critique
                )
            }
            
            results[variant.variant_id] = combined_assessment
        
        return results
    
    def _calculate_combined_score(self, plot_critique, literary_critique, 
                                 engagement_critique, llm_critique) -> float:
        """Calculate combined score from all critiques."""
        
        # Traditional critics scores
        traditional_scores = [
            plot_critique.continuity_score,
            getattr(literary_critique, 'overall_score', 0.7),
            getattr(engagement_critique, 'engagement_score', 0.7)
        ]
        traditional_avg = sum(traditional_scores) / len(traditional_scores)
        
        # LLM critique score
        llm_score = llm_critique.overall_score
        
        # Weight LLM critique more heavily due to its comprehensive nature
        # But balance with traditional rule-based analysis
        combined_score = (traditional_avg * 0.4) + (llm_score * 0.6)
        
        return min(1.0, max(0.0, combined_score))


# =============================================================================
# Usage Example
# =============================================================================

async def example_usage():
    """Example of using the LLM discriminator in practice."""
    
    print("🔧 Setting up LLM Discriminator example...")
    
    # Create and initialize the discriminator
    llm_discriminator = await integrate_llm_discriminator_into_pipeline()
    if not llm_discriminator:
        return
    
    # Create sample chapter variant for testing
    sample_chapter = ChapterVariant(
        variant_id="test_001",
        chapter_number=5,
        approach=ChapterApproach.CHARACTER_FOCUSED,
        chapter_text="""
        Chapter 5: The Revelation
        
        Sarah stood at the edge of the cliff, her hands trembling as she held the letter. 
        The wind whipped through her hair, carrying with it the scent of the approaching storm. 
        
        "I never meant for you to find out this way," she whispered to the empty sky.
        
        The letter contained the truth she had been running from for three years. 
        Her father hadn't died in an accident—he had been murdered. And the killer 
        was someone she trusted, someone who had been by her side through everything.
        
        Lightning flashed across the dark clouds, illuminating her tear-stained face. 
        She had a choice to make: confront the truth or continue living the lie that 
        had become her life.
        """,
        chapter_title="The Revelation",
        word_count=125,
        characters_featured=["Sarah"],
        plot_threads_advanced=["father_mystery", "truth_revelation"],
        emotional_beats_achieved=["shock", "betrayal", "determination"]
    )
    
    # Create sample story state
    sample_story_state = DynamicStoryState()
    
    # Create input for LLM discriminator
    llm_input = LLMDiscriminatorInput(
        chapter_variant=sample_chapter,
        story_state=sample_story_state,
        previous_chapters_summary="Sarah has been investigating her father's death, gradually uncovering clues that point to a conspiracy."
    )
    
    # Process the chapter
    print("📝 Analyzing chapter with LLM discriminator...")
    critique = await llm_discriminator.process(llm_input)
    
    # Display results
    print("\n" + "="*60)
    print("🎭 LLM DISCRIMINATOR ANALYSIS RESULTS")
    print("="*60)
    print(f"Overall Score: {critique.overall_score:.2f}")
    print(f"Confidence: {critique.critique_confidence:.2f}")
    
    print("\n📊 Dimension Scores:")
    for dimension, score in critique.dimension_scores.items():
        print(f"  {dimension}: {score:.2f}")
    
    print("\n✅ Strengths:")
    for strength in critique.strengths:
        print(f"  • {strength}")
    
    print("\n⚠️ Weaknesses:")
    for weakness in critique.weaknesses:
        print(f"  • {weakness}")
    
    print("\n💡 Improvement Suggestions:")
    for suggestion in critique.improvement_suggestions:
        print(f"  • {suggestion}")
    
    if critique.market_viability_assessment:
        print(f"\n💰 Market Assessment: {critique.market_viability_assessment}")
    
    print("\n" + "="*60)
    
    # Cleanup
    await llm_discriminator.cleanup()
    print("✅ Example completed successfully")


# =============================================================================
# Configuration Comparison
# =============================================================================

def compare_discriminator_approaches():
    """Compare LLM discriminator with traditional rule-based critics."""
    
    comparison = {
        "Traditional Rule-Based Critics": {
            "advantages": [
                "Fast and deterministic",
                "Consistent scoring criteria", 
                "Specific domain expertise",
                "No API costs",
                "Explainable logic"
            ],
            "limitations": [
                "Limited by predefined rules",
                "Cannot adapt to creative innovations",
                "May miss subtle literary qualities",
                "Requires manual rule updates",
                "Cannot provide holistic assessment"
            ]
        },
        
        "LLM Discriminator": {
            "advantages": [
                "Holistic literary understanding",
                "Adapts to creative innovations",
                "Provides nuanced feedback",
                "Considers context and subtlety",
                "Can assess market appeal",
                "Offers specific improvement suggestions"
            ],
            "limitations": [
                "Higher computational cost",
                "Potential inconsistency",
                "Requires careful prompt engineering",
                "May have model biases",
                "Dependent on LLM availability"
            ]
        },
        
        "Hybrid Approach (Recommended)": {
            "strategy": [
                "Use rule-based critics for fast initial screening",
                "Apply LLM discriminator for comprehensive analysis",
                "Combine scores with weighted approach",
                "Use LLM for final quality decisions",
                "Traditional critics for consistency checks"
            ]
        }
    }
    
    return comparison


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
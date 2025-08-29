"""
Enhanced Component Registry Setup - WITH LLM DISCRIMINATOR INTEGRATION

Provides functions to register all components with the global component registry
including the LLMDiscriminator that uses models from musequill/v3/models/llm_discriminator_models.py
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

from typing import Dict, Any
from datetime import datetime
from musequill.v3.components.base.component_interface import (
    component_registry, ComponentConfiguration, ComponentType
)

# Import existing component classes
from musequill.v3.components.generators.chapter_generator import ChapterGenerator, ChapterGeneratorConfig
from musequill.v3.components.discriminators.plot_coherence_critic import PlotCoherenceCritic, PlotCoherenceCriticConfig
from musequill.v3.components.quality_control.comprehensive_quality_controller import ComprehensiveQualityController, QualityControllerConfig
from musequill.v3.components.researcher.researcher_agent import ResearcherComponent, ResearcherConfig
from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestratorConfig, OrchestrationStrategy
from musequill.v3.components.orchestration.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
from musequill.v3.components.discriminators.literary_quality_critic import LiteraryQualityCritic
from musequill.v3.components.discriminators.reader_engagement_critic import ReaderEngagementCritic
from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceEngine
# Import LLM Discriminator components
from musequill.v3.components.discriminators.llm_discriminator import LLMDiscriminator, LLMDiscriminatorConfig

# Import LLM discriminator models for proper configuration
from musequill.v3.models.llm_discriminator_models import (
    CritiqueDimension
)


def register_llm_discriminator_component() -> bool:
    """Register the LLM Discriminator component specifically."""
    try:
        # Register LLM Discriminator with proper component type
        component_registry.register_component_type("llm_discriminator", LLMDiscriminator)
        print("✅ LLMDiscriminator registered successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to register LLMDiscriminator: {e}")
        return False


def register_existing_components() -> bool:
    """Register only the existing component types (for current testing)."""
    try:
        # Register Generator components
        component_registry.register_component_type("chapter_generator", ChapterGenerator)
        
        # Register Discriminator components  
        component_registry.register_component_type("plot_coherence_critic", PlotCoherenceCritic)
        component_registry.register_component_type('literary_quality_critic', LiteraryQualityCritic)
        component_registry.register_component_type('reader_engagement_critic', ReaderEngagementCritic)
        register_llm_discriminator_component()

        # Register Quality Controller
        component_registry.register_component_type("quality_controller", ComprehensiveQualityController)

        # Market Intelligence engine
        component_registry.register_component_type('market_intelligence', MarketIntelligenceEngine)
        
        # Register Researcher
        component_registry.register_component_type("researcher", ResearcherComponent)
        
        # Register Orchestrator
        component_registry.register_component_type("pipeline_orchestrator", EnhancedPipelineOrchestrator)
        
        print(f"✅ Registered {len(component_registry.registered_types)} existing component types")
        return True
        
    except Exception as e:
        print(f"Failed to register existing components: {e}")
        return False


def register_all_components_with_llm() -> bool:
    """Register all component types including the LLM Discriminator."""
    try:
        # Register existing components first
        success = register_existing_components()
        if not success:
            return False
        
        # Register LLM Discriminator
        llm_success = register_llm_discriminator_component()
        if not llm_success:
            return False
        
        print(f"✅ All components registered with LLM. Total: {len(component_registry.registered_types)} types")
        return True
        
    except Exception as e:
        print(f"Failed to register all components with LLM: {e}")
        return False


def create_comprehensive_llm_discriminator_config() -> LLMDiscriminatorConfig:
    """Create a comprehensive configuration for LLM discriminator using the enhanced models."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.2,  # Low temperature for consistent analysis
        max_analysis_tokens=2500,
        critique_depth="comprehensive",
        focus_areas=[
            CritiqueDimension.PLOT_COHERENCE.value,
            CritiqueDimension.CHARACTER_DEVELOPMENT.value,
            CritiqueDimension.PROSE_QUALITY.value,
            CritiqueDimension.PACING.value,
            CritiqueDimension.DIALOGUE_AUTHENTICITY.value,
            CritiqueDimension.EMOTIONAL_RESONANCE.value,
            CritiqueDimension.MARKET_APPEAL.value,
            CritiqueDimension.ORIGINALITY.value
        ],
        scoring_strictness=0.8,  # Fairly strict scoring
        include_suggestions=True,
        max_analysis_time_seconds=90
    )


def create_market_focused_llm_config() -> LLMDiscriminatorConfig:
    """Create configuration focused on market appeal using the enhanced models."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.25,
        max_analysis_tokens=1500,
        critique_depth="detailed",
        focus_areas=[
            CritiqueDimension.MARKET_APPEAL.value,
            CritiqueDimension.READER_ENGAGEMENT.value,
            CritiqueDimension.GENRE_CONVENTIONS.value,
            CritiqueDimension.PACING.value,
            CritiqueDimension.EMOTIONAL_RESONANCE.value
        ],
        scoring_strictness=0.7,
        include_suggestions=True,
        max_analysis_time_seconds=60
    )


def create_fast_llm_config() -> LLMDiscriminatorConfig:
    """Create configuration for fast LLM critique using the enhanced models."""
    return LLMDiscriminatorConfig(
        llm_model_name="llama3.3:70b",
        analysis_temperature=0.3,
        max_analysis_tokens=1000,
        critique_depth="basic",
        focus_areas=[
            CritiqueDimension.PROSE_QUALITY.value,
            CritiqueDimension.READER_ENGAGEMENT.value,
            CritiqueDimension.PACING.value
        ],
        scoring_strictness=0.6,  # More lenient for speed
        include_suggestions=False,
        max_analysis_time_seconds=30
    )


def create_enhanced_component_configurations() -> Dict[str, ComponentConfiguration]:
    """Create configurations for all components including LLM discriminator."""
    
    # Create existing component configs
    chapter_gen_config = ChapterGeneratorConfig(
        max_chapter_length=3000,
        min_chapter_length=1000,
        creativity_level=0.8,
        style_consistency=0.9
    )
    
    plot_critic_config = PlotCoherenceCriticConfig(
        coherence_threshold=0.7,
        consistency_weight=0.6,
        logic_weight=0.4,
        enable_deep_analysis=True
    )
    
    quality_controller_config = QualityControllerConfig(
        quality_threshold=0.75,
        max_revision_attempts=3,
        require_all_critics=False,
        enable_progressive_standards=True
    )
    
    researcher_config = ResearcherConfig(
        tavily_api_key="",  # To be configured
        max_research_queries=5,
        enable_caching=True,
        cache_duration_hours=24
    )
    
    orchestrator_config = PipelineOrchestratorConfig(
        orchestration_strategy=OrchestrationStrategy.QUALITY_FIRST,
        max_concurrent_components=3,
        component_timeout_seconds=300,
        enable_checkpoints=True
    )
    
    # Create LLM discriminator config
    llm_discriminator_config = create_comprehensive_llm_discriminator_config()
    
    return {
        "chapter_generator": ComponentConfiguration(
            component_type=ComponentType.GENERATOR,
            component_name="Chapter Generator",
            version="1.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=300,
            auto_recycle_after_uses=100,
            recycle_on_error_count=3,
            specific_config=chapter_gen_config
        ),
        
        "plot_coherence_critic": ComponentConfiguration(
            component_type=ComponentType.DISCRIMINATOR,
            component_name="Plot Coherence Critic",
            version="1.0.0",
            max_concurrent_executions=2,
            execution_timeout_seconds=120,
            auto_recycle_after_uses=200,
            recycle_on_error_count=5,
            specific_config=plot_critic_config
        ),
        
        "llm_discriminator": ComponentConfiguration(
            component_type=ComponentType.DISCRIMINATOR,
            component_name="LLM Literary Discriminator",
            version="1.0.0",
            max_concurrent_executions=1,  # LLM analysis is resource intensive
            execution_timeout_seconds=120,  # Allow time for LLM processing
            auto_recycle_after_uses=50,   # Recycle more frequently due to LLM overhead
            recycle_on_error_count=3,
            specific_config=llm_discriminator_config
        ),
        
        "quality_controller": ComponentConfiguration(
            component_type=ComponentType.QUALITY_CONTROLLER,
            component_name="Comprehensive Quality Controller",
            version="1.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=120,
            auto_recycle_after_uses=500,
            recycle_on_error_count=3,
            specific_config=quality_controller_config
        ),
        
        "researcher": ComponentConfiguration(
            component_type=ComponentType.MARKET_INTELLIGENCE,
            component_name="Research Agent",
            version="1.0.0",
            max_concurrent_executions=2,
            execution_timeout_seconds=300,
            auto_recycle_after_uses=100,
            recycle_on_error_count=5,
            specific_config=researcher_config
        ),
        
        "pipeline_orchestrator": ComponentConfiguration(
            component_type=ComponentType.ORCHESTRATOR,
            component_name="Pipeline Orchestrator",
            version="1.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=1800,
            auto_recycle_after_uses=1000,
            recycle_on_error_count=2,
            specific_config=orchestrator_config
        )
    }


def create_llm_focused_configurations() -> Dict[str, ComponentConfiguration]:
    """Create configurations with multiple LLM discriminator variants."""
    base_configs = create_enhanced_component_configurations()
    
    # Add market-focused LLM discriminator
    market_llm_config = create_market_focused_llm_config()
    base_configs["llm_market_discriminator"] = ComponentConfiguration(
        component_type=ComponentType.DISCRIMINATOR,
        component_name="LLM Market-Focused Discriminator",
        version="1.0.0",
        max_concurrent_executions=1,
        execution_timeout_seconds=90,
        auto_recycle_after_uses=75,
        recycle_on_error_count=3,
        specific_config=market_llm_config
    )
    
    # Add fast LLM discriminator for quick feedback
    fast_llm_config = create_fast_llm_config()
    base_configs["llm_fast_discriminator"] = ComponentConfiguration(
        component_type=ComponentType.DISCRIMINATOR,
        component_name="LLM Fast Discriminator",
        version="1.0.0",
        max_concurrent_executions=2,  # Can handle more concurrent for speed
        execution_timeout_seconds=45,
        auto_recycle_after_uses=100,
        recycle_on_error_count=3,
        specific_config=fast_llm_config
    )
    
    return base_configs


def setup_enhanced_component_system() -> bool:
    """Setup component system with LLM discriminator integration."""
    try:
        # Register all components including LLM
        if not register_all_components_with_llm():
            return False
        
        # Create and store configurations
        component_configs = create_enhanced_component_configurations()
        
        print("📋 Enhanced Component Configuration Summary:")
        print("-" * 50)
        for name, config in component_configs.items():
            print(f"  {name}: {config.component_name} (v{config.version})")
            if name == "llm_discriminator":
                llm_config = config.specific_config
                print(f"    LLM Model: {llm_config.llm_model_name}")
                print(f"    Analysis Depth: {llm_config.critique_depth}")
                print(f"    Focus Areas: {len(llm_config.focus_areas)} dimensions")
                print(f"    Scoring Strictness: {llm_config.scoring_strictness}")
        
        print(f"\n✅ Enhanced component system ready with {len(component_configs)} components")
        print("🎭 LLM Discriminator fully integrated with enhanced models!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced component system setup failed: {e}")
        return False


def setup_llm_focused_system() -> bool:
    """Setup system with multiple LLM discriminator variants."""
    try:
        # Register all components including LLM
        if not register_all_components_with_llm():
            return False
        
        # Create LLM-focused configurations
        llm_configs = create_llm_focused_configurations()
        
        print("🎭 LLM-Focused Component Configuration Summary:")
        print("-" * 60)
        llm_discriminators = [name for name in llm_configs.keys() if 'llm' in name and 'discriminator' in name]
        print(f"📊 LLM Discriminator Variants: {len(llm_discriminators)}")
        
        for name in llm_discriminators:
            config = llm_configs[name]
            llm_config = config.specific_config
            print(f"\n  {config.component_name}:")
            print(f"    Model: {llm_config.llm_model_name}")
            print(f"    Depth: {llm_config.critique_depth}")
            print(f"    Focus Areas: {llm_config.focus_areas}")
            print(f"    Max Time: {llm_config.max_analysis_time_seconds}s")
            print(f"    Concurrent Executions: {config.max_concurrent_executions}")
        
        print(f"\n✅ LLM-focused system ready with {len(llm_discriminators)} LLM discriminator variants")
        return True
        
    except Exception as e:
        print(f"❌ LLM-focused system setup failed: {e}")
        return False


def validate_llm_discriminator_models():
    """Validate that all required LLM discriminator models are properly imported."""
    try:
        # Test model imports
        from musequill.v3.models.llm_discriminator_models import (
            ComprehensiveLLMCritique,
            CritiqueDimension,
            ImprovementSuggestion,
            MarketViabilityAssessment,
            DimensionAnalysis,
            LLMCritiqueMetadata,
            create_sample_llm_critique
        )
        
        # Test model creation
        sample_critique = create_sample_llm_critique()
        
        print("✅ All LLM discriminator models validated successfully")
        print(f"📊 Sample critique overall score: {sample_critique.overall_score}")
        print(f"🎯 Sample confidence level: {sample_critique.confidence_level.value}")
        print(f"📝 Critique dimensions available: {len(sample_critique.dimension_analyses)}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM discriminator model validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Enhanced Component Registry Setup with LLM Integration")
    print("=" * 70)
    
    # Validate models first
    print("\n1️⃣ Validating LLM Discriminator Models...")
    if not validate_llm_discriminator_models():
        print("❌ Model validation failed. Exiting.")
        exit(1)
    
    # Setup enhanced system
    print("\n2️⃣ Setting up Enhanced Component System...")
    if not setup_enhanced_component_system():
        print("❌ Enhanced system setup failed. Exiting.")
        exit(1)
    
    # Setup LLM-focused system variants
    print("\n3️⃣ Setting up LLM-Focused System Variants...")
    if not setup_llm_focused_system():
        print("❌ LLM-focused system setup failed. Exiting.")
        exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS: LLM Discriminator Integration Complete!")
    print("=" * 70)
    print("\n🎭 Available LLM Discriminator Variants:")
    print("   • llm_discriminator: Comprehensive literary analysis")
    print("   • llm_market_discriminator: Market-focused assessment")
    print("   • llm_fast_discriminator: Quick feedback for iterations")
    print("\n🔧 Integration Features:")
    print("   • Uses models from musequill/v3/models/llm_discriminator_models.py")
    print("   • Supports all CritiqueDimension types")
    print("   • Includes MarketViabilityAssessment")
    print("   • Provides ComprehensiveLLMCritique output")
    print("   • Configurable analysis depth and focus areas")
    print("\n✅ Ready for pipeline integration!")
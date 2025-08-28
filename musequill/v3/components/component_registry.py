"""
Component Registry Setup Utility

Provides functions to register all components with the global component registry
and create proper component configurations.
"""

from typing import Dict, Any
from musequill.v3.components.base.component_interface import (
    component_registry, ComponentConfiguration, ComponentType
)

# Import all component classes
from musequill.v3.components.generators.chapter_generator import ChapterGenerator, ChapterGeneratorConfig
from musequill.v3.components.discriminators.plot_coherence_critic import PlotCoherenceCritic, PlotCoherenceCriticConfig
from musequill.v3.components.quality_control.comprehensive_quality_controller import ComprehensiveQualityController, QualityControllerConfig
from musequill.v3.components.researcher.researcher_agent import ResearcherComponent, ResearcherConfig
from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestrator, PipelineOrchestratorConfig

# Import the new components we just created
# These would need to be saved to their respective files:
# - musequill/v3/components/discriminators/literary_quality_critic.py
# - musequill/v3/components/discriminators/reader_engagement_critic.py  
# - musequill/v3/components/market_intelligence/market_intelligence_engine.py

# For now, we'll reference them as if they exist
# from musequill.v3.components.discriminators.literary_quality_critic import LiteraryQualityCritic, LiteraryQualityCriticConfig
# from musequill.v3.components.discriminators.reader_engagement_critic import ReaderEngagementCritic, ReaderEngagementCriticConfig
# from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceEngine, MarketIntelligenceEngineConfig


def register_all_components() -> bool:
    """Register all available component types with the global registry."""
    try:
        # Register Generator components
        component_registry.register_component_type("chapter_generator", ChapterGenerator)
        
        # Register Discriminator components  
        component_registry.register_component_type("plot_coherence_critic", PlotCoherenceCritic)
        # component_registry.register_component_type("literary_quality_critic", LiteraryQualityCritic)
        # component_registry.register_component_type("reader_engagement_critic", ReaderEngagementCritic)
        
        # Register Quality Controller
        component_registry.register_component_type("quality_controller", ComprehensiveQualityController)
        
        # Register Market Intelligence
        # component_registry.register_component_type("market_intelligence_engine", MarketIntelligenceEngine)
        
        # Register Researcher
        component_registry.register_component_type("researcher", ResearcherComponent)
        
        # Register Orchestrator
        component_registry.register_component_type("pipeline_orchestrator", PipelineOrchestrator)
        
        return True
        
    except Exception as e:
        print(f"Failed to register components: {e}")
        return False


def create_default_component_configurations() -> Dict[str, ComponentConfiguration]:
    """Create default configurations for all components."""
    
    # Chapter Generator Configuration
    chapter_gen_config = ChapterGeneratorConfig(
        llm_model_name="claude-3-sonnet",
        max_generation_time_seconds=120,
        variants_to_generate=3,
        creativity_temperature=0.7,
        max_context_tokens=8000,
        adaptive_parameters=True,
        market_intelligence_weight=0.3,
        enable_self_critique=True,
        banned_phrases=[]
    )
    
    # Plot Coherence Critic Configuration
    plot_critic_config = PlotCoherenceCriticConfig(
        inconsistency_detection_threshold=0.7,
        minimum_advancement_score=0.4,
        character_knowledge_tracking=True,
        world_rules_enforcement=True,
        timeline_validation=True,
        max_analysis_time_seconds=30
    )
    
    # Literary Quality Critic Configuration (placeholder)
    literary_critic_config = {
        "prose_quality_threshold": 0.6,
        "originality_threshold": 0.5,
        "voice_authenticity_threshold": 0.7,
        "cliche_detection_sensitivity": 0.8,
        "max_analysis_time_seconds": 45,
        "enable_style_analysis": True,
        "enable_vocabulary_analysis": True,
        "minimum_word_count": 100
    }
    
    # Reader Engagement Critic Configuration (placeholder) 
    engagement_critic_config = {
        "engagement_threshold": 0.65,
        "commercial_viability_weight": 0.4,
        "emotional_impact_weight": 0.35,
        "curiosity_factor_weight": 0.25,
        "max_analysis_time_seconds": 40,
        "enable_market_alignment": True,
        "target_demographics": ["young_adult", "adult", "general_fiction"],
        "minimum_emotional_variety": 3
    }
    
    # Quality Controller Configuration
    quality_controller_config = QualityControllerConfig(
        plot_coherence_weight=0.35,
        literary_quality_weight=0.30,
        reader_engagement_weight=0.35,
        acceptance_threshold=0.75,
        revision_threshold=0.60,
        max_revision_cycles=3,
        enable_adaptive_thresholds=True,
        critical_issue_rejection=True,
        market_alignment_weight=0.15,
        enable_comprehensive_feedback=True
    )
    
    # Market Intelligence Engine Configuration (placeholder)
    market_intel_config = {
        "tavily_api_key": "your-tavily-api-key-here",
        "cache_duration_hours": 24,
        "max_queries_per_session": 10,
        "analysis_depth": "comprehensive",
        "target_markets": ["US", "UK", "Canada", "Australia"],
        "enable_trend_prediction": True,
        "competitor_tracking": True,
        "reader_sentiment_analysis": True,
        "max_analysis_time_seconds": 300
    }
    
    # Researcher Configuration
    researcher_config = ResearcherConfig(
        tavily_api_key="your-tavily-api-key-here",
        chroma_collection_name="musequill_research",
        embedding_model="nomic-embed-text",
        chunk_size=500,
        chunk_overlap=50,
        max_results_per_query=10,
        enable_caching=True,
        cache_duration_hours=24
    )
    
    # Pipeline Orchestrator Configuration  
    orchestrator_config = PipelineOrchestratorConfig(
        max_generation_attempts=3,
        max_revision_cycles=2,
        quality_gate_threshold=0.75,
        enable_market_intelligence=True,
        market_refresh_interval_hours=24,
        component_health_check_interval=300,
        fallback_on_component_failure=True,
        orchestration_strategy="quality_first",
        enable_learning_feedback=True,
        parallel_criticism=True,
        adaptive_quality_thresholds=True
    )
    
    # Create ComponentConfiguration objects
    configurations = {
        "chapter_generator": ComponentConfiguration(
            component_type=ComponentType.GENERATOR,
            component_name="Chapter Generator",
            version="1.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=300,
            auto_recycle_after_uses=100,
            recycle_on_error_count=5,
            specific_config=chapter_gen_config
        ),
        
        "plot_coherence_critic": ComponentConfiguration(
            component_type=ComponentType.DISCRIMINATOR,
            component_name="Plot Coherence Critic",
            version="1.0.0",
            max_concurrent_executions=3,
            execution_timeout_seconds=60,
            auto_recycle_after_uses=200,
            recycle_on_error_count=5,
            specific_config=plot_critic_config
        ),
        
        "literary_quality_critic": ComponentConfiguration(
            component_type=ComponentType.DISCRIMINATOR,
            component_name="Literary Quality Critic", 
            version="1.0.0",
            max_concurrent_executions=3,
            execution_timeout_seconds=90,
            auto_recycle_after_uses=200,
            recycle_on_error_count=5,
            specific_config=literary_critic_config
        ),
        
        "reader_engagement_critic": ComponentConfiguration(
            component_type=ComponentType.DISCRIMINATOR,
            component_name="Reader Engagement Critic",
            version="1.0.0", 
            max_concurrent_executions=3,
            execution_timeout_seconds=80,
            auto_recycle_after_uses=200,
            recycle_on_error_count=5,
            specific_config=engagement_critic_config
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
        
        "market_intelligence_engine": ComponentConfiguration(
            component_type=ComponentType.MARKET_INTELLIGENCE,
            component_name="Market Intelligence Engine",
            version="1.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=600,
            auto_recycle_after_uses=50,
            recycle_on_error_count=3,
            specific_config=market_intel_config
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
    
    return configurations


def setup_component_system() -> bool:
    """Complete setup of the component system."""
    try:
        # Register all component types
        success = register_all_components()
        if not success:
            return False
        
        print("✅ All component types registered successfully")
        
        # Create default configurations
        configurations = create_default_component_configurations()
        print(f"✅ Created {len(configurations)} component configurations")
        
        # Test component registry
        registered_types = list(component_registry.registered_types.keys())
        print(f"✅ Component registry contains: {registered_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component system setup failed: {e}")
        return False


def create_test_component_instance(component_type: str) -> str:
    """Create a test instance of a specific component type."""
    try:
        # Get default configurations
        configurations = create_default_component_configurations()
        
        if component_type not in configurations:
            raise ValueError(f"No configuration found for component type: {component_type}")
        
        config = configurations[component_type]
        
        # Create component instance
        component_id = component_registry.create_component(component_type, config)
        
        print(f"✅ Created test instance of {component_type} with ID: {component_id}")
        return component_id
        
    except Exception as e:
        print(f"❌ Failed to create {component_type} instance: {e}")
        return ""


async def test_component_lifecycle(component_id: str) -> bool:
    """Test the complete lifecycle of a component."""
    try:
        component = component_registry.get_component(component_id)
        if not component:
            print(f"❌ Component {component_id} not found in registry")
            return False
        
        print(f"Testing lifecycle for component: {component.config.component_name}")
        
        # Test initialization
        init_success = await component.initialize()
        print(f"  Initialize: {'✅' if init_success else '❌'}")
        
        # Test health check
        health_ok = await component.health_check()
        print(f"  Health Check: {'✅' if health_ok else '❌'}")
        
        # Test cleanup
        cleanup_success = await component.cleanup()
        print(f"  Cleanup: {'✅' if cleanup_success else '❌'}")
        
        return init_success and health_ok and cleanup_success
        
    except Exception as e:
        print(f"❌ Component lifecycle test failed: {e}")
        return False


def get_component_status_summary() -> Dict[str, Any]:
    """Get status summary of all registered components."""
    try:
        summary = {
            'registered_types': list(component_registry.registered_types.keys()),
            'active_components': len(component_registry.active_components),
            'component_details': []
        }
        
        for component_id, component in component_registry.active_components.items():
            details = {
                'id': component_id,
                'name': component.config.component_name,
                'type': component.config.component_type.value,
                'status': component.state.status.value,
                'health': component.state.health.value,
                'executions': component.state.metrics.total_invocations,
                'success_rate': component.state.metrics.success_rate,
                'error_count': component.state.error_count
            }
            summary['component_details'].append(details)
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    """Run component system setup if executed directly."""
    import asyncio
    
    async def main():
        print("🚀 Starting Component System Setup...")
        
        # Setup component system
        success = setup_component_system()
        if not success:
            print("❌ Setup failed")
            return
        
        # Test creating component instances
        print("\n🧪 Testing Component Creation...")
        
        # Test components that we know exist
        existing_components = [
            "chapter_generator",
            "plot_coherence_critic", 
            "quality_controller",
            "researcher"
        ]
        
        created_components = []
        for comp_type in existing_components:
            comp_id = create_test_component_instance(comp_type)
            if comp_id:
                created_components.append(comp_id)
        
        # Test component lifecycles
        print(f"\n🔄 Testing Component Lifecycles...")
        for comp_id in created_components:
            await test_component_lifecycle(comp_id)
        
        # Show final status
        print(f"\n📊 Final Component System Status:")
        status = get_component_status_summary()
        
        print(f"  Registered Types: {len(status['registered_types'])}")
        print(f"  Active Components: {status['active_components']}")
        
        for detail in status.get('component_details', []):
            print(f"    - {detail['name']}: {detail['status']} ({detail['health']})")
    
    asyncio.run(main())
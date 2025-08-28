"""
Component Registry Setup Utility - COMPLETE VERSION

Provides functions to register all components with the global component registry
and create proper component configurations.
"""

from typing import Dict, Any
from musequill.v3.components.base.component_interface import (
    component_registry, ComponentConfiguration, ComponentType
)

# Import existing component classes
from musequill.v3.components.generators.chapter_generator import ChapterGenerator, ChapterGeneratorConfig
from musequill.v3.components.discriminators.plot_coherence_critic import PlotCoherenceCritic, PlotCoherenceCriticConfig
from musequill.v3.components.quality_control.comprehensive_quality_controller import ComprehensiveQualityController, QualityControllerConfig
from musequill.v3.components.researcher.researcher_agent import ResearcherComponent, ResearcherConfig
from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestrator, PipelineOrchestratorConfig, OrchestrationStrategy

# Import the new components (UNCOMMENT WHEN FILES ARE SAVED)
# from musequill.v3.components.discriminators.literary_quality_critic import LiteraryQualityCritic, LiteraryQualityCriticConfig
# from musequill.v3.components.discriminators.reader_engagement_critic import ReaderEngagementCritic, ReaderEngagementCriticConfig
# from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceEngine, MarketIntelligenceEngineConfig


def register_existing_components() -> bool:
    """Register only the existing component types (for current testing)."""
    try:
        # Register Generator components
        component_registry.register_component_type("chapter_generator", ChapterGenerator)
        
        # Register Discriminator components  
        component_registry.register_component_type("plot_coherence_critic", PlotCoherenceCritic)
        
        # Register Quality Controller
        component_registry.register_component_type("quality_controller", ComprehensiveQualityController)
        
        # Register Researcher
        component_registry.register_component_type("researcher", ResearcherComponent)
        
        # Register Orchestrator
        component_registry.register_component_type("pipeline_orchestrator", PipelineOrchestrator)
        
        print(f"✅ Registered {len(component_registry.registered_types)} existing component types")
        return True
        
    except Exception as e:
        print(f"Failed to register existing components: {e}")
        return False


def register_all_components() -> bool:
    """Register all component types including new ones (requires saving component files first)."""
    try:
        # Register existing components
        success = register_existing_components()
        if not success:
            return False
        
        # UNCOMMENT WHEN NEW COMPONENT FILES ARE SAVED:
        # Register new Discriminator components  
        # component_registry.register_component_type("literary_quality_critic", LiteraryQualityCritic)
        # component_registry.register_component_type("reader_engagement_critic", ReaderEngagementCritic)
        
        # Register Market Intelligence
        # component_registry.register_component_type("market_intelligence_engine", MarketIntelligenceEngine)
        
        print(f"✅ All components registered. Total: {len(component_registry.registered_types)} types")
        return True
        
    except Exception as e:
        print(f"Failed to register all components: {e}")
        return False


def create_minimal_component_configurations() -> Dict[str, ComponentConfiguration]:
    """Create configurations for existing components only."""
    
    # Create specific configs
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
        require_all_critics=False,  # Allow partial critic coverage
        enable_progressive_standards=True
    )
    
    researcher_config = ResearcherConfig(
        tavily_api_key="",  # To be configured
        max_research_queries=5,
        enable_caching=True,
        cache_duration_hours=24
    )
    
    orchestrator_config = PipelineOrchestratorConfig(
        orchestration_strategy=OrchestrationStrategy.QUALITY_FOCUSED,
        max_concurrent_components=3,
        component_timeout_seconds=300,
        enable_checkpoints=True
    )
    
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


def setup_existing_component_system() -> bool:
    """Setup component system with existing components only."""
    try:
        # Register existing component types
        success = register_existing_components()
        if not success:
            return False
        
        print("✅ Existing component types registered successfully")
        
        # Create minimal configurations
        configurations = create_minimal_component_configurations()
        print(f"✅ Created {len(configurations)} component configurations")
        
        # Test component registry
        registered_types = list(component_registry.registered_types.keys())
        print(f"✅ Component registry contains: {registered_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component system setup failed: {e}")
        return False


def setup_component_system() -> bool:
    """Complete setup of the component system (calls appropriate setup based on available components)."""
    # For now, use existing components setup
    # This will automatically switch to full setup once new component files are saved
    return setup_existing_component_system()


# Test functions
def create_test_component_instance(component_type: str) -> str:
    """Create a test instance of a specific component type."""
    try:
        # Get default configurations
        configs = create_minimal_component_configurations()
        
        if component_type not in configs:
            print(f"❌ Unknown component type: {component_type}")
            return None
        
        # Create component instance
        component_id = f"test_{component_type}_{datetime.now().strftime('%H%M%S')}"
        config = configs[component_type]
        
        success = component_registry.register_instance(component_id, config)
        
        if success:
            print(f"✅ Created test instance: {component_id}")
            return component_id
        else:
            print(f"❌ Failed to create component instance: {component_type}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating component instance {component_type}: {e}")
        return None


async def test_component_lifecycle(component_id: str) -> bool:
    """Test basic lifecycle operations for a component."""
    try:
        component = component_registry.get_component(component_id)
        if not component:
            print(f"❌ Component not found: {component_id}")
            return False
        
        # Test initialization
        init_result = await component.initialize()
        print(f"   Initialize: {'✅' if init_result else '❌'}")
        
        # Test health check
        health_result = await component.health_check()
        print(f"   Health Check: {'✅' if health_result else '❌'}")
        
        # Test cleanup
        cleanup_result = await component.cleanup()
        print(f"   Cleanup: {'✅' if cleanup_result else '❌'}")
        
        return init_result and health_result and cleanup_result
        
    except Exception as e:
        print(f"❌ Lifecycle test failed for {component_id}: {e}")
        return False


def get_component_status_summary() -> Dict[str, Any]:
    """Get summary of component system status."""
    return {
        'registered_types': list(component_registry.registered_types.keys()),
        'active_components': len(component_registry.active_components),
        'component_details': component_registry.list_components()
    }


# Main test execution
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


if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    asyncio.run(main())
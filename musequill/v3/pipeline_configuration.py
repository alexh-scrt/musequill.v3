"""
Pipeline Configuration Classes

Defines configuration models for the enhanced pipeline system with component registry support.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Import existing configuration classes
from musequill.v3.components.orchestration.pipeline_orchestrator import (
    PipelineOrchestratorConfig, OrchestrationStrategy
)
from musequill.v3.components.base.component_interface import ComponentConfiguration

class ResearchIntegrationMode(str, Enum):
    """Research integration modes for the pipeline."""
    DISABLED = "disabled"
    MANUAL_ONLY = "manual_only"
    AUTO_TRIGGERS = "auto_triggers"
    CONTINUOUS = "continuous"


class PipelineConfiguration(BaseModel):
    """
    Enhanced configuration for the complete pipeline system.
    
    This configuration class unifies all pipeline settings including component
    configurations, research integration, and orchestration settings.
    """
    
    # Component configurations from registry
    component_configs: Dict[str, ComponentConfiguration] = Field(
        description="Component configurations from the registry"
    )
    
    # Orchestrator configuration (embedded)
    orchestrator_config: ComponentConfiguration[PipelineOrchestratorConfig] = Field(
        description="Pipeline orchestrator configuration"
    )
    
    # Pipeline behavior settings
    orchestration_strategy: str = Field(
        default="balanced",
        description="Overall orchestration strategy"
    )
    
    max_generation_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum generation attempts per chapter"
    )
    
    max_revision_cycles: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum revision cycles per chapter"
    )
    
    enable_progressive_quality: bool = Field(
        default=True,
        description="Whether to use progressive quality standards"
    )
    
    # Research integration settings
    enable_research_integration: bool = Field(
        default=True,
        description="Whether to enable research integration"
    )
    
    research_integration_mode: ResearchIntegrationMode = Field(
        default=ResearchIntegrationMode.AUTO_TRIGGERS,
        description="Mode of research integration"
    )
    
    research_triggers: List[str] = Field(
        default_factory=lambda: ["market_stale", "quality_drop"],
        description="Automatic research trigger conditions"
    )
    
    research_caching: bool = Field(
        default=True,
        description="Whether to cache research results"
    )
    
    # Performance and resource settings
    max_concurrent_components: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent component operations"
    )
    
    component_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for individual component operations"
    )
    
    enable_checkpoints: bool = Field(
        default=True,
        description="Whether to enable pipeline checkpoints"
    )
    
    memory_optimization: bool = Field(
        default=True,
        description="Whether to enable memory optimization"
    )
    
    # Quality control settings
    quality_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for acceptance"
    )
    
    enable_quality_gates: bool = Field(
        default=True,
        description="Whether to enable quality gates between stages"
    )
    
    progressive_quality_scaling: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="How much to increase quality standards per iteration"
    )
    
    # Error handling and recovery
    enable_automatic_recovery: bool = Field(
        default=True,
        description="Whether to enable automatic error recovery"
    )
    
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed operations"
    )
    
    fallback_strategies: bool = Field(
        default=True,
        description="Whether to enable fallback strategies"
    )
    
    # Logging and monitoring
    enable_detailed_logging: bool = Field(
        default=True,
        description="Whether to enable detailed pipeline logging"
    )
    
    log_component_interactions: bool = Field(
        default=True,
        description="Whether to log component interactions"
    )
    
    enable_performance_metrics: bool = Field(
        default=True,
        description="Whether to collect performance metrics"
    )
    
    # Output and persistence
    save_intermediate_results: bool = Field(
        default=True,
        description="Whether to save intermediate pipeline results"
    )
    
    checkpoint_frequency: int = Field(
        default=5,
        ge=1,
        description="How often to save checkpoints (every N operations)"
    )
    
    # Pipeline metadata
    pipeline_version: str = Field(
        default="3.0.0",
        description="Pipeline version identifier"
    )
    
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata"
    )


class EnhancedPipelineSettings(BaseModel):
    """
    Additional settings specifically for the enhanced pipeline runner.
    """
    
    # Registry-specific settings
    auto_recycle_components: bool = Field(
        default=True,
        description="Whether to automatically recycle components"
    )
    
    max_component_instances: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum component instances to maintain"
    )
    
    component_health_check_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Seconds between component health checks"
    )
    
    enable_component_metrics: bool = Field(
        default=True,
        description="Whether to collect component-level metrics"
    )
    
    # Advanced orchestration
    dynamic_resource_allocation: bool = Field(
        default=True,
        description="Whether to dynamically allocate resources"
    )
    
    adaptive_timeout_scaling: bool = Field(
        default=True,
        description="Whether to adapt timeouts based on performance"
    )
    
    intelligent_error_recovery: bool = Field(
        default=True,
        description="Whether to use intelligent error recovery strategies"
    )


# Configuration factory functions

# Add this to the pipeline_configuration.py file or modify the existing function

# Replace the create_pipeline_configuration_from_dict function in pipeline_configuration.py

def create_pipeline_configuration_from_dict(config_dict: Dict[str, Any]) -> PipelineOrchestratorConfig:
    """
    Create PipelineOrchestratorConfig from dictionary configuration.
    
    Args:
        config_dict: Configuration dictionary from YAML/JSON
        
    Returns:
        PipelineOrchestratorConfig instance
    """
    try:
        # Extract settings from config dict with defaults
        pipeline_settings = config_dict.get('pipeline', {})
        orchestration_settings = config_dict.get('orchestration', {})
        components_settings = config_dict.get('components', {})
        pipeline_orchestrator_settings = components_settings.get('pipeline_orchestrator', {})
        research_settings = config_dict.get('research', {})
        error_handling_settings = config_dict.get('error_handling', {})
        
        # Map orchestration strategy string to enum
        strategy_str = pipeline_settings.get('orchestration_strategy', 'balanced')
        if isinstance(strategy_str, str):
            strategy_mapping = {
                'balanced': OrchestrationStrategy.BALANCED,
                'quality_first': OrchestrationStrategy.QUALITY_FIRST,
                'speed_optimized': OrchestrationStrategy.SPEED_OPTIMIZED,
                'experimental': OrchestrationStrategy.EXPERIMENTAL
            }
            orchestration_strategy = strategy_mapping.get(strategy_str.lower(), OrchestrationStrategy.BALANCED)
        else:
            orchestration_strategy = strategy_str
        
        # Create PipelineOrchestratorConfig
        config = PipelineOrchestratorConfig(
            # Core orchestration settings
            max_generation_attempts=pipeline_settings.get('max_generation_attempts', 
                                   pipeline_orchestrator_settings.get('max_generation_attempts', 5)),
            max_revision_cycles=pipeline_settings.get('max_revision_cycles',
                              pipeline_orchestrator_settings.get('max_revision_cycles', 3)),
            orchestration_strategy=orchestration_strategy,
            
            # Component behavior settings
            parallel_variant_evaluation=orchestration_settings.get('enable_parallel_evaluation',
                                       pipeline_orchestrator_settings.get('parallel_variant_evaluation', True)),
            
            # Market intelligence settings
            enable_market_intelligence_refresh=research_settings.get('enable_research_integration',
                                              pipeline_orchestrator_settings.get('enable_market_intelligence_refresh', True)),
            market_refresh_interval_hours=research_settings.get('market_refresh_interval_hours',
                                         pipeline_orchestrator_settings.get('market_refresh_interval_hours', 24)),
            
            # Health and monitoring settings
            component_health_check_interval=orchestration_settings.get('component_health_check_interval',
                                           pipeline_orchestrator_settings.get('component_health_check_interval', 300)),
            
            # Adaptive orchestration
            enable_adaptive_orchestration=orchestration_settings.get('enable_adaptive_orchestration',
                                         pipeline_orchestrator_settings.get('enable_adaptive_orchestration', True)),
            
            # Timeout settings
            pipeline_timeout_minutes=orchestration_settings.get('pipeline_timeout_minutes',
                                    pipeline_orchestrator_settings.get('pipeline_timeout_minutes', 60)),
            
            # Logging settings
            enable_comprehensive_logging=orchestration_settings.get('enable_comprehensive_logging',
                                        pipeline_orchestrator_settings.get('enable_comprehensive_logging', True)),
            
            # Error handling
            fallback_on_component_failure=error_handling_settings.get('fallback_strategies',
                                         pipeline_orchestrator_settings.get('fallback_on_component_failure', True))
        )
        
        return config
        
    except Exception as e:
        # If anything fails, create a minimal valid configuration
        print(f"Warning: Error creating pipeline orchestrator configuration: {e}")
        print("Creating minimal default configuration...")
        
        return PipelineOrchestratorConfig(
            max_generation_attempts=5,
            max_revision_cycles=3,
            orchestration_strategy=OrchestrationStrategy.BALANCED,
            parallel_variant_evaluation=True,
            enable_market_intelligence_refresh=True,
            market_refresh_interval_hours=24,
            component_health_check_interval=300,
            enable_adaptive_orchestration=True,
            pipeline_timeout_minutes=60,
            enable_comprehensive_logging=True,
            fallback_on_component_failure=True
        )


def create_default_pipeline_configuration() -> PipelineOrchestratorConfig:
    """Create a default pipeline orchestrator configuration for testing/fallback."""
    
    return create_pipeline_configuration_from_dict({})


# Export main classes and functions
__all__ = [
    'PipelineConfiguration',
    'EnhancedPipelineSettings',
    'ResearchIntegrationMode',
    'create_pipeline_configuration_from_dict',
    'create_default_pipeline_configuration'
]
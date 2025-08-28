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

def create_pipeline_configuration_from_dict(config_dict: Dict[str, Any]) -> PipelineConfiguration:
    """
    Create a PipelineConfiguration from a dictionary configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        PipelineConfiguration instance
    """
    from musequill.v3.components.component_registry import create_enhanced_component_configurations

    from musequill.v3.components.base.component_interface import ComponentConfiguration, ComponentType
    
    # Get component configurations from registry
    component_configs = create_enhanced_component_configurations()
    
    # Extract orchestrator configuration
    orchestrator_config = component_configs.get('pipeline_orchestrator')
    if not orchestrator_config:
        # Create default orchestrator config
        from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestratorConfig
        
        orchestrator_specific_config = PipelineOrchestratorConfig(
            orchestration_strategy=OrchestrationStrategy.BALANCED,
            max_generation_attempts=config_dict.get('pipeline', {}).get('max_generation_attempts', 3),
            max_revision_cycles=config_dict.get('pipeline', {}).get('max_revision_cycles', 2),
            enable_adaptive_orchestration=True,
            pipeline_timeout_minutes=60
        )
        
        orchestrator_config = ComponentConfiguration(
            component_type=ComponentType.ORCHESTRATOR,
            component_name="Enhanced Pipeline Orchestrator",
            version="3.0.0",
            max_concurrent_executions=1,
            execution_timeout_seconds=3600,
            specific_config=orchestrator_specific_config
        )
    
    # Extract settings from config dict
    pipeline_settings = config_dict.get('pipeline', {})
    research_settings = config_dict.get('research', {})
    performance_settings = config_dict.get('performance', {})
    quality_settings = config_dict.get('quality', {})
    
    return PipelineConfiguration(
        component_configs=component_configs,
        orchestrator_config=orchestrator_config,
        
        # Pipeline behavior
        orchestration_strategy=pipeline_settings.get('orchestration_strategy', 'balanced'),
        max_generation_attempts=pipeline_settings.get('max_generation_attempts', 3),
        max_revision_cycles=pipeline_settings.get('max_revision_cycles', 2),
        enable_progressive_quality=pipeline_settings.get('enable_progressive_quality', True),
        
        # Research integration
        enable_research_integration=research_settings.get('enable_research_integration', True),
        research_triggers=research_settings.get('automatic_triggers', ['market_stale', 'quality_drop']),
        research_caching=research_settings.get('enable_caching', True),
        
        # Performance
        max_concurrent_components=performance_settings.get('max_concurrent_components', 3),
        component_timeout_seconds=performance_settings.get('component_timeout_seconds', 300),
        enable_checkpoints=performance_settings.get('enable_checkpoints', True),
        
        # Quality
        quality_threshold=quality_settings.get('minimum_threshold', 0.75),
        enable_quality_gates=quality_settings.get('enable_gates', True),
        
        # Execution metadata
        execution_metadata={
            'config_source': 'dictionary',
            'created_at': config_dict.get('timestamp', 'unknown')
        }
    )


def create_default_pipeline_configuration() -> PipelineConfiguration:
    """
    Create a default pipeline configuration with sensible defaults.
    
    Returns:
        Default PipelineConfiguration instance
    """
    return create_pipeline_configuration_from_dict({
        'pipeline': {
            'orchestration_strategy': 'balanced',
            'max_generation_attempts': 3,
            'max_revision_cycles': 2,
            'enable_progressive_quality': True
        },
        'research': {
            'enable_research_integration': True,
            'automatic_triggers': ['market_stale', 'quality_drop'],
            'enable_caching': True
        },
        'performance': {
            'max_concurrent_components': 3,
            'component_timeout_seconds': 300,
            'enable_checkpoints': True
        },
        'quality': {
            'minimum_threshold': 0.75,
            'enable_gates': True
        },
        'timestamp': 'default_configuration'
    })


# Export main classes and functions
__all__ = [
    'PipelineConfiguration',
    'EnhancedPipelineSettings',
    'ResearchIntegrationMode',
    'create_pipeline_configuration_from_dict',
    'create_default_pipeline_configuration'
]
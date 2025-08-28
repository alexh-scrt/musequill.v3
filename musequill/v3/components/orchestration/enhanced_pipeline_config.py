# Add this to the pipeline_orchestrator.py or create a new enhanced config

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import re
import os
from typing import List

from .pipeline_orchestrator import (
    PipelineOrchestratorConfig,
    OrchestrationStrategy
)

from musequill.v3.models.llm_discriminator_models import CritiqueDimension

class EnhancedPipelineOrchestratorConfig(PipelineOrchestratorConfig):
    """
    Enhanced Pipeline Orchestrator Configuration that includes research capabilities.
    Extends the base PipelineOrchestratorConfig with research-specific settings.
    """
    
    # Research integration settings
    enable_auto_research: bool = Field(
        default=True,
        description="Whether to enable automatic research triggers"
    )
    
    researcher_tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for research integration"
    )
    
    researcher_max_concurrent_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent research requests"
    )
    
    researcher_enable_caching: bool = Field(
        default=True,
        description="Whether to enable research result caching"
    )
    
    researcher_cache_ttl_hours: int = Field(
        default=12,
        ge=1,
        le=168,
        description="Cache time-to-live in hours"
    )
    
    researcher_request_timeout_seconds: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Request timeout for research operations"
    )
    
    researcher_max_results_per_query: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results per research query"
    )
    
    # Research trigger conditions
    plot_inconsistency_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Plot inconsistency threshold for auto research"
    )
    
    character_development_gaps_trigger: bool = Field(
        default=True,
        description="Whether to trigger research on character development gaps"
    )
    
    # LLM Discriminator settings
    enable_llm_discriminator: bool = Field(
        default=True,
        description="Whether to use LLM discriminator"
    )
    
    llm_discriminator_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight of LLM discriminator in final assessment"
    )
    
    llm_discriminator_model: str = Field(
        default="llama3.3:70b",
        description="LLM model for discriminator"
    )
    
    llm_discriminator_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM discriminator"
    )

    llm_discriminator_max_tokens: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Max tokens for LLM discriminator"
    )
    
    llm_discriminator_depth: str = Field(
        default="comprehensive",
        description="Depth of analysis for LLM discriminator"
    )

    llm_discriminator_focus_areas: Optional[List[str]] = Field(
        default=[
                CritiqueDimension.PLOT_COHERENCE.value,
                CritiqueDimension.CHARACTER_DEVELOPMENT.value,
                CritiqueDimension.PROSE_QUALITY.value,
                CritiqueDimension.PACING.value,
                CritiqueDimension.EMOTIONAL_RESONANCE.value,
                CritiqueDimension.MARKET_APPEAL.value
            ],
        description="Areas to focus critique on for LLM discriminator"
    )
    
    llm_discriminator_strictness: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="How strict the LLM discriminator should be (1.0 = very strict)"
    )

    llm_discriminator_include_suggestions: bool = Field(
        default=True,
        description="Whether to include specific improvement suggestions for LLM discriminator"
    )
    
    llm_discriminator_max_time: int = Field(
        default=120,
        ge=10,
        le=300,
        description="Maximum time to spend on LLM analysis"
    )

# Updated function to create the enhanced config
def resolve_environment_variables(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve environment variable references in configuration dictionary.
    Replaces ${VAR_NAME} patterns with os.getenv('VAR_NAME') values.
    
    Args:
        config_dict: Configuration dictionary that may contain env var references
        
    Returns:
        Configuration dictionary with resolved environment variables
    """
    
    def resolve_value(value: Any) -> Any:
        """Resolve environment variables in a single value."""
        if isinstance(value, str):
            # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
            env_var_pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else None
                
                # Get environment variable value
                env_value = os.getenv(var_name, default_value)
                
                if env_value is None:
                    print(f"Warning: Environment variable '{var_name}' not found and no default provided")
                    return match.group(0)  # Return original ${VAR_NAME} if not found
                
                return env_value
            
            # Replace all environment variable references
            resolved_value = re.sub(env_var_pattern, replace_env_var, value)
            return resolved_value
            
        elif isinstance(value, dict):
            # Recursively resolve dictionary values
            return {k: resolve_value(v) for k, v in value.items()}
            
        elif isinstance(value, list):
            # Recursively resolve list values
            return [resolve_value(item) for item in value]
            
        else:
            # Return other types unchanged
            return value
    
    return resolve_value(config_dict)


# Updated function with environment variable resolution
def create_enhanced_pipeline_configuration_from_dict(config_dict: Dict[str, Any]) -> EnhancedPipelineOrchestratorConfig:
    """
    Create EnhancedPipelineOrchestratorConfig from dictionary configuration.
    Automatically resolves environment variable references like ${VAR_NAME}.
    
    Args:
        config_dict: Configuration dictionary from YAML/JSON
        
    Returns:
        EnhancedPipelineOrchestratorConfig instance
    """
    try:
        # First, resolve all environment variable references
        resolved_config = resolve_environment_variables(config_dict)
        
        # Extract all the settings from resolved config
        pipeline_settings = resolved_config.get('pipeline', {})
        orchestration_settings = resolved_config.get('orchestration', {})
        components_settings = resolved_config.get('components', {})
        pipeline_orchestrator_settings = components_settings.get('pipeline_orchestrator', {})
        research_settings = resolved_config.get('research', {})
        researcher_settings = research_settings.get('researcher', {})
        error_handling_settings = resolved_config.get('error_handling', {})
        quality_settings = resolved_config.get('quality', {})
        llm_discriminator_settings = quality_settings.get('llm_discriminator', {})
        
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
        
        # Create EnhancedPipelineOrchestratorConfig with all resolved settings
        config = EnhancedPipelineOrchestratorConfig(
            # Base PipelineOrchestratorConfig settings
            max_generation_attempts=pipeline_settings.get('max_generation_attempts', 
                                   pipeline_orchestrator_settings.get('max_generation_attempts', 5)),
            max_revision_cycles=pipeline_settings.get('max_revision_cycles',
                              pipeline_orchestrator_settings.get('max_revision_cycles', 3)),
            orchestration_strategy=orchestration_strategy,
            parallel_variant_evaluation=orchestration_settings.get('enable_parallel_evaluation',
                                       pipeline_orchestrator_settings.get('parallel_variant_evaluation', True)),
            enable_market_intelligence_refresh=research_settings.get('enable_research_integration',
                                              pipeline_orchestrator_settings.get('enable_market_intelligence_refresh', True)),
            market_refresh_interval_hours=research_settings.get('market_refresh_interval_hours',
                                         pipeline_orchestrator_settings.get('market_refresh_interval_hours', 24)),
            component_health_check_interval=orchestration_settings.get('component_health_check_interval',
                                           pipeline_orchestrator_settings.get('component_health_check_interval', 300)),
            enable_adaptive_orchestration=orchestration_settings.get('enable_adaptive_orchestration',
                                         pipeline_orchestrator_settings.get('enable_adaptive_orchestration', True)),
            pipeline_timeout_minutes=orchestration_settings.get('pipeline_timeout_minutes',
                                    pipeline_orchestrator_settings.get('pipeline_timeout_minutes', 60)),
            enable_comprehensive_logging=orchestration_settings.get('enable_comprehensive_logging',
                                        pipeline_orchestrator_settings.get('enable_comprehensive_logging', True)),
            fallback_on_component_failure=error_handling_settings.get('fallback_strategies',
                                         pipeline_orchestrator_settings.get('fallback_on_component_failure', True)),
            
            # Enhanced settings for research (now with resolved env vars)
            enable_auto_research=research_settings.get('enable_research_integration', True),
            researcher_tavily_api_key=researcher_settings.get('tavily_api_key'),
            researcher_max_concurrent_requests=researcher_settings.get('max_concurrent_requests', 3),
            researcher_enable_caching=researcher_settings.get('enable_caching', True),
            researcher_cache_ttl_hours=researcher_settings.get('cache_ttl_hours', 12),
            researcher_request_timeout_seconds=researcher_settings.get('request_timeout_seconds', 30),
            researcher_max_results_per_query=researcher_settings.get('max_results_per_query', 10),
            
            # Research trigger conditions
            plot_inconsistency_threshold=research_settings.get('plot_inconsistency_threshold', 0.7),
            character_development_gaps_trigger=research_settings.get('character_development_gaps', True),
            
            # LLM Discriminator settings
            enable_llm_discriminator=quality_settings.get('enable_llm_discriminator', True),
            llm_discriminator_weight=llm_discriminator_settings.get('weight', 0.6),
            llm_discriminator_model=llm_discriminator_settings.get('model', 'llama3.3:70b'),
            llm_discriminator_temperature=llm_discriminator_settings.get('temperature', 0.2)
        )
        
        return config
        
    except Exception as e:
        # If anything fails, create a minimal valid configuration
        print(f"Warning: Error creating enhanced pipeline orchestrator configuration: {e}")
        print("Creating minimal default configuration...")
        
        return EnhancedPipelineOrchestratorConfig(
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
            fallback_on_component_failure=True,
            enable_auto_research=True,
            researcher_max_concurrent_requests=3,
            researcher_enable_caching=True,
            researcher_cache_ttl_hours=12,
            researcher_request_timeout_seconds=30,
            researcher_max_results_per_query=10,
            plot_inconsistency_threshold=0.7,
            character_development_gaps_trigger=True,
            enable_llm_discriminator=True,
            llm_discriminator_weight=0.6,
            llm_discriminator_model='llama3.3:70b',
            llm_discriminator_temperature=0.2
        )
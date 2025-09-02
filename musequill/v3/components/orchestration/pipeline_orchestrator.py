"""
Pipeline Orchestrator Component

Implements the main orchestration logic for the adversarial book generation system.
Manages the Generator-Discriminator loop, coordinates all components, handles
revision cycles, and provides comprehensive pipeline control.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from pydantic import Field, BaseModel
import logging
from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError,
    component_registry
)
from musequill.v3.models.chapter_objective import ChapterObjective
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.market_intelligence import MarketIntelligence
from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceEngineConfig
from musequill.v3.components.generators.chapter_generator import ChapterGeneratorInput, ChapterGeneratorOutput, ChapterGeneratorConfig
from musequill.v3.components.generators.plot_outliner import PlotOutlinerConfig, PlotOutlinerComponent
from musequill.v3.components.discriminators.plot_coherence_critic import PlotCoherenceCriticInput
from musequill.v3.components.discriminators.literary_quality_critic import LiteraryQualityCriticInput
from musequill.v3.components.discriminators.reader_engagement_critic import ReaderEngagementCriticInput
from musequill.v3.components.quality_control.comprehensive_quality_controller import (
    QualityControllerInput, ComprehensiveQualityAssessment, QualityDecision
)
from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceEngineInput
from musequill.v3.components.discriminators.llm_discriminator import LLMDiscriminator, LLMDiscriminatorConfig
from musequill.v3.models.llm_discriminator_models import (
    CritiqueDimension
)

from musequill.v3.components.character_developer.character_development_config import (
    create_character_development_config_from_dict
)

logger = logging.getLogger(__name__)

class PipelineState(str, Enum):
    """States of the pipeline execution."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    REVISING = "revising"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class OrchestrationStrategy(str, Enum):
    """Different orchestration strategies."""
    QUALITY_FIRST = "quality_first"
    SPEED_OPTIMIZED = "speed_optimized"
    BALANCED = "balanced"
    EXPERIMENTAL = "experimental"


class PipelineOrchestratorConfig(BaseModel):
    """Configuration for Pipeline Orchestrator."""
    
    max_generation_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum generation attempts before giving up"
    )
    
    max_revision_cycles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum revision cycles per chapter"
    )
    
    orchestration_strategy: OrchestrationStrategy = Field(
        default=OrchestrationStrategy.BALANCED,
        description="Pipeline orchestration strategy"
    )
    
    parallel_variant_evaluation: bool = Field(
        default=True,
        description="Whether to evaluate variants in parallel"
    )
    
    enable_market_intelligence_refresh: bool = Field(
        default=True,
        description="Whether to refresh market intelligence periodically"
    )
    
    market_refresh_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours between market intelligence refreshes"
    )
    
    component_health_check_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Seconds between component health checks"
    )
    
    enable_adaptive_orchestration: bool = Field(
        default=True,
        description="Whether to adapt orchestration based on performance"
    )
    
    pipeline_timeout_minutes: int = Field(
        default=60,
        ge=10,
        le=480,
        description="Maximum time for complete pipeline execution"
    )
    
    enable_comprehensive_logging: bool = Field(
        default=True,
        description="Whether to log detailed pipeline execution data"
    )
    
    fallback_on_component_failure: bool = Field(
        default=True,
        description="Whether to continue with reduced functionality on component failures"
    )


class PipelineOrchestratorInput(BaseModel):
    """Input data for Pipeline Orchestrator."""
    
    chapter_objective: ChapterObjective = Field(
        description="Chapter generation objectives"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current dynamic story state"
    )
    
    target_genre: str = Field(
        description="Target genre for market intelligence"
    )
    
    force_market_refresh: bool = Field(
        default=False,
        description="Force refresh of market intelligence"
    )
    
    override_quality_thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional quality threshold overrides"
    )
    
    execution_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution context and metadata"
    )


class PipelineExecutionResult(BaseModel):
    """Result of complete pipeline execution."""
    
    execution_id: str = Field(
        description="Unique identifier for this execution"
    )
    
    final_chapter_variant: Optional[ChapterVariant] = Field(
        default=None,
        description="Final accepted chapter variant"
    )
    
    execution_success: bool = Field(
        description="Whether execution completed successfully"
    )
    
    total_execution_time_seconds: float = Field(
        description="Total time for complete execution"
    )
    
    generation_attempts: int = Field(
        description="Number of generation attempts made"
    )
    
    revision_cycles: int = Field(
        description="Number of revision cycles completed"
    )
    
    final_quality_assessment: Optional[ComprehensiveQualityAssessment] = Field(
        default=None,
        description="Final comprehensive quality assessment"
    )
    
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed execution metadata and statistics"
    )
    
    component_performance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance metrics for each component"
    )
    
    market_intelligence_used: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence data used in execution"
    )
    
    error_details: Optional[str] = Field(
        default=None,
        description="Error details if execution failed"
    )


class PipelineOrchestrator(BaseComponent[PipelineOrchestratorInput, PipelineExecutionResult, PipelineOrchestratorConfig]):
    """
    Pipeline Orchestrator managing the complete adversarial generation system.
    
    Coordinates Generator, Critics, Quality Controller, and Market Intelligence
    components in an adversarial loop to produce high-quality chapter content.
    """
    
    def __init__(self, config: ComponentConfiguration[PipelineOrchestratorConfig]):
        super().__init__(config)
        
        # Component references
        self._chapter_generator = None
        self._plot_outliner = None
        self._plot_coherence_critic = None
        self._literary_quality_critic = None
        self._reader_engagement_critic = None
        self._quality_controller = None
        self._market_intelligence_engine = None
        self._llm_discriminator = None
        self._character_developer = None

        # Pipeline state
        self.pipeline_state = PipelineState.IDLE
        self._current_execution_id: Optional[str] = None
        self._execution_history: List[Dict[str, Any]] = []
        
        # Market intelligence cache
        self._market_intelligence_cache: Dict[str, Tuple[MarketIntelligence, datetime]] = {}
        
        # Performance tracking
        self._component_performance_tracking: Dict[str, List[float]] = {}
        self._orchestration_metrics: Dict[str, Any] = {}
        
        # Health monitoring
        self._last_health_check: datetime = datetime.now()
        self._component_health_status: Dict[str, bool] = {}

        self.components: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize pipeline orchestrator and all managed components."""
        try:
            self.pipeline_state = PipelineState.INITIALIZING
            
            # Initialize component registry if needed
            await self._ensure_component_registry()
            
            # Initialize all managed components
            initialization_success = await self._initialize_all_components()
            if not initialization_success:
                self.pipeline_state = PipelineState.ERROR
                return False
            
            # Perform initial component health check
            await self._perform_component_health_checks()
            
            # Initialize orchestration metrics
            self._orchestration_metrics = {
                'total_executions': 0,
                'successful_executions': 0,
                'average_execution_time': 0.0,
                'average_generation_attempts': 0.0,
                'average_revision_cycles': 0.0,
                'component_failure_counts': {},
                'quality_score_trends': []
            }
            
            # Start background monitoring if enabled
            if self.config.specific_config.component_health_check_interval > 0:
                asyncio.create_task(self._background_health_monitoring())
            
            self.pipeline_state = PipelineState.IDLE
            return True
            
        except Exception as e:
            self.state.last_error = f"Pipeline orchestrator initialization failed: {str(e)}"
            self.pipeline_state = PipelineState.ERROR
            return False
    
    async def process(self, input_data: PipelineOrchestratorInput) -> PipelineExecutionResult:
        """
        Execute complete adversarial generation pipeline.
        
        Args:
            input_data: Chapter objectives, story state, and execution parameters
            
        Returns:
            Complete pipeline execution result with final chapter or error details
        """
        execution_id = str(uuid.uuid4())
        self._current_execution_id = execution_id
        start_time = datetime.now()
        
        try:
            self.pipeline_state = PipelineState.GENERATING
            
            # Initialize execution tracking
            execution_metadata = {
                'execution_id': execution_id,
                'start_time': start_time,
                'chapter_number': input_data.chapter_objective.chapter_number,
                'strategy': self.config.specific_config.orchestration_strategy.value,
                'component_versions': await self._get_component_versions()
            }
            
            # Refresh market intelligence if needed
            market_intelligence = await self._get_market_intelligence(
                input_data.target_genre,
                input_data.force_market_refresh
            )
            
            # Execute adversarial generation loop
            final_result = await self._execute_adversarial_loop(
                input_data,
                market_intelligence,
                execution_metadata
            )
            
            # Update orchestration metrics
            await self._update_orchestration_metrics(final_result)
            
            return final_result
            
        except Exception as e:
            error_result = PipelineExecutionResult(
                execution_id=execution_id,
                execution_success=False,
                total_execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                generation_attempts=0,
                revision_cycles=0,
                error_details=f"Pipeline execution failed: {str(e)}",
                execution_metadata=execution_metadata if 'execution_metadata' in locals() else {}
            )
            
            self.pipeline_state = PipelineState.ERROR
            return error_result
            
        finally:
            self._current_execution_id = None
            if self.pipeline_state != PipelineState.ERROR:
                self.pipeline_state = PipelineState.IDLE
    
    async def health_check(self) -> bool:
        """Perform comprehensive health check on entire pipeline."""
        try:
            # Check orchestrator state
            if self.pipeline_state == PipelineState.ERROR:
                return False
            
            # Check all managed components
            component_health = await self._perform_component_health_checks()
            if not all(component_health.values()):
                return False
            
            # Check recent execution success rates
            if len(self._execution_history) > 5:
                recent_successes = sum(1 for exec_record in self._execution_history[-10:] 
                                     if exec_record.get('success', False))
                success_rate = recent_successes / min(10, len(self._execution_history))
                if success_rate < 0.3:  # Less than 30% success rate
                    return False
            
            # Check component performance trends
            if self._orchestration_metrics.get('component_failure_counts'):
                total_failures = sum(self._orchestration_metrics['component_failure_counts'].values())
                if total_failures > 20:  # Too many component failures
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup orchestrator and all managed components."""
        try:
            self.pipeline_state = PipelineState.IDLE
            
            # Cleanup all managed components
            cleanup_tasks = []
            
            for component in [self._chapter_generator, self._plot_coherence_critic,
                            self._literary_quality_critic, self._reader_engagement_critic,
                            self._quality_controller, self._market_intelligence_engine]:
                if component:
                    cleanup_tasks.append(component.cleanup())
            
            if cleanup_tasks:
                cleanup_results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
                # Log any cleanup failures
                for i, result in enumerate(cleanup_results):
                    if isinstance(result, Exception):
                        self.state.error_count += 1
            
            # Clear caches and state
            self._market_intelligence_cache.clear()
            self._component_performance_tracking.clear()
            
            # Preserve execution history for analysis
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-50:]
            
            return True
            
        except Exception:
            return False
    
    async def _ensure_component_registry(self) -> None:
        """Ensure component registry has all required component types registered."""
        # In real implementation, would register all component classes
        # This is a placeholder showing the pattern
        pass
    
    def _create_llm_discriminator_config(self) -> LLMDiscriminatorConfig:
        """Create LLM discriminator configuration from pipeline config."""
        llm_settings = self.config.get('llm_discriminator', {})
        
        return LLMDiscriminatorConfig(
            llm_model_name=llm_settings.get('model', "llama3.3:70b"),
            analysis_temperature=llm_settings.get('temperature', 0.2),
            max_analysis_tokens=llm_settings.get('max_tokens', 2000),
            critique_depth=llm_settings.get('depth', "comprehensive"),
            focus_areas=llm_settings.get('focus_areas', [
                CritiqueDimension.PLOT_COHERENCE.value,
                CritiqueDimension.CHARACTER_DEVELOPMENT.value,
                CritiqueDimension.PROSE_QUALITY.value,
                CritiqueDimension.PACING.value,
                CritiqueDimension.EMOTIONAL_RESONANCE.value,
                CritiqueDimension.MARKET_APPEAL.value
            ]),
            scoring_strictness=llm_settings.get('strictness', 0.75),
            include_suggestions=llm_settings.get('include_suggestions', True),
            max_analysis_time_seconds=llm_settings.get('max_time', 90)
        )

    async def _initialize_all_components(self) -> bool:
        """Initialize all pipeline components."""
        try:
            # Create component configurations
            # In real implementation, these would be loaded from configuration
            component_configs = await self._create_component_configurations()
            
            # Initialize Chapter Generator
            if 'chapter_generator' in component_configs:
                generator_id = component_registry.create_component(
                    'chapter_generator', 
                    component_configs['chapter_generator']
                )
                self._chapter_generator = component_registry.get_component(generator_id)
                await self._chapter_generator.initialize()
                await self._chapter_generator.start()
            
            # Initialize Plot Outliner
            if 'plot_outliner' in component_configs:
                outliner_id = component_registry.create_component(
                    'plot_outliner', 
                    component_configs['plot_outliner']
                )
                self._plot_outliner = component_registry.get_component(outliner_id)
                await self._plot_outliner.initialize()
                await self._plot_outliner.start()


            # Initialize Critics
            if 'plot_coherence_critic' in component_configs:
                critic_id = component_registry.create_component(
                    'plot_coherence_critic',
                    component_configs['plot_coherence_critic']
                )
                self._plot_coherence_critic = component_registry.get_component(critic_id)
                await self._plot_coherence_critic.initialize()
                await self._plot_coherence_critic.start()
            
            if 'literary_quality_critic' in component_configs:
                critic_id = component_registry.create_component(
                    'literary_quality_critic',
                    component_configs['literary_quality_critic']
                )
                self._literary_quality_critic = component_registry.get_component(critic_id)
                await self._literary_quality_critic.initialize()
                await self._literary_quality_critic.start()
            
            if 'reader_engagement_critic' in component_configs:
                critic_id = component_registry.create_component(
                    'reader_engagement_critic',
                    component_configs['reader_engagement_critic']
                )
                self._reader_engagement_critic = component_registry.get_component(critic_id)
                await self._reader_engagement_critic.initialize()
                await self._reader_engagement_critic.start()
            
            # Initialize Quality Controller
            if 'quality_controller' in component_configs:
                controller_id = component_registry.create_component(
                    'quality_controller',
                    component_configs['quality_controller']
                )
                self._quality_controller = component_registry.get_component(controller_id)
                await self._quality_controller.initialize()
                await self._quality_controller.start()
            
            # Initialize Market Intelligence Engine
            if 'market_intelligence' in component_configs:
                mi_id = component_registry.create_component(
                    'market_intelligence',
                    component_configs['market_intelligence']
                )
                self._market_intelligence_engine = component_registry.get_component(mi_id)
                await self._market_intelligence_engine.initialize()
                await self._market_intelligence_engine.start()

            if 'character_developer' in component_configs:
                developer_id = component_registry.create_component(
                    'character_developer',
                    component_configs['character_developer']
                )
                self._character_developer = component_registry.get_component(developer_id)
                await self._character_developer.initialize()
                await self._character_developer.start()
                logger.info("✅ Character Developer initialized successfully")

            if 'llm_discriminator' in component_configs:
                llm_config = self._create_llm_discriminator_config()
                component_config = ComponentConfiguration(
                    component_type=ComponentType.DISCRIMINATOR,
                    component_name="LLM Literary Discriminator",
                    version="1.0.0",
                    max_concurrent_executions=1,
                    execution_timeout_seconds=120,
                    specific_config=llm_config
                )
                
                self._llm_discriminator = LLMDiscriminator(component_config)
                if await self._llm_discriminator.initialize():
                    logger.info("✅ LLM Discriminator initialized successfully")
                else:
                    logger.error("❌ Failed to initialize LLM Discriminator")
                    self._llm_discriminator = None

            return True
            
        except Exception as e:
            self.state.last_error = f"Component initialization failed: {str(e)}"
            return False
    
    async def _create_component_configurations(self) -> Dict[str, ComponentConfiguration]:
        """Create configurations for all pipeline components."""
        # Placeholder - in real implementation would load from config files
        config = self.config.specific_config.components

        character_dev_config = create_character_development_config_from_dict(
            config.get('character_developer', {})
        )


        return {
            'character_developer': ComponentConfiguration(
                component_type=ComponentType.GENERATOR,  # Character development is generative
                component_name="Character Development Component",
                version="1.0.0",
                max_concurrent_executions=1,
                execution_timeout_seconds=character_dev_config.max_processing_time_seconds,
                auto_recycle_after_uses=100,
                recycle_on_error_count=3,
                specific_config=character_dev_config
            ),
            'plot_outliner': ComponentConfiguration(
                component_type=ComponentType.GENERATOR,
                component_name="Plot Outliner",
                specific_config=PlotOutlinerConfig(**config['generators']['plot_outliner'])  # Assuming PlotOutlinerConfig is defined config['generators']['plot_outliner']  # Would contain actual config
            ),
            'chapter_generator': ComponentConfiguration(
                component_type=ComponentType.GENERATOR,
                component_name="Chapter Generator",
                specific_config=ChapterGeneratorConfig(**config['generators']['chapter_generator'])  # Assuming ChapterGeneratorConfig is defined config['generators']['chapter_generator']
            ),
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
            'llm_discriminator': ComponentConfiguration(
                component_type=ComponentType.DISCRIMINATOR,
                component_name="LLM Critic",
                specific_config={}
            ),
            'quality_controller': ComponentConfiguration(
                component_type=ComponentType.QUALITY_CONTROLLER,
                component_name="Comprehensive Quality Controller",
                specific_config={}
            ),
            'market_intelligence': ComponentConfiguration(
                component_type=ComponentType.MARKET_INTELLIGENCE,
                component_name="Market Intelligence Engine",
                specific_config=MarketIntelligenceEngineConfig(
                    **config['market_intelligence']
                )  
            )
        }
    
    async def _perform_component_health_checks(self) -> Dict[str, bool]:
        """Perform health checks on all components."""
        health_status = {}
        
        components = [
            ('chapter_generator', self._chapter_generator),
            ('plot_outliner', self._plot_outliner),
            ('plot_coherence_critic', self._plot_coherence_critic),
            ('literary_quality_critic', self._literary_quality_critic),
            ('reader_engagement_critic', self._reader_engagement_critic),
            ('quality_controller', self._quality_controller),
            ('market_intelligence_engine', self._market_intelligence_engine)
        ]
        
        for component_name, component in components:
            if component:
                try:
                    health_status[component_name] = await component.health_check()
                except Exception:
                    health_status[component_name] = False
            else:
                health_status[component_name] = False
        
        self._component_health_status = health_status
        self._last_health_check = datetime.now()
        return health_status
    
    async def _get_market_intelligence(self, genre: str, force_refresh: bool = False) -> Optional[MarketIntelligence]:
        """Get market intelligence with caching."""
        if not self._market_intelligence_engine:
            return None
        
        cache_key = genre.lower()
        
        # Check cache validity
        if cache_key in self._market_intelligence_cache and not force_refresh:
            market_data, cache_time = self._market_intelligence_cache[cache_key]
            cache_age = datetime.now() - cache_time
            
            if cache_age.total_seconds() < self.config.specific_config.market_refresh_interval_hours * 3600:
                return market_data
        
        # Refresh market intelligence
        try:
            mi_input = MarketIntelligenceEngineInput(
                target_genre=genre,
                force_refresh=force_refresh
            )
            
            market_intelligence = await self._market_intelligence_engine.invoke(mi_input)
            
            # Cache the result
            self._market_intelligence_cache[cache_key] = (market_intelligence, datetime.now())
            
            return market_intelligence
            
        except Exception as e:
            self.state.error_count += 1
            if self.config.specific_config.fallback_on_component_failure:
                return None  # Continue without market intelligence
            else:
                raise ComponentError(f"Market intelligence retrieval failed: {str(e)}")
    
    async def _execute_adversarial_loop(self, input_data: PipelineOrchestratorInput,
                                       market_intelligence: Optional[MarketIntelligence],
                                       execution_metadata: Dict[str, Any]) -> PipelineExecutionResult:
        """Execute the main adversarial generation loop."""
        
        generation_attempts = 0
        revision_cycles = 0
        best_variant = None
        best_assessment = None
        
        # Track all attempts for analysis
        attempt_history = []
        
        # Main generation loop
        while generation_attempts < self.config.specific_config.max_generation_attempts:
            generation_attempts += 1
            attempt_start = datetime.now()
            
            try:
                # GENERATION PHASE
                self.pipeline_state = PipelineState.GENERATING
                
                generation_result = await self._execute_generation_phase(
                    input_data,
                    market_intelligence,
                    revision_cycles,
                    attempt_history
                )
                
                if not generation_result.chapter_variants:
                    continue  # Try again
                
                # EVALUATION PHASE  
                self.pipeline_state = PipelineState.EVALUATING
                
                evaluation_results = await self._execute_evaluation_phase(
                    generation_result.chapter_variants,
                    input_data.story_state,
                    market_intelligence,
                    input_data.chapter_objective.chapter_number / (input_data.story_state.total_planned_chapters or 20)
                )
                
                # QUALITY CONTROL PHASE
                best_variant_this_attempt, quality_assessment = await self._select_best_variant(
                    generation_result.chapter_variants,
                    evaluation_results,
                    input_data.override_quality_thresholds
                )
                
                # Record attempt
                attempt_record = {
                    'attempt': generation_attempts,
                    'revision_cycle': revision_cycles,
                    'generation_time': (datetime.now() - attempt_start).total_seconds(),
                    'variants_generated': len(generation_result.chapter_variants),
                    'quality_score': quality_assessment.overall_quality_score,
                    'decision': quality_assessment.decision.value
                }
                attempt_history.append(attempt_record)
                
                # Check quality decision
                if quality_assessment.decision == QualityDecision.ACCEPT:
                    # SUCCESS - Chapter accepted
                    return await self._create_successful_result(
                        input_data.execution_context.get('execution_id', str(uuid.uuid4())),
                        best_variant_this_attempt,
                        quality_assessment,
                        generation_attempts,
                        revision_cycles,
                        execution_metadata,
                        attempt_history,
                        market_intelligence,
                        datetime.now() - execution_metadata['start_time']
                    )
                
                elif quality_assessment.decision == QualityDecision.REVISE:
                    # REVISION PHASE
                    if revision_cycles < self.config.specific_config.max_revision_cycles:
                        revision_cycles += 1
                        self.pipeline_state = PipelineState.REVISING
                        
                        # Store best attempt so far
                        if (best_assessment is None or 
                            quality_assessment.overall_quality_score > best_assessment.overall_quality_score):
                            best_variant = best_variant_this_attempt
                            best_assessment = quality_assessment
                        
                        # Apply revision guidance for next attempt
                        await self._apply_revision_guidance(
                            input_data,
                            quality_assessment,
                            attempt_history
                        )
                        
                        continue  # Try again with revisions
                    else:
                        # Max revisions reached - use best attempt
                        break
                
                else:  # QualityDecision.REJECT
                    # Store if this is better than previous attempts
                    if (best_assessment is None or 
                        quality_assessment.overall_quality_score > best_assessment.overall_quality_score):
                        best_variant = best_variant_this_attempt
                        best_assessment = quality_assessment
                    
                    continue  # Try again
                
            except Exception as e:
                # Log attempt failure and continue
                attempt_record = {
                    'attempt': generation_attempts,
                    'revision_cycle': revision_cycles,
                    'generation_time': (datetime.now() - attempt_start).total_seconds(),
                    'error': str(e),
                    'decision': 'error'
                }
                attempt_history.append(attempt_record)
                
                if not self.config.specific_config.fallback_on_component_failure:
                    raise
        
        # All attempts exhausted - return best result or failure
        if best_variant and best_assessment:
            return await self._create_partial_success_result(
                input_data.execution_context.get('execution_id', str(uuid.uuid4())),
                best_variant,
                best_assessment,
                generation_attempts,
                revision_cycles,
                execution_metadata,
                attempt_history,
                market_intelligence,
                datetime.now() - execution_metadata['start_time']
            )
        else:
            return await self._create_failure_result(
                input_data.execution_context.get('execution_id', str(uuid.uuid4())),
                generation_attempts,
                revision_cycles,
                execution_metadata,
                attempt_history,
                datetime.now() - execution_metadata['start_time'],
                "All generation attempts failed to produce acceptable content"
            )
    
    async def _execute_generation_phase(self, input_data: PipelineOrchestratorInput,
                                       market_intelligence: Optional[MarketIntelligence],
                                       revision_cycle: int,
                                       attempt_history: List[Dict[str, Any]]) -> ChapterGeneratorOutput:
        """Execute chapter generation phase."""
        
        if not self._chapter_generator:
            raise ComponentError("Chapter generator not available")
        
        # Prepare generation input
        generator_input = ChapterGeneratorInput(
            chapter_objective=input_data.chapter_objective,
            story_state=input_data.story_state,
            market_intelligence=market_intelligence,
            critic_feedback_history=[
                attempt for attempt in attempt_history 
                if 'quality_assessment' in attempt
            ],
            generation_constraints={
                'revision_cycle': revision_cycle,
                'previous_attempts': len(attempt_history)
            }
        )
        
        # Execute generation
        return await self._chapter_generator.invoke(generator_input)
    
    async def _execute_evaluation_phase(self, chapter_variants: List[ChapterVariant],
                                       story_state: DynamicStoryState,
                                       market_intelligence: Optional[MarketIntelligence],
                                       story_position: float) -> Dict[str, List[Any]]:
        """Execute critic evaluation phase for all variants."""
        
        evaluation_tasks = []
        
        for variant in chapter_variants:
            variant_evaluations = []
            
            # Plot Coherence Critic
            if self._plot_coherence_critic:
                plot_input = PlotCoherenceCriticInput(
                    chapter_variant=variant,
                    story_state=story_state
                )
                variant_evaluations.append(
                    ('plot_coherence', self._plot_coherence_critic.invoke(plot_input))
                )
            
            # Literary Quality Critic
            if self._literary_quality_critic:
                literary_input = LiteraryQualityCriticInput(
                    chapter_variant=variant,
                    story_state=story_state
                )
                variant_evaluations.append(
                    ('literary_quality', self._literary_quality_critic.invoke(literary_input))
                )
            
            # Reader Engagement Critic
            if self._reader_engagement_critic:
                engagement_input = ReaderEngagementCriticInput(
                    chapter_variant=variant,
                    story_state=story_state,
                    market_intelligence=market_intelligence
                )
                variant_evaluations.append(
                    ('reader_engagement', self._reader_engagement_critic.invoke(engagement_input))
                )
            
            evaluation_tasks.append((variant.variant_id, variant_evaluations))
        
        # Execute evaluations
        if self.config.specific_config.parallel_variant_evaluation:
            # Execute all evaluations in parallel
            all_tasks = []
            task_mapping = {}
            
            for variant_id, variant_evals in evaluation_tasks:
                for eval_type, eval_task in variant_evals:
                    task_id = f"{variant_id}_{eval_type}"
                    all_tasks.append(eval_task)
                    task_mapping[len(all_tasks) - 1] = (variant_id, eval_type)
            
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Organize results by variant
            evaluation_results = {}
            for i, result in enumerate(results):
                variant_id, eval_type = task_mapping[i]
                
                if variant_id not in evaluation_results:
                    evaluation_results[variant_id] = {}
                
                if isinstance(result, Exception):
                    evaluation_results[variant_id][eval_type] = None
                    self.state.error_count += 1
                else:
                    evaluation_results[variant_id][eval_type] = result
            
        else:
            # Execute evaluations sequentially
            evaluation_results = {}
            
            for variant_id, variant_evals in evaluation_tasks:
                evaluation_results[variant_id] = {}
                
                for eval_type, eval_task in variant_evals:
                    try:
                        result = await eval_task
                        evaluation_results[variant_id][eval_type] = result
                    except Exception as e:
                        evaluation_results[variant_id][eval_type] = None
                        self.state.error_count += 1
        
        return evaluation_results
    
    async def _select_best_variant(self, chapter_variants: List[ChapterVariant],
                                  evaluation_results: Dict[str, Dict[str, Any]],
                                  quality_overrides: Optional[Dict[str, float]]) -> Tuple[ChapterVariant, ComprehensiveQualityAssessment]:
        """Select best variant using quality controller."""
        
        if not self._quality_controller:
            # Fallback selection without quality controller
            return chapter_variants[0], None
        
        best_variant = None
        best_assessment = None
        best_score = -1.0
        
        for variant in chapter_variants:
            variant_evals = evaluation_results.get(variant.variant_id, {})
            
            # Skip variants with missing critical evaluations
            if (not variant_evals.get('plot_coherence') or
                not variant_evals.get('literary_quality') or 
                not variant_evals.get('reader_engagement')):
                continue
            
            # Create quality controller input
            quality_input = QualityControllerInput(
                chapter_variant=variant,
                plot_coherence_assessment=variant_evals['plot_coherence'],
                literary_quality_assessment=variant_evals['literary_quality'],
                reader_engagement_assessment=variant_evals['reader_engagement'],
                story_position=0.5,  # Would calculate from story state
                revision_cycle=0
            )
            
            # Get quality assessment
            try:
                quality_assessment = await self._quality_controller.invoke(quality_input)
                
                # Track best variant
                if quality_assessment.overall_quality_score > best_score:
                    best_variant = variant
                    best_assessment = quality_assessment
                    best_score = quality_assessment.overall_quality_score
                    
            except Exception as e:
                self.state.error_count += 1
                continue
        
        if best_variant and best_assessment:
            return best_variant, best_assessment
        else:
            # Fallback if all quality assessments failed
            return chapter_variants[0], ComprehensiveQualityAssessment(
                overall_quality_score=0.5,
                plot_coherence_score=0.5,
                literary_quality_score=0.5,
                reader_engagement_score=0.5,
                market_alignment_score=0.5,
                decision=QualityDecision.REJECT,
                decision_rationale="Quality assessment failed - using fallback",
                adaptive_threshold_applied=0.75
            )
    
    async def _apply_revision_guidance(self, input_data: PipelineOrchestratorInput,
                                      quality_assessment: ComprehensiveQualityAssessment,
                                      attempt_history: List[Dict[str, Any]]) -> None:
        """Apply revision guidance to improve next generation attempt."""
        
        # Store revision guidance in input context for generator
        revision_context = {
            'improvement_priorities': quality_assessment.improvement_priorities,
            'critical_issues': quality_assessment.critical_issues,
            'revision_guidance': quality_assessment.revision_guidance,
            'strengths_to_preserve': quality_assessment.strengths_identified
        }
        
        input_data.execution_context['revision_context'] = revision_context
        
        # Adjust generation constraints based on feedback
        if 'generation_constraints' not in input_data.execution_context:
            input_data.execution_context['generation_constraints'] = {}
        
        constraints = input_data.execution_context['generation_constraints']
        
        # Focus on most critical issues
        if quality_assessment.critical_issues:
            constraints['focus_areas'] = quality_assessment.critical_issues[:2]
        
        # Adjust creativity temperature based on feedback patterns
        if len(attempt_history) > 2:
            recent_scores = [attempt.get('quality_score', 0.5) for attempt in attempt_history[-3:]]
            if all(score < 0.6 for score in recent_scores):
                constraints['increase_creativity'] = True
    
    # Result creation methods
    
    async def _create_successful_result(self, execution_id: str, final_variant: ChapterVariant,
                                       quality_assessment: ComprehensiveQualityAssessment,
                                       generation_attempts: int, revision_cycles: int,
                                       execution_metadata: Dict[str, Any],
                                       attempt_history: List[Dict[str, Any]],
                                       market_intelligence: Optional[MarketIntelligence],
                                       total_time: timedelta) -> PipelineExecutionResult:
        """Create successful execution result."""
        
        component_performance = await self._collect_component_performance()
        
        execution_metadata.update({
            'final_quality_score': quality_assessment.overall_quality_score,
            'decision': quality_assessment.decision.value,
            'attempt_history': attempt_history,
            'success': True
        })
        
        return PipelineExecutionResult(
            execution_id=execution_id,
            final_chapter_variant=final_variant,
            execution_success=True,
            total_execution_time_seconds=total_time.total_seconds(),
            generation_attempts=generation_attempts,
            revision_cycles=revision_cycles,
            final_quality_assessment=quality_assessment,
            execution_metadata=execution_metadata,
            component_performance=component_performance,
            market_intelligence_used=market_intelligence
        )
    
    async def _create_partial_success_result(self, execution_id: str, best_variant: ChapterVariant,
                                           best_assessment: ComprehensiveQualityAssessment,
                                           generation_attempts: int, revision_cycles: int,
                                           execution_metadata: Dict[str, Any],
                                           attempt_history: List[Dict[str, Any]],
                                           market_intelligence: Optional[MarketIntelligence],
                                           total_time: timedelta) -> PipelineExecutionResult:
        """Create partial success result (best effort)."""
        
        component_performance = await self._collect_component_performance()
        
        execution_metadata.update({
            'final_quality_score': best_assessment.overall_quality_score,
            'decision': best_assessment.decision.value,
            'attempt_history': attempt_history,
            'success': False,
            'partial_success': True
        })
        
        return PipelineExecutionResult(
            execution_id=execution_id,
            final_chapter_variant=best_variant,
            execution_success=False,
            total_execution_time_seconds=total_time.total_seconds(),
            generation_attempts=generation_attempts,
            revision_cycles=revision_cycles,
            final_quality_assessment=best_assessment,
            execution_metadata=execution_metadata,
            component_performance=component_performance,
            market_intelligence_used=market_intelligence,
            error_details=f"Failed to meet acceptance criteria after {generation_attempts} attempts"
        )
    
    async def _create_failure_result(self, execution_id: str, generation_attempts: int,
                                   revision_cycles: int, execution_metadata: Dict[str, Any],
                                   attempt_history: List[Dict[str, Any]],
                                   total_time: timedelta, error_message: str) -> PipelineExecutionResult:
        """Create failure result."""
        
        component_performance = await self._collect_component_performance()
        
        execution_metadata.update({
            'attempt_history': attempt_history,
            'success': False,
            'failure_reason': error_message
        })
        
        return PipelineExecutionResult(
            execution_id=execution_id,
            execution_success=False,
            total_execution_time_seconds=total_time.total_seconds(),
            generation_attempts=generation_attempts,
            revision_cycles=revision_cycles,
            execution_metadata=execution_metadata,
            component_performance=component_performance,
            error_details=error_message
        )
    
    # Utility and monitoring methods
    
    async def _collect_component_performance(self) -> Dict[str, Dict[str, Any]]:
        """Collect performance metrics from all components."""
        performance = {}
        
        components = [
            ('chapter_generator', self._chapter_generator),
            ('plot_coherence_critic', self._plot_coherence_critic),
            ('literary_quality_critic', self._literary_quality_critic),
            ('reader_engagement_critic', self._reader_engagement_critic),
            ('quality_controller', self._quality_controller),
            ('market_intelligence_engine', self._market_intelligence_engine)
        ]
        
        for component_name, component in components:
            if component:
                try:
                    performance[component_name] = component.get_status_summary()
                except Exception:
                    performance[component_name] = {'status': 'error'}
        
        return performance
    
    async def _get_component_versions(self) -> Dict[str, str]:
        """Get version information for all components."""
        # Placeholder - would collect actual version info
        return {
            'orchestrator': '1.0.0',
            'chapter_generator': '1.0.0',
            'critics': '1.0.0',
            'quality_controller': '1.0.0'
        }
    
    async def _update_orchestration_metrics(self, result: PipelineExecutionResult) -> None:
        """Update orchestration performance metrics."""
        self._orchestration_metrics['total_executions'] += 1
        
        if result.execution_success:
            self._orchestration_metrics['successful_executions'] += 1
        
        # Update averages
        total = self._orchestration_metrics['total_executions']
        self._orchestration_metrics['average_execution_time'] = (
            (self._orchestration_metrics['average_execution_time'] * (total - 1) + 
             result.total_execution_time_seconds) / total
        )
        
        self._orchestration_metrics['average_generation_attempts'] = (
            (self._orchestration_metrics['average_generation_attempts'] * (total - 1) + 
             result.generation_attempts) / total
        )
        
        self._orchestration_metrics['average_revision_cycles'] = (
            (self._orchestration_metrics['average_revision_cycles'] * (total - 1) + 
             result.revision_cycles) / total
        )
        
        # Track quality scores
        if result.final_quality_assessment:
            self._orchestration_metrics['quality_score_trends'].append(
                result.final_quality_assessment.overall_quality_score
            )
            
            # Keep recent trend data
            if len(self._orchestration_metrics['quality_score_trends']) > 100:
                self._orchestration_metrics['quality_score_trends'] = (
                    self._orchestration_metrics['quality_score_trends'][-50:]
                )
        
        # Add to execution history
        self._execution_history.append({
            'execution_id': result.execution_id,
            'timestamp': datetime.now(),
            'success': result.execution_success,
            'quality_score': result.final_quality_assessment.overall_quality_score if result.final_quality_assessment else 0.0,
            'execution_time': result.total_execution_time_seconds,
            'generation_attempts': result.generation_attempts,
            'revision_cycles': result.revision_cycles
        })
    
    async def _background_health_monitoring(self) -> None:
        """Background task for continuous component health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.config.specific_config.component_health_check_interval)
                
                if self.pipeline_state not in [PipelineState.ERROR, PipelineState.PAUSED]:
                    await self._perform_component_health_checks()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if health check fails
                continue
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            'pipeline_state': self.pipeline_state.value,
            'current_execution_id': self._current_execution_id,
            'component_health': self._component_health_status,
            'last_health_check': self._last_health_check.isoformat(),
            'orchestration_metrics': self._orchestration_metrics,
            'execution_history_size': len(self._execution_history),
            'market_intelligence_cache_size': len(self._market_intelligence_cache)
        }
    
    def pause_pipeline(self) -> bool:
        """Pause pipeline execution."""
        if self.pipeline_state in [PipelineState.IDLE, PipelineState.COMPLETED]:
            self.pipeline_state = PipelineState.PAUSED
            return True
        return False
    
    def resume_pipeline(self) -> bool:
        """Resume pipeline execution."""
        if self.pipeline_state == PipelineState.PAUSED:
            self.pipeline_state = PipelineState.IDLE
            return True
        return False
    
"""
Pipeline Integration with Auto-Research

This module demonstrates how to integrate the enhanced researcher
into your existing pipeline components with automatic research triggers.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import chromadb

from musequill.v3.components.base.component_interface import BaseComponent, ComponentConfiguration, ComponentType
from musequill.v3.components.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator, 
    PipelineOrchestratorConfig,
)
from musequill.v3.components.orchestration.enhanced_pipeline_config import EnhancedPipelineOrchestratorConfig
from musequill.v3.components.base.component_interface import component_registry
# Import the enhanced researcher
from .pipeline_researcher import (
    PipelineResearcher, PipelineResearcherConfig, ResearchRequest, 
    ResearchResponse, ResearchScope, ResearchPriority, ResearchContext,
    ResearchableMixin
)
from musequill.v3.components.discriminators.llm_discriminator import (
    LLMDiscriminator, 
    LLMDiscriminatorConfig,
    LLMDiscriminatorInput,
    LLMDiscriminatorOutput
)
from musequill.v3.models.llm_discriminator_models import (
    ComprehensiveLLMCritique,
    CritiqueDimension
)

from musequill.v3.models.researcher_agent_model import (
    ResearchQueryEx
)
from musequill.v3.components.market_intelligence.market_intelligence_engine import (
    MarketIntelligenceEngineInput
)

from musequill.v3.models.character_arc import (
    CharacterArc,
    NarrativeFunction
)

from musequill.v3.components.generators.plot_outliner import (
    PlotOutlinerInput, PlotOutlinerOutput
)

logger = logging.getLogger(__name__)


class ResearchTrigger(Enum):
    """Types of research triggers."""
    MANUAL = "manual"                   # Explicit research request
    AUTO_MARKET = "auto_market"         # Automatic market intelligence
    AUTO_QUALITY = "auto_quality"       # Quality-based research
    AUTO_PLOT = "auto_plot"            # Plot development research
    AUTO_CHARACTER = "auto_character"   # Character development research
    AUTO_TREND = "auto_trend"          # Trend monitoring


@dataclass
class ResearchRule:
    """Rules for automatic research triggering."""
    trigger: ResearchTrigger
    condition: Callable[[Dict[str, Any]], bool]
    query_template: str
    scope: ResearchScope
    priority: ResearchPriority
    cooldown_hours: int = 1  # Minimum time between triggers


class EnhancedPipelineOrchestrator(PipelineOrchestrator):
    """
    Enhanced pipeline orchestrator with integrated research capabilities.
    Automatically triggers research based on story state and component needs.
    """
    
    def __init__(self, config: ComponentConfiguration[EnhancedPipelineOrchestratorConfig]):
        super().__init__(config)
        
        # Access the enhanced config from the ComponentConfiguration
        enhanced_config = config.specific_config
        
        # Initialize researcher with settings from enhanced config
        researcher_config = PipelineResearcherConfig(
            tavily_api_key=enhanced_config.researcher_tavily_api_key or '',
            max_concurrent_requests=enhanced_config.researcher_max_concurrent_requests,
            enable_caching=enhanced_config.researcher_enable_caching,
            cache_ttl_hours=enhanced_config.researcher_cache_ttl_hours,
            request_timeout_seconds=enhanced_config.researcher_request_timeout_seconds,
            max_results_per_query=enhanced_config.researcher_max_results_per_query,
            
            # Auto-trigger conditions from enhanced config
            auto_trigger_conditions={
                'market_data_age_hours': enhanced_config.market_refresh_interval_hours,
                'plot_inconsistency_threshold': enhanced_config.plot_inconsistency_threshold,
                'character_development_gaps': enhanced_config.character_development_gaps_trigger
            }
        )
        
        self.researcher = PipelineResearcher(researcher_config)
        
        # Research management
        self.research_rules: List[ResearchRule] = []
        self.last_research_times: Dict[str, datetime] = {}
        self.research_enabled = enhanced_config.enable_auto_research
        
        # LLM Discriminator integration settings
        self._llm_discriminator = None
        self.llm_discriminator_enabled = enhanced_config.enable_llm_discriminator
        self.llm_discriminator_weight = enhanced_config.llm_discriminator_weight
        self.llm_discriminator_config = {
            'model': enhanced_config.llm_discriminator_model,
            'temperature': enhanced_config.llm_discriminator_temperature
        }
        
        # Setup default research rules
        self._setup_default_research_rules()
        self.components  = {}
    
    async def initialize(self):
        """Initialize the orchestrator and researcher."""
        await super().initialize()
        await self.researcher.initialize()
        
        # Initialize LLM Discriminator
        # if self.config.specific_config.enable_llm_discriminator:
        #     await self._initialize_llm_discriminator()
        
        # Inject researcher into components
        self._inject_researcher_into_components()
        
        logger.info("Enhanced pipeline orchestrator with research capabilities initialized")

        return True
    
    async def _initialize_llm_discriminator(self):
        """Initialize the LLM discriminator component."""
        try:
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
                logger.info("LLM Discriminator initialized successfully")
            else:
                logger.error("Failed to initialize LLM Discriminator")
                self._llm_discriminator = None
                
        except Exception as e:
            logger.error(f"LLM Discriminator initialization failed: {e}")
            self._llm_discriminator = None
    
    def _create_llm_discriminator_config(self) -> LLMDiscriminatorConfig:
        """Create LLM discriminator configuration from pipeline config."""
        llm_settings = self.config.specific_config
        return LLMDiscriminatorConfig(
            llm_model_name=llm_settings.llm_discriminator_model,
            analysis_temperature=llm_settings.llm_discriminator_temperature,
            max_analysis_tokens=llm_settings.llm_discriminator_max_tokens,
            critique_depth=llm_settings.llm_discriminator_depth,
            focus_areas=llm_settings.llm_discriminator_focus_areas if llm_settings.llm_discriminator_focus_areas is not None else [
                CritiqueDimension.PLOT_COHERENCE.value,
                CritiqueDimension.CHARACTER_DEVELOPMENT.value,
                CritiqueDimension.PROSE_QUALITY.value,
                CritiqueDimension.PACING.value,
                CritiqueDimension.EMOTIONAL_RESONANCE.value,
                CritiqueDimension.MARKET_APPEAL.value
            ],
            scoring_strictness=llm_settings.llm_discriminator_strictness,
            include_suggestions=llm_settings.llm_discriminator_include_suggestions,
            max_analysis_time_seconds=llm_settings.llm_discriminator_max_time
        )
    
    async def shutdown(self):
        """Shutdown the orchestrator and researcher."""
        if self._llm_discriminator:
            try:
                await self._llm_discriminator.cleanup()
                logger.info("LLM Discriminator cleaned up")
            except Exception as e:
                logger.error(f"LLM Discriminator cleanup failed: {e}")
        
        await self.researcher.shutdown()
        await super().shutdown()
    
    # RESEARCH INTEGRATION METHODS
    
    def _inject_researcher_into_components(self):
        """Inject researcher instance into components that support it."""
        # Build components dictionary from individual component references        
        self.components  = {
            'chapter_generator': self._chapter_generator,
            'plot_outliner': self._plot_outliner,
            'plot_coherence_critic': self._plot_coherence_critic,
            'literary_quality_critic': self._literary_quality_critic,
            'reader_engagement_critic': self._reader_engagement_critic,
            'quality_controller': self._quality_controller,
            'market_intelligence': self._market_intelligence_engine,
            'llm_discriminator': self._llm_discriminator,
            'character_developer': self._character_developer
        }

        for component_name, component in self.components.items():
            if component and isinstance(component, ResearchableMixin):
                component.set_researcher(self.researcher)
                logger.info(f"Injected researcher into component: {component_name}")

    def add_research_rule(self, rule: ResearchRule):
        """Add a custom research rule."""
        self.research_rules.append(rule)
        logger.info(f"Added research rule for trigger: {rule.trigger.value}")
    
    async def manual_research(
        self,
        query: str,
        component_name: str,
        scope: ResearchScope = ResearchScope.STANDARD,
        story_state: Optional[Dict[str, Any]] = None
    ) -> ResearchResponse:
        """Manually trigger research from any component."""
        context = self.researcher.create_story_research_context(
            pipeline_stage=self.current_stage,
            component_name=component_name,
            story_state=story_state or self.story_state
        )
        
        raise NotImplementedError
        # return await self.researcher.research(
        #     query=query,
        #     scope=scope,
        #     priority=ResearchPriority.HIGH,
        #     context=context
        # )
    
    # AUTOMATIC RESEARCH TRIGGERS
    
    async def check_research_triggers(self, story_state: Dict[str, Any]) -> List[ResearchResponse]:
        """Check all research rules and trigger as needed."""
        if not self.research_enabled:
            return []
        
        triggered_research = []
        
        for rule in self.research_rules:
            # Check cooldown
            rule_key = f"{rule.trigger.value}_{hash(rule.query_template)}"
            if self._is_in_cooldown(rule_key, rule.cooldown_hours):
                continue
            
            # Check condition
            if rule.condition(story_state):
                try:
                    # Generate query from template
                    if story_state['config'].get('research'):
                        queries: List[ResearchQueryEx] = self._generate_queries_from_story(story_state['config']['research'])
                    else:
                        queries = self._generate_query_from_template(rule.query_template, story_state)
                    
                    # Create context
                    context = ResearchContext(
                        pipeline_stage=self.current_stage,
                        component_name="auto_researcher",
                        story_state=story_state,
                        market_context=story_state.get('market_intelligence', {}),
                        previous_research=story_state.get('research_history', {})
                    )
                    
                    # Execute research
                    response = await self.researcher.research(
                        queries=queries,
                        scope=rule.scope,
                        priority=rule.priority,
                        context=context
                    )
                    
                    # Update cooldown
                    self.last_research_times[rule_key] = datetime.now(timezone.utc)
                    
                    # Store research in story state
                    self._store_research_in_story_state(rule.trigger, response, story_state)
                    
                    triggered_research.append(response)
                    logger.info(f"Auto-triggered research: {rule.trigger.value}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute auto-research for {rule.trigger.value}: {e}")
        
        return triggered_research
    
    # ENHANCED DISCRIMINATION PIPELINE WITH LLM
    
    async def _execute_discrimination_pipeline(
        self,
        chapter_variant,
        story_state,
        market_intelligence: Optional = None
    ) -> Dict[str, Any]:
        """
        Execute discriminator pipeline with LLM discriminator integration.
        
        NEW EVALUATION ORDER:
        1. LLM Discriminator (provides comprehensive initial assessment)
        2. Traditional discriminators (plot, literary, engagement)
        3. Comparative analysis between LLM and traditional results
        """
        discrimination_results = {
            'llm_critique': None,
            'traditional_critiques': {},
            'comparative_analysis': None,
            'final_assessment': None
        }
        
        # STEP 1: LLM Discrimination (First - provides comprehensive overview)
        if self._llm_discriminator:
            try:
                logger.info("Running LLM discriminator (first pass)")
                llm_input = LLMDiscriminatorInput(
                    chapter_variant=chapter_variant,
                    story_state=story_state,
                    market_intelligence=market_intelligence,
                    previous_chapters_summary=self._get_previous_chapters_summary(story_state)
                )
                
                llm_result = await self._llm_discriminator.process(llm_input)
                discrimination_results['llm_critique'] = llm_result
                
                logger.info(f"LLM critique completed - Overall score: {llm_result.overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"LLM discriminator failed: {e}")
        
        # STEP 2: Traditional Discriminators (Enhanced with LLM context)
        traditional_results = {}
        
        # Plot Coherence Critic
        plot_component = self.get_component('plot_coherence_critic')
        if plot_component:
            try:
                plot_input = self._create_plot_critic_input(
                    chapter_variant, 
                    story_state,
                    llm_context=discrimination_results['llm_critique']
                )
                plot_result = await plot_component.execute(plot_input)
                traditional_results['plot_coherence'] = plot_result
                logger.info(f"Plot coherence critique: {plot_result.get('overall_score', 0.0):.2f}")
                
            except Exception as e:
                logger.error(f"Plot coherence critic failed: {e}")
        
        # Literary Quality Critic
        literary_component = self.get_component('literary_quality_critic')
        if literary_component:
            try:
                literary_input = self._create_literary_critic_input(
                    chapter_variant,
                    story_state,
                    llm_context=discrimination_results['llm_critique']
                )
                literary_result = await literary_component.execute(literary_input)
                traditional_results['literary_quality'] = literary_result
                logger.info(f"Literary quality critique: {literary_result.get('overall_score', 0.0):.2f}")
                
            except Exception as e:
                logger.error(f"Literary quality critic failed: {e}")
        
        # Reader Engagement Critic
        engagement_component = self.get_component('reader_engagement_critic')
        if engagement_component:
            try:
                engagement_input = self._create_engagement_critic_input(
                    chapter_variant,
                    story_state,
                    market_intelligence,
                    llm_context=discrimination_results['llm_critique']
                )
                engagement_result = await engagement_component.execute(engagement_input)
                traditional_results['reader_engagement'] = engagement_result
                logger.info(f"Reader engagement critique: {engagement_result.get('overall_score', 0.0):.2f}")
                
            except Exception as e:
                logger.error(f"Reader engagement critic failed: {e}")
        
        discrimination_results['traditional_critiques'] = traditional_results
        
        # STEP 3: Comparative Analysis
        if discrimination_results['llm_critique'] and traditional_results:
            comparative_analysis = self._analyze_critique_agreement(
                discrimination_results['llm_critique'],
                traditional_results
            )
            discrimination_results['comparative_analysis'] = comparative_analysis
            logger.info(f"Critique agreement score: {comparative_analysis['agreement_score']:.2f}")
        
        # STEP 4: Final Assessment (Weighted combination)
        final_assessment = self._calculate_final_assessment(discrimination_results)
        discrimination_results['final_assessment'] = final_assessment
        
        logger.info(f"Final assessment score: {final_assessment['weighted_score']:.2f}")
        
        return discrimination_results
    
    # LLM INTEGRATION HELPER METHODS
    
    def _create_plot_critic_input(
        self, 
        chapter_variant, 
        story_state,
        llm_context: Optional[LLMDiscriminatorOutput] = None
    ):
        """Create plot critic input with optional LLM context."""
        additional_context = {}
        if llm_context:
            plot_score = llm_context.dimension_scores.get('plot_coherence', 0.0)
            additional_context['llm_plot_assessment'] = plot_score
            additional_context['llm_plot_feedback'] = llm_context.specific_feedback.get('plot', '')
        
        return {
            'chapter_variant': chapter_variant,
            'story_state': story_state,
            'additional_context': additional_context
        }
    
    def _create_literary_critic_input(
        self,
        chapter_variant,
        story_state,
        llm_context: Optional[LLMDiscriminatorOutput] = None
    ):
        """Create literary critic input with optional LLM context."""
        additional_context = {}
        if llm_context:
            prose_score = llm_context.dimension_scores.get('prose_quality', 0.0)
            additional_context['llm_prose_assessment'] = prose_score
            additional_context['llm_literary_feedback'] = llm_context.specific_feedback.get('literary', '')
        
        return {
            'chapter_variant': chapter_variant,
            'story_state': story_state,
            'additional_context': additional_context
        }
    
    def _create_engagement_critic_input(
        self,
        chapter_variant,
        story_state,
        market_intelligence,
        llm_context: Optional[LLMDiscriminatorOutput] = None
    ):
        """Create engagement critic input with optional LLM context."""
        additional_context = {}
        if llm_context:
            engagement_score = llm_context.dimension_scores.get('reader_engagement', 0.0)
            additional_context['llm_engagement_assessment'] = engagement_score
            additional_context['llm_engagement_feedback'] = llm_context.specific_feedback.get('engagement', '')
        
        return {
            'chapter_variant': chapter_variant,
            'story_state': story_state,
            'market_intelligence': market_intelligence,
            'additional_context': additional_context
        }
    
    def _analyze_critique_agreement(
        self,
        llm_critique: LLMDiscriminatorOutput,
        traditional_critiques: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze agreement between LLM and traditional critiques."""
        
        agreements = []
        disagreements = []
        score_comparisons = {}
        
        # Compare scores across comparable dimensions
        if 'plot_coherence' in traditional_critiques:
            llm_plot_score = llm_critique.dimension_scores.get('plot_coherence', 0.0)
            trad_plot_score = traditional_critiques['plot_coherence'].get('overall_score', 0.0)
            
            agreement = 1.0 - abs(llm_plot_score - trad_plot_score)
            score_comparisons['plot_coherence'] = {
                'llm_score': llm_plot_score,
                'traditional_score': trad_plot_score,
                'agreement': agreement
            }
            
            if agreement > 0.7:
                agreements.append("Plot coherence assessment")
            elif agreement < 0.3:
                disagreements.append("Significant plot coherence disagreement")
        
        if 'literary_quality' in traditional_critiques:
            llm_literary_score = llm_critique.dimension_scores.get('prose_quality', 0.0)
            trad_literary_score = traditional_critiques['literary_quality'].get('overall_score', 0.0)
            
            agreement = 1.0 - abs(llm_literary_score - trad_literary_score)
            score_comparisons['literary_quality'] = {
                'llm_score': llm_literary_score,
                'traditional_score': trad_literary_score,
                'agreement': agreement
            }
            
            if agreement > 0.7:
                agreements.append("Literary quality assessment")
            elif agreement < 0.3:
                disagreements.append("Significant literary quality disagreement")
        
        if 'reader_engagement' in traditional_critiques:
            llm_engagement_score = llm_critique.dimension_scores.get('reader_engagement', 0.0)
            trad_engagement_score = traditional_critiques['reader_engagement'].get('overall_score', 0.0)
            
            agreement = 1.0 - abs(llm_engagement_score - trad_engagement_score)
            score_comparisons['reader_engagement'] = {
                'llm_score': llm_engagement_score,
                'traditional_score': trad_engagement_score,
                'agreement': agreement
            }
            
            if agreement > 0.7:
                agreements.append("Reader engagement assessment")
            elif agreement < 0.3:
                disagreements.append("Significant engagement disagreement")
        
        # Overall agreement score
        agreement_scores = [comp['agreement'] for comp in score_comparisons.values()]
        overall_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
        
        return {
            'agreement_score': overall_agreement,
            'score_comparisons': score_comparisons,
            'agreements': agreements,
            'disagreements': disagreements,
            'consensus_strength': 'high' if overall_agreement > 0.8 else 'medium' if overall_agreement > 0.5 else 'low'
        }
    
    def _calculate_final_assessment(self, discrimination_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final weighted assessment from all discriminators."""
        
        # Configurable weights
        llm_weight = self.config.get('llm_discriminator_weight', 0.6)
        traditional_weight = 1.0 - llm_weight
        
        scores = []
        
        # LLM score
        if discrimination_results['llm_critique']:
            llm_score = discrimination_results['llm_critique'].overall_score
            scores.append(('llm', llm_score, llm_weight))
        
        # Traditional scores
        traditional_critiques = discrimination_results['traditional_critiques']
        if traditional_critiques:
            # Distribute traditional weight among available critics
            num_traditional = len(traditional_critiques)
            individual_traditional_weight = traditional_weight / num_traditional if num_traditional > 0 else 0
            
            for critic_name, result in traditional_critiques.items():
                score = result.get('overall_score', 0.0) if isinstance(result, dict) else getattr(result, 'overall_score', 0.0)
                scores.append((critic_name, score, individual_traditional_weight))
        
        # Calculate weighted score
        if scores:
            weighted_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            
            if total_weight > 0:
                weighted_score = weighted_score / total_weight
            else:
                weighted_score = 0.0
        else:
            weighted_score = 0.0
        
        # Agreement bonus/penalty
        comparative_analysis = discrimination_results['comparative_analysis']
        if comparative_analysis:
            agreement_score = comparative_analysis['agreement_score']
            # Slight bonus for high agreement, small penalty for major disagreement
            if agreement_score > 0.8:
                weighted_score = min(1.0, weighted_score * 1.02)  # 2% bonus
            elif agreement_score < 0.3:
                weighted_score = weighted_score * 0.98  # 2% penalty
        
        # Quality threshold check
        quality_threshold = self.config.get('quality_threshold', 0.75)
        meets_threshold = weighted_score >= quality_threshold
        
        return {
            'weighted_score': weighted_score,
            'individual_scores': {name: score for name, score, _ in scores},
            'weights_applied': {name: weight for name, _, weight in scores},
            'meets_threshold': meets_threshold,
            'recommendation': 'accept' if meets_threshold else 'revise',
            'confidence': comparative_analysis['agreement_score'] if comparative_analysis else 0.5
        }
    
    def _get_previous_chapters_summary(self, story_state):
        """Get summary of previous chapters for LLM context."""
        chapters = story_state.get('chapters', {})
        if not chapters:
            return "No previous chapters available."
        
        # Create a simple summary of previous chapters
        summaries = []
        for chapter_id, chapter_data in chapters.items():
            if isinstance(chapter_data, dict):
                summary = chapter_data.get('summary', f'Chapter {chapter_id}')
                summaries.append(summary)
        
        return '; '.join(summaries) if summaries else "Previous chapters establish the story foundation."
    
    # PIPELINE OVERRIDES WITH RESEARCH INTEGRATION
    
    async def execute_story_generation_pipeline(
        self,
        story_config: Dict[str, Any],
        manual_research_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute story generation pipeline with integrated research.
        
        Args:
            story_config: Story configuration
            manual_research_queries: Optional list of research queries to execute upfront
            
        Returns:
            Enhanced story state with research insights
        """
        # Perform upfront research if requested
        if manual_research_queries:
            logger.info(f"Executing {len(manual_research_queries)} upfront research queries")
            for query in manual_research_queries:
                response = await self.manual_research(
                    query=query,
                    component_name="upfront_research",
                    scope=ResearchScope.DEEP
                )
                # Integrate research into story config
                self._integrate_research_into_config(response, story_config)
        
        # Execute the normal pipeline with research integration
        return await self._execute_pipeline_with_research(story_config)
    
    async def _execute_pipeline_with_research(self, story_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline with automatic research integration."""
        story_state = {'config': story_config, 'research_history': {}}
        
        # Pipeline stages with research integration
        stages = [
            ('market_analysis', self._market_analysis_with_research),
            ('plot_outline', self._plot_outline_with_research),
            ('character_development', self._character_development_with_research),
            ('chapter_generation', self._chapter_generation_with_research),
            ('quality_assessment', self._quality_assessment_with_research)
        ]
        
        for stage_name, stage_func in stages:
            self.current_stage = stage_name
            logger.info(f"Executing pipeline stage: {stage_name}")
            
            try:
                # Check for automatic research triggers before stage
                research_responses = await self.check_research_triggers(story_state)
                if research_responses:
                    logger.info(f"Triggered {len(research_responses)} automatic research requests")
                
                # Execute stage
                story_state = await stage_func(story_state)
                
                # Check for post-stage research triggers
                post_research = await self.check_research_triggers(story_state)
                if post_research:
                    logger.info(f"Triggered {len(post_research)} post-stage research requests")
                
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                # Potentially trigger error-recovery research
                await self._handle_stage_error(stage_name, e, story_state)
                raise
        
        return story_state
    
    # STAGE IMPLEMENTATIONS WITH RESEARCH
    
    def get_component(self, component_name: str) -> Optional[BaseComponent]:
        """Retrieve a component by name."""
        return self.components.get(component_name)

    async def _extract_research_from_chroma(self, research_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract research data from Chroma storage based on the research results metadata.
        
        Args:
            research_results: Results from researcher containing chroma_storage_info
            
        Returns:
            Dict containing extracted research data organized by category
        """
        try:
            # Get chroma storage info
            chroma_info = research_results.get('chroma_storage_info', {})
            if not chroma_info:
                logger.warning("No chroma_storage_info found in research results")
                return None
                
            research_id = chroma_info.get('research_id')
            host = chroma_info.get('host', 'localhost')
            port = chroma_info.get('port', 18000)
            collection_name = chroma_info.get('collection_name', 'research_collection')
            
            if not research_id:
                logger.warning("No research_id found in chroma_storage_info")
                return None
            
            # Initialize Chroma client
            chroma_client = chromadb.HttpClient(host=host, port=port)
            chroma_collection = chroma_client.get_collection(name=collection_name)
            
            # Query all chunks for this research session
            results = chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas", "documents"]
            )
            
            if not results.get('metadatas'):
                logger.info(f"No research data found in Chroma for research_id: {research_id}")
                return None
            
            # Organize data by category
            research_data = {}
            category_content = {}
            
            for i, metadata in enumerate(results['metadatas']):
                category = metadata.get('query_category', 'Unknown')
                
                if category not in category_content:
                    category_content[category] = []
                
                # Get the document content
                if i < len(results['documents']):
                    document_content = results['documents'][i]
                    
                    # Only include substantial content
                    if len(document_content.strip()) > 50:
                        # Check for similarity to avoid duplicates
                        is_duplicate = any(
                            self._is_content_similar(document_content, existing_content.get('content', ''))
                            for existing_content in category_content[category]
                        )
                        
                        if not is_duplicate:
                            category_content[category].append({
                                'content': document_content.strip(),
                                'source_url': metadata.get('source_url'),
                                'source_title': metadata.get('source_title'),
                                'source_domain': metadata.get('source_domain'),
                                'tavily_score': metadata.get('tavily_score', 0.0),
                                'query_priority': metadata.get('query_priority', 'Medium')
                            })
            
            # Format for market intelligence component
            research_data = {
                'research_id': research_id,
                'categories': category_content,
                'total_categories': len(category_content),
                'total_chunks': len(results['metadatas']),
                'extraction_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Extracted research data: {len(category_content)} categories, {len(results['metadatas'])} total chunks")
            return research_data
            
        except Exception as e:
            logger.error(f"Failed to extract research data from Chroma: {e}")
            return None
    
    def _is_content_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar based on their first 100 characters.
        
        Args:
            text1: First text to compare
            text2: Second text to compare  
            threshold: Similarity threshold (0.0-1.0), default 0.8
            
        Returns:
            True if texts are similar, False otherwise
        """
        if not text1 or not text2:
            return False
        
        from difflib import SequenceMatcher
        
        # Compare first 100 characters
        sample1 = text1[:100].strip().lower()
        sample2 = text2[:100].strip().lower()
        
        # Use SequenceMatcher to calculate similarity ratio
        similarity = SequenceMatcher(None, sample1, sample2).ratio()
        return similarity >= threshold

    async def _market_analysis_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Market analysis stage with research integration."""
        config = story_state['config']
        genre = config.get('genre', 'general')
        
        # Trigger market intelligence research
        market_response = await self.researcher.market_intelligence(
            genre=genre,
            market_aspect="current trends and reader preferences",
            context=ResearchContext(
                pipeline_stage="market_analysis",
                component_name="market_analyzer",
                story_state=story_state
            )
        )
        
        # Extract actual research data from Chroma storage
        research_data = None
        if market_response.status == "completed" and market_response.results:
            research_data = await self._extract_research_from_chroma(market_response.results)
        
        # Execute market analysis component with research data
        market_component = self.get_component('market_intelligence')
        if market_component:
            market_input_dict = {
                'target_genre': genre,
                'research_results': research_data or {},
                'analysis_depth': 'comprehensive'
            }

            market_input: MarketIntelligenceEngineInput = MarketIntelligenceEngineInput(**market_input_dict)
            market_result = await market_component.process(market_input)
            story_state['market_intelligence'] = market_result
        
        return story_state
    
    async def _plot_outline_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Plot outline stage with research integration."""
        config = story_state['config']
        genre = config.get('genre', 'general')
        plot_type = config.get('plot_type', 'standard')
        
        # Research successful plot structures
        plot_research = await self.researcher.plot_research(
            plot_element=f"{plot_type} plot structure",
            genre=genre,
            story_context=story_state
        )
        
        # Execute plot outline component
        plot_component = self.get_component('plot_outliner')
        if plot_component and isinstance(plot_component, ResearchableMixin):
            plot_input = {
                'story_config': config,
                'genre': genre,
                'plot_type': plot_type,
                'market_intelligence': story_state.get('market_intelligence', {}),
                'research_insights': plot_research.results if plot_research.status == "completed" else None
            }
            plot_outliner_input = PlotOutlinerInput(**plot_input)
            plot_result = await plot_component.process(plot_outliner_input)
            story_state['plot_outline'] = plot_result
        
        return story_state
    
    async def _character_development_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Character development stage with research integration - FIXED VERSION."""
        config = story_state['config']
        genre = config.get('genre', 'general')
        
        # Research character development techniques
        character_research = await self.researcher.deep_research(
            topic=f"{genre} character development techniques",
            specific_questions=[
                f"What makes {genre} characters memorable and relatable?",
                f"What character arcs work best in {genre} fiction?",
                f"What character flaws and strengths resonate with {genre} readers?",
                f"How do successful {genre} authors develop character voice?"
            ],
            context=ResearchContext(
                pipeline_stage="character_development",
                component_name="character_developer",
                story_state=story_state
            )
        )
        
        # Execute character development component
        character_component = self.get_component('character_developer')
        if character_component:
            # CORRECTED IMPORT PATH
            from musequill.v3.components.character_developer.character_development_models import CharacterDevelopmentInput
            from musequill.v3.models.character_arc import CharacterArc, VoiceCharacteristics, NarrativeFunction, CharacterStability
            
            # Extract refined research insights from chroma results
            research_insights = None
            if character_research.status == "completed" and character_research.results:
                chroma_info = character_research.results.get('chroma_storage_info', {})
                
                if chroma_info and chroma_info.get('total_chunks', 0) > 0:
                    # Use chroma_storage_info to access the ChromaDB store
                    collection_name = chroma_info['collection_name']
                    research_id = chroma_info['research_id']
                    host = chroma_info['host']
                    port = chroma_info['port']
                    
                    # Access ChromaDB to retrieve stored research results
                    try:
                        import chromadb
                        chroma_client = chromadb.HttpClient(host=host, port=port)
                        collection = chroma_client.get_collection(name=collection_name)
                        
                        # Query all documents related to this research_id
                        results = collection.get(
                            where={"research_id": research_id},
                            include=["documents", "metadatas"]
                        )
                        
                        # Transform results into organized format
                        research_insights = {}
                        for doc, metadata in zip(results['documents'], results['metadatas']):
                            category = metadata.get('query_category', 'General')
                            if category not in research_insights:
                                research_insights[category] = []
                            research_insights[category].append(doc)
                            
                    except Exception as e:
                        print(f"Warning: Could not retrieve ChromaDB results: {e}")
                        # Fallback to basic results
                        research_insights = character_research.results
                else:
                    # No chunks stored, fallback to basic results
                    research_insights = character_research.results
            
            # Construct characters from story_state
            characters = self._construct_characters_from_story_state(story_state)
            
            # Extract market intelligence data safely using helper methods
            market_intel = story_state.get('market_intelligence', {})
            market_preferences = self._safe_get_attribute(market_intel, 'reader_preferences')
            success_patterns = self._safe_get_attribute(market_intel, 'success_patterns')
            
            character_input = CharacterDevelopmentInput(
                characters=characters,
                current_chapter=story_state.get('current_chapter', 1),
                plot_outline=story_state.get('plot_outline', {}),
                genre=genre,
                market_preferences=market_preferences,
                success_patterns=success_patterns,
                research_insights=research_insights,
                chapter_objectives=story_state.get('chapter_objectives', []),
                constraints=story_state.get('constraints', {}),
                force_development=story_state.get('force_character_development', [])
            )
            
            # USE invoke() method from BaseComponent interface
            character_result = await character_component.invoke(character_input)
            story_state['characters'] = character_result.updated_characters
            story_state['character_development_result'] = character_result
        else:
            logger.warning("Character developer component not available")
        
        return story_state
    
    def _construct_characters_from_story_state(self, story_state: Dict[str, Any]) -> Dict[str, CharacterArc]:
        """Construct CharacterArc objects from story_state configuration."""
        from musequill.v3.models.character_arc import CharacterArc, VoiceCharacteristics, NarrativeFunction, CharacterStability
        
        characters = {}
        config = story_state.get('config', {})
        current_chapter = story_state.get('current_chapter', 1)
        
        # Process main character
        main_char = config.get('main_character')
        if main_char:
            char_id = f"main_{main_char.get('name', 'character').lower().replace(' ', '_')}"
            characters[char_id] = self._create_character_arc_from_config(
                main_char, char_id, NarrativeFunction.PROTAGONIST, current_chapter
            )
        
        # Process supporting characters
        supporting_chars = config.get('supporting_characters', [])
        for i, char_data in enumerate(supporting_chars):
            char_id = f"supporting_{char_data.get('name', f'character_{i}').lower().replace(' ', '_')}"
            
            # Determine narrative function from role
            role = char_data.get('role', 'support').lower()
            if role == 'antagonist':
                narrative_function = NarrativeFunction.ANTAGONIST
            elif role == 'mentor':
                narrative_function = NarrativeFunction.MENTOR
            elif role == 'ally':
                narrative_function = NarrativeFunction.SUPPORT
            elif role == 'catalyst':
                narrative_function = NarrativeFunction.CATALYST
            elif role == 'foil':
                narrative_function = NarrativeFunction.FOIL
            else:
                narrative_function = NarrativeFunction.SUPPORT
            
            characters[char_id] = self._create_character_arc_from_config(
                char_data, char_id, narrative_function, current_chapter
            )
        
        return characters
    
    def _create_character_arc_from_config(
        self, 
        char_data: Dict[str, Any], 
        char_id: str, 
        narrative_function: NarrativeFunction,
        current_chapter: int
    ) -> CharacterArc:
        """Create a CharacterArc from character configuration data."""
        from musequill.v3.models.character_arc import CharacterArc, VoiceCharacteristics, CharacterStability
        
        name = char_data.get('name', 'Unnamed Character')
        background = char_data.get('background', '')
        motivation = char_data.get('motivation', '')
        personality_traits = char_data.get('personality_traits', [])
        
        # Create voice characteristics based on background and traits
        voice_chars = VoiceCharacteristics(
            vocabulary_level=self._infer_vocabulary_level(background, personality_traits),
            speech_patterns=[],  # Would be populated later or from more detailed config
            formality_level=self._infer_formality_level(background, personality_traits),
            emotional_expressiveness=self._infer_emotional_expressiveness(personality_traits),
            dialogue_quirks=[]  # Would be populated later or from more detailed config
        )
        
        # Create character arc
        char_arc = CharacterArc(
            character_id=char_id,
            name=name,
            emotional_state=self._infer_initial_emotional_state(motivation, personality_traits),
            stability=CharacterStability.STABLE,  # Default, could be inferred
            narrative_function=narrative_function,
            voice_characteristics=voice_chars,
            last_development_chapter=1,  # Assume introduced in chapter 1
            introduction_chapter=1,
            personality_traits=personality_traits if isinstance(personality_traits, list) else [],
            goals_motivations=[motivation] if motivation else [],
            remaining_obstacles=self._infer_initial_obstacles(motivation, background),
            internal_conflicts=self._infer_internal_conflicts(personality_traits, motivation)
        )
        
        return char_arc
    
    def _infer_vocabulary_level(self, background: str, traits: List[str]) -> str:
        """Infer vocabulary level from character background and traits."""
        background_lower = background.lower() if background else ""
        traits_lower = [t.lower() for t in traits] if traits else []
        
        # Check for academic/professional indicators
        academic_indicators = ['professor', 'doctor', 'researcher', 'scientist', 'academic', 'scholar']
        professional_indicators = ['lawyer', 'executive', 'manager', 'consultant', 'analyst']
        
        if any(indicator in background_lower for indicator in academic_indicators):
            return "academic"
        elif any(indicator in background_lower for indicator in professional_indicators):
            return "complex"
        elif any(trait in traits_lower for trait in ['intelligent', 'brilliant', 'educated']):
            return "complex"
        elif any(trait in traits_lower for trait in ['simple', 'down-to-earth', 'practical']):
            return "simple"
        else:
            return "moderate"
    
    def _infer_formality_level(self, background: str, traits: List[str]) -> str:
        """Infer formality level from character background and traits."""
        background_lower = background.lower() if background else ""
        traits_lower = [t.lower() for t in traits] if traits else []
        
        formal_indicators = ['professor', 'doctor', 'executive', 'diplomat', 'judge']
        casual_indicators = ['hacker', 'artist', 'rebel', 'street', 'young']
        
        if any(indicator in background_lower for indicator in formal_indicators):
            return "formal"
        elif any(indicator in background_lower for indicator in casual_indicators):
            return "casual"
        elif any(trait in traits_lower for trait in ['formal', 'proper', 'dignified']):
            return "formal"
        elif any(trait in traits_lower for trait in ['casual', 'relaxed', 'informal', 'rebellious']):
            return "casual"
        else:
            return "informal"
    
    def _infer_emotional_expressiveness(self, traits: List[str]) -> str:
        """Infer emotional expressiveness from personality traits."""
        traits_lower = [t.lower() for t in traits] if traits else []
        
        if any(trait in traits_lower for trait in ['dramatic', 'expressive', 'passionate', 'emotional']):
            return "expressive"
        elif any(trait in traits_lower for trait in ['reserved', 'stoic', 'controlled', 'calm']):
            return "reserved"
        elif any(trait in traits_lower for trait in ['intense', 'fiery', 'volatile']):
            return "dramatic"
        else:
            return "moderate"
    
    def _infer_initial_emotional_state(self, motivation: str, traits: List[str]) -> str:
        """Infer initial emotional state from motivation and traits."""
        motivation_lower = motivation.lower() if motivation else ""
        traits_lower = [t.lower() for t in traits] if traits else []
        
        # Look for emotional indicators in motivation
        if any(word in motivation_lower for word in ['protect', 'save', 'help']):
            return "determined and protective"
        elif any(word in motivation_lower for word in ['revenge', 'justice', 'expose']):
            return "driven and focused"
        elif any(word in motivation_lower for word in ['discover', 'find', 'learn']):
            return "curious and motivated"
        elif any(trait in traits_lower for trait in ['anxious', 'worried', 'fearful']):
            return "anxious but determined"
        elif any(trait in traits_lower for trait in ['confident', 'bold', 'brave']):
            return "confident and ready"
        else:
            return "focused and purposeful"
    
    def _infer_initial_obstacles(self, motivation: str, background: str) -> List[str]:
        """Infer initial obstacles from character motivation and background."""
        obstacles = []
        motivation_lower = motivation.lower() if motivation else ""
        background_lower = background.lower() if background else ""
        
        # Common obstacles based on motivation themes
        if "protect" in motivation_lower:
            obstacles.append("Balancing personal safety with protective instincts")
        if "technology" in motivation_lower or "corporate" in motivation_lower:
            obstacles.append("Overcoming corporate/institutional power")
        if "past" in background_lower or "former" in background_lower:
            obstacles.append("Dealing with consequences of past decisions")
        if not obstacles:  # Default obstacles
            obstacles.append("Learning to trust others")
            obstacles.append("Overcoming self-doubt")
        
        return obstacles
    
    def _infer_internal_conflicts(self, traits: List[str], motivation: str) -> List[str]:
        """Infer internal conflicts from personality traits and motivation."""
        conflicts = []
        traits_lower = [t.lower() for t in traits] if traits else []
        motivation_lower = motivation.lower() if motivation else ""
        
        # Look for conflicting traits or challenging motivations
        if any(trait in traits_lower for trait in ['brilliant', 'intelligent']) and any(trait in traits_lower for trait in ['haunted', 'guilty']):
            conflicts.append("Intelligence vs emotional burden from past")
        if "protect" in motivation_lower and any(trait in traits_lower for trait in ['independent', 'lone']):
            conflicts.append("Desire to protect vs tendency to work alone")
        if any(trait in traits_lower for trait in ['ethical', 'moral']) and "corporate" in motivation_lower:
            conflicts.append("Personal ethics vs practical necessities")
        if not conflicts:  # Default conflicts
            conflicts.append("Balancing personal needs with greater responsibilities")
        
        return conflicts

    
    async def _chapter_generation_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Chapter generation stage with research integration."""
        config = story_state['config']
        genre = config.get('genre', 'general')
        
        # Execute chapter generation with research support
        chapter_component = self.get_component('chapter_generator')
        if chapter_component and isinstance(chapter_component, ResearchableMixin):
            # Component can perform its own research as needed
            chapter_input = {
                'plot_outline': story_state.get('plot_outline', {}),
                'characters': story_state.get('characters', {}),
                'genre': genre,
                'story_state': story_state
            }
            chapter_result = await chapter_component.process(chapter_input)
            story_state['chapters'] = chapter_result
        
        return story_state
    
    async def _quality_assessment_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quality assessment stage with research integration."""
        # Research current quality standards
        quality_research = await self.researcher.quick_search(
            "fiction quality assessment criteria reader expectations 2025"
        )
        
        # Execute quality assessment
        quality_component = self.get_component('quality_assessor')
        if quality_component:
            quality_input = {
                'chapters': story_state.get('chapters', {}),
                'quality_criteria': quality_research.results if quality_research.status == "completed" else None,
                'market_standards': story_state.get('market_intelligence', {})
            }
            quality_result = await quality_component.execute(quality_input)
            story_state['quality_assessment'] = quality_result
        
        return story_state
    
    # HELPER METHODS
    
    def _safe_get_attribute(self, obj: Any, attribute: str, default: Any = None) -> Any:
        """Safely get an attribute from an object or dictionary."""
        if hasattr(obj, attribute):
            return getattr(obj, attribute)
        elif isinstance(obj, dict):
            return obj.get(attribute, default)
        else:
            return default
    
    def _safe_get_nested_attribute(self, obj: Any, path: str, default: Any = None) -> Any:
        """Safely get a nested attribute using dot notation."""
        try:
            current = obj
            for part in path.split('.'):
                current = self._safe_get_attribute(current, part)
                if current is None:
                    return default
            return current
        except (AttributeError, KeyError, TypeError):
            return default
    
    def _safe_get_score(self, obj: Any, score_field: str = 'overall_score', default: float = 0.0) -> float:
        """Safely extract a score from an object or dict."""
        score = self._safe_get_attribute(obj, score_field, default)
        try:
            return float(score)
        except (ValueError, TypeError):
            return default
    
    def _setup_default_research_rules(self):
        """Setup default automatic research rules."""
        
        # Market intelligence refresh
        self.add_research_rule(ResearchRule(
            trigger=ResearchTrigger.AUTO_MARKET,
            condition=lambda state: self._market_data_is_stale(state),
            query_template="{genre} fiction market trends reader preferences 2025",
            scope=ResearchScope.TARGETED,
            priority=ResearchPriority.NORMAL,
            cooldown_hours=6
        ))
        
        # Plot quality check
        self.add_research_rule(ResearchRule(
            trigger=ResearchTrigger.AUTO_PLOT,
            condition=lambda state: self._plot_needs_research(state),
            query_template="{genre} plot structure techniques effective storytelling",
            scope=ResearchScope.STANDARD,
            priority=ResearchPriority.HIGH,
            cooldown_hours=2
        ))
        
        # Character development gaps
        self.add_research_rule(ResearchRule(
            trigger=ResearchTrigger.AUTO_CHARACTER,
            condition=lambda state: self._character_development_gaps(state),
            query_template="{genre} character development memorable protagonists",
            scope=ResearchScope.TARGETED,
            priority=ResearchPriority.HIGH,
            cooldown_hours=3
        ))
        
        # Trend monitoring
        self.add_research_rule(ResearchRule(
            trigger=ResearchTrigger.AUTO_TREND,
            condition=lambda state: self._should_monitor_trends(state),
            query_template="emerging fiction trends {genre} publishing industry 2025",
            scope=ResearchScope.QUICK,
            priority=ResearchPriority.LOW,
            cooldown_hours=12
        ))
    
    def _market_data_is_stale(self, state: Dict[str, Any]) -> bool:
        """Check if market data needs refreshing."""
        market_intel = state.get('market_intelligence', {})
        if not market_intel:
            return True
        
        # Safely access timestamp field from MarketIntelligenceReport or dict
        last_update = (self._safe_get_attribute(market_intel, 'generated_at') or 
                      self._safe_get_attribute(market_intel, 'last_updated'))
        
        if not last_update:
            return True
        
        # Convert string to datetime if needed
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        
        hours_since = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
        return hours_since > 6
    
    def _plot_needs_research(self, state: Dict[str, Any]) -> bool:
        """Check if plot development needs research support."""
        plot_outline = state.get('plot_outline', {})
        if not plot_outline:
            return False
        
        # Check for plot consistency issues
        consistency_score = plot_outline.get('consistency_score', 1.0)
        return consistency_score < 0.8
    
    def _character_development_gaps(self, state: Dict[str, Any]) -> bool:
        """Check if character development has gaps needing research."""
        characters = state.get('characters', {})
        character_development_result = state.get('character_development_result', {})
        
        if not characters:
            return False
        
        # Check for character development issues using safe access methods
        char_scores = []
        
        # First try to get scores from character_development_result assessments
        assessments = self._safe_get_attribute(character_development_result, 'assessments', {})
        
        if assessments:
            for assessment in assessments.values():
                score = self._safe_get_score(assessment, 'overall_health_score', 1.0)
                char_scores.append(score)
        else:
            # Fallback: use CharacterArc objects directly
            for char in characters.values():
                development_staleness = self._safe_get_attribute(char, 'development_staleness', 0)
                if development_staleness is not None:
                    # Convert development_staleness to a score (lower staleness = higher score)
                    score = max(0.0, 1.0 - (float(development_staleness) / 10.0))
                    char_scores.append(score)
                else:
                    # Try other score fields or use default
                    score = self._safe_get_score(char, 'development_score', 0.7)
                    char_scores.append(score)
        
        avg_score = sum(char_scores) / len(char_scores) if char_scores else 1.0
        return avg_score < 0.7
    
    def _should_monitor_trends(self, state: Dict[str, Any]) -> bool:
        """Check if trend monitoring is needed."""
        research_history = state.get('research_history', {})
        last_trend_research = research_history.get('last_trend_research')
        
        if not last_trend_research:
            return True
        
        if isinstance(last_trend_research, str):
            last_trend_research = datetime.fromisoformat(last_trend_research.replace('Z', '+00:00'))
        
        hours_since = (datetime.now(timezone.utc) - last_trend_research).total_seconds() / 3600
        return hours_since > 24
    
    def _is_in_cooldown(self, rule_key: str, cooldown_hours: int) -> bool:
        """Check if a research rule is in cooldown period."""
        if rule_key not in self.last_research_times:
            return False
        
        last_time = self.last_research_times[rule_key]
        hours_since = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
        return hours_since < cooldown_hours

    def _generate_queries_from_story(self, queries: List[Dict[str, Any]]) -> List[ResearchQueryEx]:
        qs: List[ResearchQueryEx] = []
        for q in queries:
            query = ResearchQueryEx(**q['query'])
            qs.append(query)
        return qs

    def _generate_query_from_template(self, template: str, state: Dict[str, Any]) -> str:
        """Generate research query from template using story state."""
        return template.format(
            genre=state.get('config', {}).get('genre', 'general'),
            plot_type=state.get('config', {}).get('plot_type', 'standard'),
            current_stage=self.current_stage
        )
    
    def _store_research_in_story_state(
        self,
        trigger: ResearchTrigger,
        response: ResearchResponse,
        state: Dict[str, Any]
    ):
        """Store research results in story state."""
        if 'research_history' not in state:
            state['research_history'] = {}
        
        state['research_history'][f'{trigger.value}_{response.request_id}'] = {
            'trigger': trigger.value,
            'query': response.request_id,
            'status': response.status,
            'sources_found': response.sources_found,
            'timestamp': response.completed_at.isoformat() if response.completed_at else None,
            'summary': response.summary
        }
        
        # Store special research types in dedicated locations
        if trigger == ResearchTrigger.AUTO_MARKET:
            state.setdefault('market_intelligence', {})['latest_research'] = response.results
            state['market_intelligence']['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        elif trigger == ResearchTrigger.AUTO_TREND:
            state.setdefault('trend_analysis', {})['latest_trends'] = response.results
            state['research_history']['last_trend_research'] = datetime.now(timezone.utc).isoformat()
    
    def _integrate_research_into_config(self, response: ResearchResponse, config: Dict[str, Any]):
        """Integrate research findings into story configuration."""
        if response.status != "completed" or not response.results:
            return
        
        # Add research insights to config
        if 'research_insights' not in config:
            config['research_insights'] = []
        
        config['research_insights'].append({
            'query': response.request_id,
            'findings': response.summary,
            'sources': response.sources_found,
            'timestamp': response.completed_at.isoformat() if response.completed_at else None
        })
    
    async def _handle_stage_error(self, stage_name: str, error: Exception, state: Dict[str, Any]):
        """Handle stage execution errors, potentially with research assistance."""
        logger.error(f"Stage {stage_name} failed with error: {error}")
        
        # Trigger research for error recovery if appropriate
        if "plot" in stage_name.lower():
            await self.manual_research(
                query=f"common {stage_name} problems solutions fiction writing",
                component_name="error_recovery",
                scope=ResearchScope.QUICK,
                story_state=state
            )


# ENHANCED COMPONENT BASE CLASS

class ResearchEnabledComponent(BaseComponent, ResearchableMixin):
    """
    Base class for components that can utilize research capabilities.
    Combines the standard component interface with research support.
    """
    
    async def research_before_execution(self, input_data: Dict[str, Any]) -> Optional[ResearchResponse]:
        """Override this method to perform research before main execution."""
        return None
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute component with optional pre-execution research."""
        
        # Perform research if needed
        research_response = await self.research_before_execution(input_data)
        if research_response and research_response.status == "completed":
            input_data['research_insights'] = research_response.results
        
        # Execute main component logic
        return await self.main_execute(input_data)
    
    async def main_execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method with the main component logic."""
        raise NotImplementedError("Subclasses must implement main_execute")


# EXAMPLE USAGE

async def example_enhanced_pipeline():
    """Example of using the enhanced pipeline with research integration."""
    
    # Configuration
    config = {
        'researcher': {
            'tavily_api_key': 'your-tavily-api-key',
            'max_concurrent_requests': 3,
            'enable_caching': True,
            'cache_ttl_hours': 12
        },
        'enable_auto_research': True,
        'enable_llm_discriminator': True,
        'llm_discriminator_weight': 0.6,
        'llm_discriminator': {
            'model': 'llama3.3:70b',
            'temperature': 0.2,
            'max_tokens': 2000,
            'depth': 'comprehensive',
            'focus_areas': [
                'plot_coherence',
                'character_development',
                'prose_quality',
                'pacing',
                'emotional_resonance',
                'market_appeal'
            ],
            'strictness': 0.75,
            'include_suggestions': True,
            'max_time': 90
        },
        'components': {
            'market_intelligence': 'MarketIntelligenceComponent',
            'plot_outliner': 'PlotOutlineComponent',
            'character_developer': 'CharacterDevelopmentComponent',
            'chapter_generator': 'ChapterGeneratorComponent',
            'quality_assessor': 'QualityAssessmentComponent'
        }
    }
    
    # Create enhanced orchestrator
    orchestrator = EnhancedPipelineOrchestrator(config)
    await orchestrator.initialize()
    
    try:
        # Story configuration
        story_config = {
            'genre': 'thriller',
            'plot_type': 'mystery',
            'target_length': 'novel',
            'target_audience': 'adult'
        }
        
        # Optional upfront research
        upfront_research = [
            "thriller fiction market trends 2025 reader preferences",
            "mystery plot structures successful techniques",
            "thriller character archetypes compelling protagonists"
        ]
        
        # Execute story generation pipeline with research
        final_story_state = await orchestrator.execute_story_generation_pipeline(
            story_config=story_config,
            manual_research_queries=upfront_research
        )
        
        print("Pipeline completed with integrated research!")
        print(f"Research history: {len(final_story_state.get('research_history', {}))}")
        
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(example_enhanced_pipeline())
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

from musequill.v3.components.base.component_interface import BaseComponent
from musequill.v3.components.orchestration.pipeline_orchestrator import PipelineOrchestrator

# Import the enhanced researcher
from .pipeline_researcher import (
    PipelineResearcher, PipelineResearcherConfig, ResearchRequest, 
    ResearchResponse, ResearchScope, ResearchPriority, ResearchContext,
    ResearchableMixin
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
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize researcher
        researcher_config = PipelineResearcherConfig(**config.get('researcher', {}))
        self.researcher = PipelineResearcher(researcher_config)
        
        # Research management
        self.research_rules: List[ResearchRule] = []
        self.last_research_times: Dict[str, datetime] = {}
        self.research_enabled = config.get('enable_auto_research', True)
        
        # Setup default research rules
        self._setup_default_research_rules()
    
    async def initialize(self):
        """Initialize the orchestrator and researcher."""
        await super().initialize()
        await self.researcher.initialize()
        
        # Inject researcher into components
        self._inject_researcher_into_components()
        
        logger.info("Enhanced pipeline orchestrator with research capabilities initialized")
    
    async def shutdown(self):
        """Shutdown the orchestrator and researcher."""
        await self.researcher.shutdown()
        await super().shutdown()
    
    # RESEARCH INTEGRATION METHODS
    
    def _inject_researcher_into_components(self):
        """Inject researcher instance into components that support it."""
        for component_name, component in self.components.items():
            if isinstance(component, ResearchableMixin):
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
        
        return await self.researcher.research(
            query=query,
            scope=scope,
            priority=ResearchPriority.HIGH,
            context=context
        )
    
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
                    query = self._generate_query_from_template(rule.query_template, story_state)
                    
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
                        query=query,
                        scope=rule.scope,
                        priority=rule.priority,
                        context=context
                    )
                    
                    # Update cooldown
                    self.last_research_times[rule_key] = datetime.now(timezone.utc)
                    
                    # Store research in story state
                    self._store_research_in_story_state(rule.trigger, response, story_state)
                    
                    triggered_research.append(response)
                    logger.info(f"Auto-triggered research: {rule.trigger.value} - {query}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute auto-research for {rule.trigger.value}: {e}")
        
        return triggered_research
    
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
        
        # Execute market analysis component with research data
        market_component = self.get_component('market_intelligence')
        if market_component:
            market_input = {
                'genre': genre,
                'research_data': market_response.results if market_response.status == "completed" else None,
                'analysis_depth': 'comprehensive'
            }
            market_result = await market_component.execute(market_input)
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
        if plot_component:
            plot_input = {
                'genre': genre,
                'plot_type': plot_type,
                'market_intelligence': story_state.get('market_intelligence', {}),
                'research_insights': plot_research.results if plot_research.status == "completed" else None
            }
            plot_result = await plot_component.execute(plot_input)
            story_state['plot_outline'] = plot_result
        
        return story_state
    
    async def _character_development_with_research(self, story_state: Dict[str, Any]) -> Dict[str, Any]:
        """Character development stage with research integration."""
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
            character_input = {
                'plot_outline': story_state.get('plot_outline', {}),
                'genre': genre,
                'research_insights': character_research.results if character_research.status == "completed" else None,
                'market_preferences': story_state.get('market_intelligence', {})
            }
            character_result = await character_component.execute(character_input)
            story_state['characters'] = character_result
        
        return story_state
    
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
            chapter_result = await chapter_component.execute(chapter_input)
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
        
        last_update = market_intel.get('last_updated')
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
        if not characters:
            return False
        
        # Check for character development issues
        char_scores = [char.get('development_score', 1.0) for char in characters.values()]
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
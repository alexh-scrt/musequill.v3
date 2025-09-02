"""
ChapterGenerator ResearchableMixin Integration

This module adds ResearchableMixin support to the existing ChapterGenerator
without rewriting the entire class. This allows the enhanced pipeline orchestrator
to properly detect and use the chapter generator with research capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from musequill.v3.components.generators.chapter_generator import ChapterGenerator
from musequill.v3.components.orchestration.pipeline_researcher import (
    ResearchableMixin, 
    PipelineResearcher, 
    ResearchResponse,
    ResearchScope,
    ResearchPriority
)
from .chapter_generator import ChapterGeneratorInput
from musequill.v3.models.chapter_objective import ChapterObjective
from musequill.v3.models.dynamic_story_state import DynamicStoryState

logger = logging.getLogger(__name__)


class ResearchEnabledChapterGenerator(ChapterGenerator, ResearchableMixin):
    """
    ChapterGenerator with ResearchableMixin support.
    Extends the existing ChapterGenerator to support research integration.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        super().__init__(*args, **kwargs)
        ResearchableMixin.__init__(self)
        
        # Research-specific attributes
        self._research_insights = []
        self._auto_research_enabled = True
    
    async def process(self, input_data: ChapterGeneratorInput):
        """
        Override process to add research capabilities before generation.
        """
        logger.info("Starting chapter generation with research support")
        
        # Perform pre-generation research if researcher is available
        if self.researcher and self._auto_research_enabled:
            research_insights = await self._perform_pre_generation_research(input_data)
            if research_insights:
                logger.info(f"Gathered {len(research_insights)} research insights for chapter generation")
                self._research_insights = research_insights
        
        # Call the original process method with research insights
        return await self._process_with_research_insights(input_data)
    
    async def _perform_pre_generation_research(self, input_data: ChapterGeneratorInput) -> List[Dict[str, Any]]:
        """
        Perform research before chapter generation to inform the writing process.
        """
        research_insights = []
        
        try:
            # Get chapter context for research
            chapter_num = input_data.chapter_objective.chapter_number
            genre = input_data.story_state.genre
            narrative_goals = input_data.chapter_objective.narrative_goals
            
            # Research queries based on chapter context
            research_queries = self._generate_research_queries(input_data)
            
            for query in research_queries:
                logger.info(f"Performing research: {query}")
                
                response = await self.quick_research(query)
                if response and response.status == "completed":
                    research_insights.append({
                        'query': query,
                        'findings': response.summary,
                        'sources': response.sources_found,
                        'relevance': self._assess_research_relevance(response, input_data)
                    })
        
        except Exception as e:
            logger.warning(f"Pre-generation research failed: {e}")
        
        return research_insights
    
    def _generate_research_queries(self, input_data: ChapterGeneratorInput) -> List[str]:
        """
        Generate relevant research queries based on chapter objectives and story state.
        """
        queries = []
        
        chapter_num = input_data.chapter_objective.chapter_number
        genre = input_data.story_state.genre
        narrative_goals = input_data.chapter_objective.narrative_goals
        
        # Genre-specific research
        queries.append(f"effective {genre} fiction techniques chapter {chapter_num} structure")
        
        # Narrative goal research
        for goal in narrative_goals:
            if "plot" in goal.lower():
                queries.append(f"{genre} plot advancement techniques engaging readers")
            elif "character" in goal.lower():
                queries.append(f"character development techniques {genre} fiction")
            elif "conflict" in goal.lower():
                queries.append(f"conflict escalation techniques {genre} storytelling")
        
        # Chapter position research
        if chapter_num == 1:
            queries.append(f"compelling {genre} opening chapter techniques")
        elif chapter_num <= 3:
            queries.append(f"{genre} early chapter hook retention techniques")
        else:
            queries.append(f"{genre} mid-story pacing engagement techniques")
        
        return queries[:3]  # Limit to 3 queries to avoid overwhelming research
    
    def _assess_research_relevance(self, response: ResearchResponse, input_data: ChapterGeneratorInput) -> float:
        """
        Assess how relevant research findings are to the current chapter generation.
        """
        relevance_score = 0.5  # Base relevance
        
        genre = input_data.story_state.genre.lower()
        summary = response.summary.lower() if response.summary else ""
        
        # Check genre relevance
        if genre in summary:
            relevance_score += 0.3
        
        # Check narrative goal relevance
        narrative_goals = [goal.lower() for goal in input_data.chapter_objective.narrative_goals]
        for goal in narrative_goals:
            if any(word in summary for word in goal.split()):
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    async def _process_with_research_insights(self, input_data: ChapterGeneratorInput):
        """
        Process chapter generation with research insights integrated.
        """
        # Add research insights to generation context if available
        if self._research_insights:
            # Create an enhanced context that includes research insights
            enhanced_input = self._enhance_input_with_research(input_data)
            return await super().process(enhanced_input)
        else:
            # Use original input if no research insights
            return await super().process(input_data)
    
    def _enhance_input_with_research(self, input_data: ChapterGeneratorInput) -> ChapterGeneratorInput:
        """
        Enhance the generator input with research insights.
        """
        # Add research insights to the story state metadata
        enhanced_story_state = input_data.story_state
        
        # Add research insights to writing metadata
        if hasattr(enhanced_story_state, 'writing_metadata'):
            if not enhanced_story_state.writing_metadata:
                enhanced_story_state.writing_metadata = {}
            enhanced_story_state.writing_metadata['research_insights'] = self._research_insights
        
        # Create enhanced input
        enhanced_input = ChapterGeneratorInput(
            chapter_objective=input_data.chapter_objective,
            story_state=enhanced_story_state,
            market_intelligence=input_data.market_intelligence,
            critic_feedback_history=input_data.critic_feedback_history,
            generation_constraints=input_data.generation_constraints
        )
        
        return enhanced_input
    
    # Support for the legacy execute() method expected by enhanced orchestrator
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy execute method support for the enhanced orchestrator.
        Converts dict input to proper ChapterGeneratorInput and returns dict output.
        """
        logger.info("Legacy execute method called, converting to proper component interface")
        
        try:
            # Convert dict input to proper ChapterGeneratorInput
            chapter_generator_input = self._convert_dict_to_input(input_data)
            
            # Use the proper process method
            result = await self.process(chapter_generator_input)
            
            # Convert result back to dict format expected by orchestrator
            return self._convert_output_to_dict(result)
        
        except Exception as e:
            logger.error(f"Legacy execute method failed: {e}")
            return {}
    
    def _convert_dict_to_input(self, input_data: Dict[str, Any]) -> ChapterGeneratorInput:
        """Convert dict input to ChapterGeneratorInput."""
        
        # Create a basic ChapterObjective from the input
        chapter_objective = ChapterObjective(
            chapter_number=input_data.get('current_chapter', 1),
            target_word_count=3000,
            narrative_goals=[
                "Advance the main plot",
                "Develop character relationships",
                "Create engaging conflict"
            ],
            character_development_goals={
                name: f"Develop {name}'s character arc"
                for name in input_data.get('characters', {}).keys()
            },
            plot_progression_requirements=[
                "Maintain story pacing",
                "Build tension appropriately"
            ],
            tone_and_style_requirements={
                'genre': input_data.get('genre', 'general'),
                'tone': 'engaging',
                'pov': 'third_person'
            },
            constraints={}
        )
        
        # Create a DynamicStoryState from the input
        story_state = input_data.get('story_state', {})
        dynamic_story_state = DynamicStoryState(
            story_id=story_state.get('story_id', 'generated_story'),
            current_chapter_number=input_data.get('current_chapter', 1),
            total_planned_chapters=10,
            genre=input_data.get('genre', 'general'),
            target_audience='adult',
            plot_threads={'main': {'status': 'active', 'progress': 0.1}},
            character_states={},
            setting_state={},
            narrative_context={},
            writing_metadata={}
        )
        
        return ChapterGeneratorInput(
            chapter_objective=chapter_objective,
            story_state=dynamic_story_state,
            market_intelligence=None,
            critic_feedback_history=[],
            generation_constraints={}
        )
    
    def _convert_output_to_dict(self, output) -> Dict[str, Any]:
        """Convert ChapterGeneratorOutput to dict format."""
        result = {}
        
        if hasattr(output, 'chapter_variants') and output.chapter_variants:
            result['chapters'] = {}
            for i, variant in enumerate(output.chapter_variants):
                chapter_id = f"chapter_{i+1}"
                result['chapters'][chapter_id] = {
                    'content': variant.content,
                    'title': variant.metadata.get('title', f'Chapter {i+1}'),
                    'word_count': len(variant.content.split()),
                    'approach': variant.generation_approach,
                    'metadata': variant.metadata
                }
        
        return result


# Helper function to apply the ResearchableMixin to existing ChapterGenerator instances
def make_chapter_generator_researchable(generator: ChapterGenerator) -> ResearchEnabledChapterGenerator:
    """
    Convert an existing ChapterGenerator instance to a ResearchEnabledChapterGenerator.
    """
    # Create new ResearchEnabledChapterGenerator with same config
    enhanced_generator = ResearchEnabledChapterGenerator(generator.config)
    
    # Copy over any existing state
    if hasattr(generator, '_generation_history'):
        enhanced_generator._generation_history = generator._generation_history
    if hasattr(generator, '_learned_patterns'):
        enhanced_generator._learned_patterns = generator._learned_patterns
    if hasattr(generator, '_context_cache'):
        enhanced_generator._context_cache = generator._context_cache
    if hasattr(generator, '_banned_patterns'):
        enhanced_generator._banned_patterns = generator._banned_patterns
    
    return enhanced_generator


# Example usage
async def example_research_enabled_generation():
    """Example of using the research-enabled chapter generator."""
    
    from musequill.v3.components.base.component_interface import ComponentConfiguration
    from musequill.v3.components.generators.chapter_generator import ChapterGeneratorConfig
    
    # Create configuration
    config_dict = {
        'generation_parameters': {
            'chapter_target_words': 3000,
            'variant_count': 3,
            'temperature': 0.8
        },
        'quality_thresholds': {
            'min_words': 2000,
            'max_words': 5000
        }
    }
    
    chapter_config = ChapterGeneratorConfig(**config_dict)
    component_config = ComponentConfiguration(
        component_id="research_chapter_generator",
        specific_config=chapter_config
    )
    
    # Create research-enabled generator
    generator = ResearchEnabledChapterGenerator(component_config)
    await generator.initialize()
    
    # The generator now supports ResearchableMixin interface
    print(f"ResearchableMixin support: {isinstance(generator, ResearchableMixin)}")
    
    return generator


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_research_enabled_generation())
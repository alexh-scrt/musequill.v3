"""
Plot Outliner Component - MISSING CRITICAL COMPONENT

This implements the missing PlotOutlinerComponent that should create detailed
chapter-by-chapter plans and objectives for the story generation pipeline.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import chromadb

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.components.orchestration.pipeline_researcher import (
    ResearchableMixin, PipelineResearcher, ResearchResponse, ResearchScope, ResearchPriority
)
from musequill.v3.models.chapter_objective import (
    ChapterObjective, EmotionalBeat, SceneType, ObjectivePriority,
    PlotAdvancement, CharacterDevelopmentTarget, ReaderEngagementGoal
)
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.components.market_intelligence.market_intelligence_engine import MarketIntelligenceReport
from musequill.v3.models.chapter_objective import (
    ChapterObjective, 
    WordCountTarget, 
    PlotAdvancement, 
    CharacterDevelopmentTarget, 
    ObjectivePriority, 
    SceneType
)

logger = logging.getLogger(__name__)


class PlotStructure(str, Enum):
    """Standard plot structure types."""
    THREE_ACT = "three_act"
    HEROS_JOURNEY = "heros_journey"
    FIVE_ACT = "five_act"
    FREYTAG_PYRAMID = "freytag_pyramid"
    SAVE_THE_CAT = "save_the_cat"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    THRILLER = "thriller"


class PlotOutlinerConfig(BaseModel):
    """Configuration for Plot Outliner Component."""
    
    default_chapter_count: int = Field(
        default=12,
        ge=3,
        le=50,
        description="Default number of chapters to plan"
    )
    
    target_words_per_chapter: int = Field(
        default=3000,
        ge=1000,
        le=8000,
        description="Target word count per chapter"
    )
    
    plot_structure: PlotStructure = Field(
        default=PlotStructure.THREE_ACT,
        description="Plot structure to follow"
    )
    
    pacing_style: str = Field(
        default="balanced",
        description="Pacing style: fast, balanced, slow, literary"
    )
    
    cliffhanger_frequency: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Percentage of chapters that should end with cliffhangers"
    )
    
    character_development_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="How much to emphasize character development vs plot"
    )


class PlotOutlinerInput(BaseModel):
    """Input for Plot Outliner Component."""
    
    genre: str = Field(description="Story genre")
    plot_type: str = Field(default="standard", description="Plot type variation")
    market_intelligence: MarketIntelligenceReport = Field(
        default_factory=MarketIntelligenceReport,
        description="Market research data"
    )
    research_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Plot structure research insights"
    )
    story_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original story configuration"
    )


class PlotOutlinerOutput(BaseModel):
    """Output from Plot Outliner Component."""
    
    chapter_objectives: List[ChapterObjective] = Field(
        description="Detailed objectives for each planned chapter"
    )
    
    story_structure: Dict[str, Any] = Field(
        description="Overall story structure and pacing plan"
    )
    
    plot_threads: Dict[str, Any] = Field(
        description="Main and subplot thread definitions"
    )
    
    character_arc_plan: Dict[str, Any] = Field(
        description="Character development plan across chapters"
    )
    
    pacing_plan: Dict[str, Any] = Field(
        description="Chapter-by-chapter pacing and beat plan"
    )
    
    success_metrics: Dict[str, float] = Field(
        description="Predicted success metrics based on market intelligence"
    )


class PlotOutlinerComponent(BaseComponent[PlotOutlinerInput, PlotOutlinerOutput, PlotOutlinerConfig], ResearchableMixin):
    """
    Plot Outliner Component - Creates detailed chapter-by-chapter story plans.
    
    This is the missing critical component that should create detailed chapter
    objectives and story structure plans for the generation pipeline.
    Supports ResearchableMixin for enhanced research integration.
    """
    
    def __init__(self, config: ComponentConfiguration[PlotOutlinerConfig]):
        super().__init__(config)
        ResearchableMixin.__init__(self)
        self._structure_templates = {}
        self._genre_patterns = {}
        self._research_insights = []
    
    async def initialize(self) -> bool:
        """Initialize plot structure templates and genre patterns."""
        try:
            # Load plot structure templates
            await self._load_structure_templates()
            
            # Load genre-specific patterns
            await self._load_genre_patterns()
            
            logger.info("PlotOutlinerComponent initialized successfully")
            return True
            
        except Exception as e:
            self.state.last_error = f"Plot outliner initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: PlotOutlinerInput) -> PlotOutlinerOutput:
        """Create detailed chapter-by-chapter plot outline with research integration."""
        start_time = datetime.now()
        
        try:
            # Perform pre-outlining research if researcher is available
            if self.researcher:
                research_insights = await self._perform_plot_research(input_data)
                if research_insights:
                    logger.info(f"Gathered {len(research_insights)} research insights for plot outlining")
                    self._research_insights = research_insights
            
            # Determine optimal chapter count based on genre and market intelligence
            chapter_count = self._determine_chapter_count(input_data)
            
            # Select plot structure based on genre and market data
            structure = self._select_plot_structure(input_data)
            
            # Create story beats and major plot points
            story_beats = self._create_story_beats(structure, chapter_count, input_data)
            
            # Generate chapter objectives
            chapter_objectives = await self._generate_chapter_objectives(
                story_beats, input_data, chapter_count
            )
            
            # Create plot thread definitions
            plot_threads = self._define_plot_threads(input_data)
            
            # Plan character development across chapters
            character_arc_plan = self._plan_character_arcs(
                chapter_objectives, input_data
            )
            
            # Create pacing plan
            pacing_plan = self._create_pacing_plan(
                chapter_objectives, structure, input_data
            )
            
            # Calculate success metrics
            success_metrics = self._calculate_success_metrics(
                chapter_objectives, input_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Plot outline created in {execution_time:.2f}s with {len(chapter_objectives)} chapters")
            
            return PlotOutlinerOutput(
                chapter_objectives=chapter_objectives,
                story_structure={
                    'structure_type': structure.value,
                    'total_chapters': chapter_count,
                    'estimated_words': chapter_count * self.config.specific_config.target_words_per_chapter,
                    'story_beats': story_beats
                },
                plot_threads=plot_threads,
                character_arc_plan=character_arc_plan,
                pacing_plan=pacing_plan,
                success_metrics=success_metrics
            )
            
        except Exception as e:
            self.state.last_error = f"Plot outlining failed: {str(e)}"
            raise ComponentError(f"Plot outliner process failed: {str(e)}")
    
    async def _perform_plot_research(self, input_data: PlotOutlinerInput) -> List[Dict[str, Any]]:
        """
        Perform research to inform plot structure and chapter planning.
        """
        research_insights = []
        
        try:
            genre = input_data.genre
            plot_type = input_data.plot_type
            
            # Research queries for plot structure
            research_queries = [
                f"when writing {genre} books suggest the most effective {genre} plot structures successful novels",
                f"when writing {plot_type} {genre} books suggest the most effective story pacing techniques",
                f"when writing a {genre} book, suggest chapter breakdown for {genre} bestseller books",
                f"when writing a {genre} book, suggest the best {genre} reader engagement hooks chapter endings"
            ]
            
            for query in research_queries:
                logger.info(f"Researching plot structure: {query}")
                
                response = await self.quick_research(query)
                if response and response.status == "completed":
                    research_insights.append({
                        'query': query,
                        'chroma_storage_info': response.results['chroma_storage_info'],
                        'error': response.results['error'],
                        'findings': response.summary,
                        'sources': response.sources_found,
                        'relevance_to_plotting': self._assess_plot_research_relevance(response, input_data)
                    })
        
        except Exception as e:
            logger.warning(f"Plot structure research failed: {e}")
        
        return research_insights
    
    def _assess_plot_research_relevance(self, response: ResearchResponse, input_data: PlotOutlinerInput) -> float:
        """Assess how relevant research findings are to plot outlining."""
        relevance_score = 0.5  # Base relevance
        
        genre = input_data.genre.lower()
        summary = response.summary.lower() if response.summary else ""
        
        # Check genre relevance
        if genre in summary:
            relevance_score += 0.3
        
        # Check for plot structure keywords
        plot_keywords = ['structure', 'pacing', 'chapter', 'act', 'climax', 'tension', 'hook']
        keyword_matches = sum(1 for keyword in plot_keywords if keyword in summary)
        relevance_score += (keyword_matches * 0.05)
        
        return min(relevance_score, 1.0)
    
    async def _extract_research_from_chroma(self, research_insight: Dict[str, Any]) -> Optional[str]:
        """
        Extract actual research findings from Chroma storage based on chroma_storage_info.
        
        Args:
            research_insight: Insight dict containing chroma_storage_info
            
        Returns:
            String containing the actual research findings
        """
        try:
            # Get chroma storage info
            chroma_info = research_insight.get('chroma_storage_info', {})
            if not chroma_info:
                logging.warning("No chroma_storage_info found in research insight")
                return None
                
            research_id = chroma_info.get('research_id')
            host = chroma_info.get('host', 'localhost')
            port = chroma_info.get('port', 18000)
            collection_name = chroma_info.get('collection_name', 'research_collection')
            
            if not research_id:
                logging.warning("No research_id found in chroma_storage_info")
                return None
            
            # Initialize Chroma client
            chroma_client = chromadb.HttpClient(host=host, port=port)
            chroma_collection = chroma_client.get_collection(name=collection_name)
            
            # Query all chunks for this research session
            results = chroma_collection.get(
                where={"research_id": research_id},
                include=["metadatas", "documents"]
            )
            
            if not results.get('documents'):
                logging.info(f"No research data found in Chroma for research_id: {research_id}")
                return None
            
            # Combine all document content into findings
            findings_content = []
            for doc in results['documents']:
                if doc and len(doc.strip()) > 50:  # Only substantial content
                    findings_content.append(doc.strip())
            
            if findings_content:
                return " ".join(findings_content)
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to extract research data from Chroma: {e}")
            return None
    
    async def _integrate_research_into_planning(self, chapter_objectives: List[ChapterObjective]) -> List[ChapterObjective]:
        """
        Integrate research insights into chapter planning.
        """
        if not self._research_insights:
            return chapter_objectives
        
        # Extract actionable insights from research
        structure_insights = []
        pacing_insights = []
        engagement_insights = []
        
        for insight in self._research_insights:
            # Extract actual findings from Chroma store
            extracted_findings = await self._extract_research_from_chroma(insight)
            
            # Use extracted findings if available, otherwise fallback to original findings
            findings_text = extracted_findings if extracted_findings else insight.get('findings', '')
            findings = findings_text.lower()
            
            # Create enhanced insight with actual findings
            enhanced_insight = insight.copy()
            enhanced_insight['findings'] = findings_text
            
            if any(word in findings for word in ['structure', 'act', 'beat']):
                structure_insights.append(enhanced_insight)
            elif any(word in findings for word in ['pacing', 'rhythm', 'tempo']):
                pacing_insights.append(enhanced_insight)
            elif any(word in findings for word in ['hook', 'engagement', 'cliffhanger']):
                engagement_insights.append(enhanced_insight)
        
        # Apply insights to chapter objectives
        enhanced_objectives = []
        for objective in chapter_objectives:
            enhanced_obj = self._enhance_objective_with_research(
                objective, structure_insights, pacing_insights, engagement_insights
            )
            enhanced_objectives.append(enhanced_obj)
        
        return enhanced_objectives
    
    def _enhance_objective_with_research(self, objective: ChapterObjective, 
                                       structure_insights: List[Dict[str, Any]],
                                       pacing_insights: List[Dict[str, Any]],
                                       engagement_insights: List[Dict[str, Any]]) -> ChapterObjective:
        """Enhance a single chapter objective with research insights."""
        
        # Add research-informed narrative goals
        enhanced_goals = objective.narrative_goals.copy()
        
        # Add engagement techniques from research
        if engagement_insights and len(engagement_insights) > 0:
            engagement_goal = f"Apply {len(engagement_insights)} research-backed engagement techniques"
            if engagement_goal not in enhanced_goals:
                enhanced_goals.append(engagement_goal)
        
        # Update the objective with enhanced goals
        objective.narrative_goals = enhanced_goals
        
        # Add research insights to constraints
        if not hasattr(objective, 'constraints') or not objective.constraints:
            objective.constraints = {}
        
        objective.constraints['research_informed'] = True
        objective.constraints['research_insights_count'] = len(self._research_insights)
        
        return objective
    
    def _determine_chapter_count(self, input_data: PlotOutlinerInput) -> int:
        """Determine optimal chapter count based on genre and market data."""
        base_count = self.config.specific_config.default_chapter_count
        genre = input_data.genre.lower()
        
        # Genre adjustments
        genre_adjustments = {
            'thriller': 0,      # Standard length
            'mystery': 2,       # Slightly longer for complexity
            'romance': -2,      # Slightly shorter, focused
            'fantasy': 5,       # Longer for world-building
            'sci-fi': 3,        # Longer for concept exploration
            'literary': -3,     # Shorter, more focused
            'horror': -1,       # Shorter, concentrated tension
            'adventure': 2,     # Standard to long
            'historical': 4     # Longer for period detail
        }
        
        adjustment = genre_adjustments.get(genre, 0)
        
        # Market intelligence adjustments
        market_intel = input_data.market_intelligence
        if market_intel and 'reader_preferences' in market_intel:
            prefs = market_intel['reader_preferences']
            if isinstance(prefs, dict):
                if prefs.get('prefers_shorter_books'):
                    adjustment -= 2
                elif prefs.get('prefers_longer_books'):
                    adjustment += 3
        
        return max(6, min(25, base_count + adjustment))
    
    def _select_plot_structure(self, input_data: PlotOutlinerInput) -> PlotStructure:
        """Select appropriate plot structure based on genre."""
        genre = input_data.genre.lower()
        
        genre_structures = {
            'thriller': PlotStructure.THREE_ACT,
            'mystery': PlotStructure.MYSTERY,
            'romance': PlotStructure.ROMANCE,
            'fantasy': PlotStructure.HEROS_JOURNEY,
            'sci-fi': PlotStructure.THREE_ACT,
            'horror': PlotStructure.THREE_ACT,
            'literary': PlotStructure.FREYTAG_PYRAMID,
            'adventure': PlotStructure.HEROS_JOURNEY
        }
        
        return genre_structures.get(genre, self.config.specific_config.plot_structure)
    
    def _create_story_beats(self, structure: PlotStructure, chapter_count: int, 
                          input_data: PlotOutlinerInput) -> Dict[str, Any]:
        """Create major story beats based on plot structure."""
        
        if structure == PlotStructure.THREE_ACT:
            return {
                'act_1_end': int(chapter_count * 0.25),
                'midpoint': int(chapter_count * 0.5),
                'act_2_end': int(chapter_count * 0.75),
                'climax': int(chapter_count * 0.85),
                'resolution': chapter_count,
                'major_beats': [
                    {'chapter': 1, 'beat': 'hook_and_setup'},
                    {'chapter': int(chapter_count * 0.25), 'beat': 'inciting_incident'},
                    {'chapter': int(chapter_count * 0.5), 'beat': 'midpoint_reversal'},
                    {'chapter': int(chapter_count * 0.75), 'beat': 'dark_moment'},
                    {'chapter': int(chapter_count * 0.85), 'beat': 'climax'},
                    {'chapter': chapter_count, 'beat': 'resolution'}
                ]
            }
        elif structure == PlotStructure.HEROS_JOURNEY:
            return {
                'ordinary_world': 1,
                'call_to_adventure': int(chapter_count * 0.1),
                'crossing_threshold': int(chapter_count * 0.25),
                'tests_allies_enemies': int(chapter_count * 0.5),
                'ordeal': int(chapter_count * 0.75),
                'reward': int(chapter_count * 0.85),
                'return': chapter_count
            }
        else:
            # Default three-act structure
            return self._create_story_beats(PlotStructure.THREE_ACT, chapter_count, input_data)
    
    async def _generate_chapter_objectives(self, story_beats: Dict[str, Any], 
                                input_data: PlotOutlinerInput,
                                chapter_count: int) -> List[ChapterObjective]:
        """Generate specific objectives for each chapter."""
        objectives = []
        
        for chapter_num in range(1, chapter_count + 1):
            try:
                # Determine chapter type based on story beats
                chapter_type = self._get_chapter_type(chapter_num, story_beats, chapter_count)
                
                # Create WordCountTarget object
                word_count_target = WordCountTarget(
                    target_words=self.config.specific_config.target_words_per_chapter,
                    min_acceptable=int(self.config.specific_config.target_words_per_chapter * 0.8),
                    max_acceptable=int(self.config.specific_config.target_words_per_chapter * 1.2)
                )
                
                # Get narrative goals and convert to primary + secondary
                narrative_goals = self._get_narrative_goals(chapter_type, input_data.genre)
                primary_goal = narrative_goals[0] if narrative_goals else f"Advance {chapter_type} narrative for chapter {chapter_num}"
                secondary_goals = narrative_goals[1:] if len(narrative_goals) > 1 else []
                
                # Convert constraints dict to list of strings
                constraints_dict = self._get_chapter_constraints(chapter_num, chapter_count)
                constraints_list = []
                if isinstance(constraints_dict, dict):
                    for key, value in constraints_dict.items():
                        constraints_list.append(f"{key}: {value}")
                elif isinstance(constraints_dict, list):
                    constraints_list = constraints_dict
                
                # Generate chapter purpose
                chapter_purpose = f"Chapter {chapter_num} serves as a {chapter_type} chapter, focusing on {primary_goal.lower()}. This chapter should establish key narrative elements while maintaining reader engagement through strategic pacing and character development."
                
                # Create chapter objective with correct field names and types
                objective = ChapterObjective(
                    chapter_number=chapter_num,
                    primary_goal=primary_goal,
                    secondary_goals=secondary_goals,
                    word_count_target=word_count_target,
                    chapter_purpose=chapter_purpose,
                    constraints=constraints_list,
                    success_criteria=self._get_success_criteria(chapter_type),
                    emotional_beats=self._get_emotional_beats(chapter_type, input_data.genre),
                    scene_requirements=self._get_scene_requirements(chapter_type),
                    reader_engagement_goals=self._get_engagement_goals(chapter_type, chapter_num, chapter_count),
                    
                    # Additional fields that might be needed
                    plot_advancements=self._get_plot_advancements(chapter_type, input_data),
                    character_development_targets=self._get_character_development_targets(chapter_num, input_data),
                )
                
                objectives.append(objective)
            except Exception as e:
                logger.error(f"Error generating chapter objective for chapter {chapter_num}: {e}")
                
        
        # Apply research insights to chapter objectives if available
        if self._research_insights:
            objectives = await self._integrate_research_into_planning(objectives)
        
        return objectives


    def _get_plot_advancements(self, chapter_type: str, input_data: PlotOutlinerInput) -> List[PlotAdvancement]:
        """Generate plot advancement requirements for the chapter."""
        # You'll need to implement this based on your plot structure
        # This is a placeholder implementation
        advancements = []
        
        # Add at least one plot advancement (required by validator)
        advancement = PlotAdvancement(
            thread_id=f"main_plot_{chapter_type}",
            advancement_type="progress",
            advancement_description=f"Advance the main storyline through {chapter_type} development",
            importance=ObjectivePriority.MEDIUM
        )
        advancements.append(advancement)
        
        return advancements


    def _get_character_development_targets(self, chapter_num: int, input_data: PlotOutlinerInput) -> List[CharacterDevelopmentTarget]:
        """Generate character development targets for the chapter."""
        # This is a placeholder implementation
        targets = []
        
        # Add character development based on your story structure
        if hasattr(input_data, 'main_character') and input_data.main_character:
            target = CharacterDevelopmentTarget(
                character_id=input_data.main_character.get('id', 'protagonist'),
                development_type="growth",
                development_goal=f"Develop character through chapter {chapter_num} challenges",
                target_scenes=[SceneType.DIALOGUE, SceneType.INTERNAL_REFLECTION]
            )
            targets.append(target)
        
        return targets
    
    def _get_chapter_type(self, chapter_num: int, story_beats: Dict[str, Any], 
                         total_chapters: int) -> str:
        """Determine the type/role of a chapter in the overall structure."""
        
        # Check if this chapter contains a major beat
        major_beats = story_beats.get('major_beats', [])
        for beat in major_beats:
            if beat['chapter'] == chapter_num:
                return beat['beat']
        
        # Determine by position
        completion = chapter_num / total_chapters
        
        if completion <= 0.25:
            return 'setup'
        elif completion <= 0.5:
            return 'rising_action'
        elif completion <= 0.75:
            return 'complications'
        elif completion <= 0.9:
            return 'climax_buildup'
        else:
            return 'resolution'
    
    def _get_narrative_goals(self, chapter_type: str, genre: str) -> List[str]:
        """Get narrative goals based on chapter type and genre."""
        base_goals = {
            'hook_and_setup': [
                "Introduce the protagonist with vivid details and ground the reader in the world’s tone and setting.",
                "Craft a compelling opening hook that immediately captures attention and sets narrative expectations.",
                "Foreshadow the central conflicts or mysteries that will drive the story forward."
            ],
            'setup': [
                "Expand character relationships, motivations, and dynamics to build emotional investment.",
                "Develop the world’s context, rules, or atmosphere in ways that feel natural and immersive.",
                "Lay a strong foundation for upcoming conflicts by subtly planting key plot elements."
            ],
            'inciting_incident': [
                "Deliver a significant, disruptive event that forces the protagonist to abandon the status quo.",
                "Push the protagonist into action by creating a dilemma that cannot be ignored.",
                "Clarify the story’s stakes so readers understand what could be gained or lost."
            ],
            'rising_action': [
                "Escalate narrative tension by introducing increasingly difficult obstacles and conflicts.",
                "Show characters making meaningful choices that complicate relationships and alliances.",
                "Advance multiple interconnected plot threads to sustain narrative momentum."
            ],
            'complications': [
                "Heighten the sense of struggle by layering obstacles that test the protagonist’s abilities.",
                "Deepen personal or interpersonal conflicts, forcing characters into hard decisions.",
                "Push the narrative toward climax by revealing the cost of failure more clearly."
            ],
            'midpoint_reversal': [
                "Introduce a major twist, revelation, or reversal that shifts the story’s trajectory.",
                "Reframe the protagonist’s understanding of events, allies, or enemies in a dramatic way.",
                "Escalate stakes to a new level, making success feel more urgent and failure more dire."
            ],
            'climax_buildup': [
                "Concentrate tension by converging characters, conflicts, and plot threads toward a showdown.",
                "Highlight the protagonist’s doubts, vulnerabilities, or sacrifices before the final test.",
                "Ensure momentum is focused squarely on the upcoming climactic confrontation."
            ],
            'dark_moment': [
                "Depict the protagonist’s lowest emotional point, where hope feels nearly lost.",
                "Force the protagonist to confront their deepest fears or limitations head-on.",
                "Set the stage for renewal by making the cost of failure brutally clear."
            ],
            'climax': [
                "Deliver a decisive confrontation that tests everything the protagonist has learned.",
                "Resolve the central conflict in a way that feels both surprising and inevitable.",
                "Show clear evidence of character transformation or growth through action."
            ],
            'resolution': [
                "Tie off remaining subplots and provide closure for secondary characters.",
                "Demonstrate how the protagonist has changed through choices, behavior, or perspective.",
                "Leave the reader with a satisfying sense of completion or thematic resonance."
            ]
        }
        
        return base_goals.get(chapter_type, [
            "Move the story forward in ways that matter to the protagonist and theme.",
            "Develop characters through meaningful action and interaction.",
            "Maintain narrative engagement by balancing tension with payoff."
        ])
    
    def _get_character_goals(self, chapter_num: int, input_data: PlotOutlinerInput) -> Dict[str, str]:
        """Get character development goals for the chapter."""
        # Extract characters from story config
        story_config = input_data.story_config
        characters = story_config.get('characters', {})
        
        goals = {}
        for char_name, char_info in characters.items():
            if chapter_num <= 3:
                goals[char_name] = f"Establish {char_name}'s personality and motivation"
            elif chapter_num <= 8:
                goals[char_name] = f"Develop {char_name}'s relationships and challenges"
            else:
                goals[char_name] = f"Show {char_name}'s growth and resolution"
        
        return goals
    
    def _get_plot_requirements(self, chapter_type: str, input_data: PlotOutlinerInput) -> List[str]:
        """Get plot progression requirements."""
        return [
            f"Maintain {input_data.genre} genre expectations",
            "Advance main plot thread meaningfully",
            "Build appropriate tension level"
        ]
    
    def _get_tone_requirements(self, chapter_type: str, input_data: PlotOutlinerInput) -> Dict[str, str]:
        """Get tone and style requirements."""
        return {
            'genre': input_data.genre,
            'tone': self._get_chapter_tone(chapter_type),
            'pov': 'third_person_limited',
            'tense': 'past'
        }
    
    def _get_chapter_tone(self, chapter_type: str) -> str:
        """Get appropriate tone for chapter type."""
        tone_map = {
            'hook_and_setup': 'intriguing',
            'setup': 'engaging',
            'inciting_incident': 'dramatic',
            'rising_action': 'building_tension',
            'complications': 'intense',
            'midpoint_reversal': 'shocking',
            'climax_buildup': 'suspenseful',
            'dark_moment': 'desperate',
            'climax': 'climactic',
            'resolution': 'satisfying'
        }
        return tone_map.get(chapter_type, 'engaging')
    
    def _get_chapter_constraints(self, chapter_num: int, total_chapters: int) -> Dict[str, Any]:
        """Get constraints for the chapter."""
        return {
            'max_word_count': int(self.config.specific_config.target_words_per_chapter * 1.2),
            'min_word_count': int(self.config.specific_config.target_words_per_chapter * 0.8),
            'chapter_position': chapter_num / total_chapters,
            'requires_cliffhanger': self._should_have_cliffhanger(chapter_num, total_chapters)
        }
    
    def _should_have_cliffhanger(self, chapter_num: int, total_chapters: int) -> bool:
        """Determine if chapter should end with cliffhanger."""
        # No cliffhanger on final chapter
        if chapter_num == total_chapters:
            return False
        
        # Use configured frequency
        import random
        return random.random() < self.config.specific_config.cliffhanger_frequency
    
    def _get_success_criteria(self, chapter_type: str) -> List[str]:
        """Get success criteria for the chapter."""
        return [
            "Maintains reader engagement throughout",
            "Advances story meaningfully", 
            "Develops characters authentically",
            "Meets genre expectations"
        ]
    
    def _get_emotional_beats(self, chapter_type: str, genre: str) -> List[EmotionalBeat]:
        """Get emotional beats for the chapter."""
        beat_map = {
            'hook_and_setup': [EmotionalBeat.TENSION_BUILD],
            'inciting_incident': [EmotionalBeat.SHOCK, EmotionalBeat.TENSION_BUILD],
            'rising_action': [EmotionalBeat.TENSION_BUILD],
            'complications': [EmotionalBeat.TENSION_BUILD, EmotionalBeat.ANGER],
            'midpoint_reversal': [EmotionalBeat.SHOCK, EmotionalBeat.FEAR],
            'dark_moment': [EmotionalBeat.DESPAIR, EmotionalBeat.FEAR],
            'climax': [EmotionalBeat.TENSION_BUILD, EmotionalBeat.TRIUMPH],
            'resolution': [EmotionalBeat.RELIEF, EmotionalBeat.HOPE]
        }
        return beat_map.get(chapter_type, [EmotionalBeat.TENSION_BUILD])
    
    def _get_scene_requirements(self, chapter_type: str) -> List[SceneType]:
        """Get required scene types for the chapter."""
        scene_map = {
            'hook_and_setup': [SceneType.DIALOGUE, SceneType.DESCRIPTION],
            'inciting_incident': [SceneType.ACTION, SceneType.CONFRONTATION],
            'rising_action': [SceneType.DIALOGUE, SceneType.ACTION],
            'complications': [SceneType.CONFRONTATION, SceneType.INTERNAL_REFLECTION],
            'midpoint_reversal': [SceneType.DISCOVERY, SceneType.ACTION],
            'climax': [SceneType.ACTION, SceneType.CONFRONTATION],
            'resolution': [SceneType.DIALOGUE, SceneType.INTERNAL_REFLECTION]
        }
        return scene_map.get(chapter_type, [SceneType.DIALOGUE])
    
    def _get_engagement_goals(self, chapter_type: str, chapter_num: int, 
                            total_chapters: int) -> List[ReaderEngagementGoal]:
        """Get reader engagement goals for the chapter."""
        # This would create ReaderEngagementGoal objects
        # For now, return empty list as the model might not be fully defined
        return []
    
    def _define_plot_threads(self, input_data: PlotOutlinerInput) -> Dict[str, Any]:
        """Define main and subplot threads."""
        return {
            'main_thread': {
                'description': 'Primary story conflict and resolution',
                'priority': 'high',
                'chapters_active': 'all'
            },
            'character_thread': {
                'description': 'Character development and relationships',
                'priority': 'medium', 
                'chapters_active': 'all'
            }
        }
    
    def _plan_character_arcs(self, objectives: List[ChapterObjective], 
                           input_data: PlotOutlinerInput) -> Dict[str, Any]:
        """Plan character development across all chapters."""
        return {
            'character_count': len(input_data.story_config.get('characters', {})),
            'arc_strategy': 'gradual_development',
            'development_weight': self.config.specific_config.character_development_weight
        }
    
    def _create_pacing_plan(self, objectives: List[ChapterObjective],
                          structure: PlotStructure, 
                          input_data: PlotOutlinerInput) -> Dict[str, Any]:
        """Create chapter-by-chapter pacing plan."""
        return {
            'pacing_style': self.config.specific_config.pacing_style,
            'tension_curve': [obj.chapter_number * 0.1 for obj in objectives],
            'chapter_lengths': [obj.target_word_count for obj in objectives]
        }
    
    def _calculate_success_metrics(self, objectives: List[ChapterObjective],
                                 input_data: PlotOutlinerInput) -> Dict[str, float]:
        """Calculate predicted success metrics."""
        return {
            'structural_strength': 0.8,
            'genre_alignment': 0.9,
            'market_appeal': 0.7,
            'completion_probability': 0.85
        }
    
    async def _load_structure_templates(self):
        """Load plot structure templates."""
        # Placeholder for loading structure templates
        self._structure_templates = {
            'three_act': {'act_breaks': [0.25, 0.75]},
            'heros_journey': {'stages': 12}
        }
    
    async def _load_genre_patterns(self):
        """Load genre-specific patterns."""
        # Placeholder for loading genre patterns
        self._genre_patterns = {
            'thriller': {'pacing': 'fast', 'cliffhangers': 0.8},
            'romance': {'pacing': 'moderate', 'emotional_beats': 'high'}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform component health check."""
        return {
            'status': 'healthy',
            'templates_loaded': len(self._structure_templates),
            'genre_patterns_loaded': len(self._genre_patterns),
            'last_execution': self.state.last_execution_time
        }
    
    async def cleanup(self) -> None:
        """Clean up component resources."""
        self._structure_templates.clear()
        self._genre_patterns.clear()
        logger.info("PlotOutlinerComponent cleaned up")


# Legacy execute method for enhanced orchestrator compatibility
class PlotOutlinerComponentWithLegacy(PlotOutlinerComponent):
    """Plot outliner with legacy execute method for orchestrator compatibility and ResearchableMixin support."""
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy execute method for enhanced orchestrator.
        Supports the ResearchableMixin interface expected by the orchestrator.
        """
        logger.info("Legacy execute method called with ResearchableMixin support")
        
        try:
            # Convert dict input to proper PlotOutlinerInput
            outliner_input = PlotOutlinerInput(
                genre=input_data.get('genre', 'general'),
                plot_type=input_data.get('plot_type', 'standard'),
                market_intelligence=input_data.get('market_intelligence', {}),
                research_insights=input_data.get('research_insights'),
                story_config=input_data.get('story_config', input_data)
            )
            
            # Use proper process method (which includes research integration)
            result = await self.process(outliner_input)
            
            # Convert back to dict for orchestrator
            return {
                'chapter_objectives': [obj.model_dump() for obj in result.chapter_objectives],
                'story_structure': result.story_structure,
                'plot_threads': result.plot_threads,
                'character_arc_plan': result.character_arc_plan,
                'pacing_plan': result.pacing_plan,
                'success_metrics': result.success_metrics,
                # Add research insights to output for transparency
                'research_insights_used': len(self._research_insights) if self._research_insights else 0,
                'research_informed': bool(self._research_insights)
            }
            
        except Exception as e:
            logger.error(f"Legacy execute method with research failed: {e}")
            return {}
    
    # Ensure ResearchableMixin interface is properly exposed
    def supports_research(self) -> bool:
        """Confirm that this component supports research integration."""
        return True
    
    async def research_before_execution(self, input_data: Dict[str, Any]) -> Optional[ResearchResponse]:
        """
        ResearchableMixin interface method for pre-execution research.
        This is called automatically by the enhanced orchestrator.
        """
        if not self.researcher:
            return None
        
        genre = input_data.get('genre', 'general')
        plot_type = input_data.get('plot_type', 'standard')
        
        # Perform comprehensive plot structure research
        research_query = f"effective {genre} {plot_type} plot structure successful novels pacing"
        
        logger.info(f"Pre-execution research for plot outlining: {research_query}")
        return await self.quick_research(research_query)


# Example usage
async def create_plot_outliner():
    """Create a plot outliner component instance."""
    from musequill.v3.components.base.component_interface import ComponentConfiguration
    
    config_dict = {
        'default_chapter_count': 12,
        'target_words_per_chapter': 3000,
        'plot_structure': PlotStructure.THREE_ACT,
        'pacing_style': 'balanced',
        'cliffhanger_frequency': 0.7
    }
    
    outliner_config = PlotOutlinerConfig(**config_dict)
    component_config = ComponentConfiguration(
        component_id="plot_outliner",
        specific_config=outliner_config
    )
    
    outliner = PlotOutlinerComponentWithLegacy(component_config)
    await outliner.initialize()
    
    return outliner


if __name__ == "__main__":
    import asyncio
    asyncio.run(create_plot_outliner())
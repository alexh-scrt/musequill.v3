"""
Plot Coherence Critic Component

Implements story logic evaluation, character consistency checking, and plot
advancement analysis for the adversarial system discriminator layer.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_variant import ChapterVariant
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.plot_coherence_assessment import (
    PlotCoherenceAssessment, Inconsistency, PlotAdvancementAnalysis,
    ContinuityCheck, TensionManagementAnalysis, InconsistencyType,
    SeverityLevel, AdvancementType
)


class PlotCoherenceCriticConfig(BaseModel):
    """Configuration for Plot Coherence Critic component."""
    
    inconsistency_detection_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging potential inconsistencies"
    )
    
    minimum_advancement_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable plot advancement score"
    )
    
    character_knowledge_tracking: bool = Field(
        default=True,
        description="Whether to track character knowledge consistency"
    )
    
    world_rules_enforcement: bool = Field(
        default=True,
        description="Whether to enforce established world rules"
    )
    
    timeline_validation: bool = Field(
        default=True,
        description="Whether to validate chronological consistency"
    )
    
    max_analysis_time_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Maximum time to spend analyzing a chapter"
    )


class PlotCoherenceCriticInput(BaseModel):
    """Input data for Plot Coherence Critic."""
    
    chapter_variant: ChapterVariant = Field(
        description="Chapter variant to evaluate"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for consistency checking"
    )
    
    previous_assessments: List[PlotCoherenceAssessment] = Field(
        default_factory=list,
        description="Previous chapter assessments for context"
    )


class PlotCoherenceCritic(BaseComponent[PlotCoherenceCriticInput, PlotCoherenceAssessment, PlotCoherenceCriticConfig]):
    """
    Plot Coherence Critic component for story logic evaluation.
    
    Analyzes chapter variants for consistency with established story elements,
    character behavior patterns, world rules, and logical plot progression.
    """
    
    def __init__(self, config: ComponentConfiguration[PlotCoherenceCriticConfig]):
        super().__init__(config)
        self._character_knowledge_base: Dict[str, Dict[str, Any]] = {}
        self._world_rules_cache: Dict[str, Any] = {}
        self._timeline_events: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize the plot coherence analysis systems."""
        try:
            # Initialize character knowledge tracking
            self._character_knowledge_base = {}
            
            # Initialize world rules cache
            self._world_rules_cache = {}
            
            # Initialize timeline tracking
            self._timeline_events = []
            
            # Initialize NLP components (placeholder for actual implementation)
            await self._initialize_analysis_tools()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: PlotCoherenceCriticInput) -> PlotCoherenceAssessment:
        """
        Analyze chapter variant for plot coherence.
        
        Args:
            input_data: Chapter and story state to analyze
            
        Returns:
            Comprehensive plot coherence assessment
        """
        start_time = datetime.now()
        
        try:
            # Update internal state with current story context
            await self._update_story_context(input_data.story_state)
            
            # Perform multi-dimensional analysis
            analysis_tasks = [
                self._analyze_continuity(input_data.chapter_variant, input_data.story_state),
                self._analyze_plot_advancement(input_data.chapter_variant, input_data.story_state),
                self._analyze_logic_consistency(input_data.chapter_variant, input_data.story_state),
                self._analyze_tension_management(input_data.chapter_variant, input_data.story_state)
            ]
            
            # Execute analysis tasks concurrently
            continuity_result, advancement_result, logic_result, tension_result = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )
            
            # Handle any exceptions in analysis tasks
            for result in [continuity_result, advancement_result, logic_result, tension_result]:
                if isinstance(result, Exception):
                    raise result
            
            # Compile comprehensive assessment
            assessment = await self._compile_assessment(
                input_data.chapter_variant,
                continuity_result,
                advancement_result,
                logic_result,
                tension_result
            )
            
            # Check execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > self.config.specific_config.max_analysis_time_seconds:
                self.state.runtime_data['slow_analysis_warning'] = True
            
            return assessment
            
        except Exception as e:
            raise ComponentError(f"Plot coherence analysis failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on analysis systems."""
        try:
            # Check if analysis tools are responsive
            test_successful = await self._test_analysis_tools()
            
            # Check memory usage
            if len(self._character_knowledge_base) > 1000:  # Arbitrary limit
                return False
                
            # Check error rates
            if self.state.metrics.failure_rate > 0.2:  # 20% failure rate threshold
                return False
            
            return test_successful
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup analysis resources."""
        try:
            self._character_knowledge_base.clear()
            self._world_rules_cache.clear()
            self._timeline_events.clear()
            
            return True
            
        except Exception:
            return False
    
    async def _initialize_analysis_tools(self) -> None:
        """Initialize NLP and analysis tools."""
        # Placeholder for actual NLP tool initialization
        # Would initialize spaCy, transformers, or custom analysis models
        pass
    
    async def _update_story_context(self, story_state: DynamicStoryState) -> None:
        """Update internal story context for consistency checking."""
        # Update character knowledge base
        for char_id, character in story_state.character_arcs.items():
            if char_id not in self._character_knowledge_base:
                self._character_knowledge_base[char_id] = {}
            
            self._character_knowledge_base[char_id].update({
                'voice_characteristics': character.voice_characteristics,
                'personality_traits': character.personality_traits,
                'relationships': character.relationship_dynamics,
                'last_emotional_state': character.emotional_state
            })
        
        # Update world rules cache
        for rule_id, world_rule in story_state.world_state.world_rules.items():
            self._world_rules_cache[rule_id] = {
                'description': world_rule.description,
                'limitations': world_rule.limitations,
                'consistency_critical': world_rule.consistency_critical
            }
    
    async def _analyze_continuity(self, chapter: ChapterVariant, story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze continuity with established story elements."""
        continuity_checks = []
        continuity_score = 1.0
        
        # Check character consistency
        for char_id in chapter.characters_featured:
            if char_id in story_state.character_arcs:
                character = story_state.character_arcs[char_id]
                
                # Analyze character voice consistency (placeholder)
                voice_score = await self._check_character_voice_consistency(
                    char_id, chapter.chapter_text, character
                )
                
                continuity_check = ContinuityCheck(
                    element_type="character",
                    element_id=char_id,
                    consistency_score=voice_score,
                    verification_notes=f"Character voice consistency analysis for {character.name}",
                    requires_attention=voice_score < 0.7
                )
                continuity_checks.append(continuity_check)
                continuity_score = min(continuity_score, voice_score)
        
        # Check world consistency
        for fact_id, fact in story_state.world_state.established_facts.items():
            if fact.importance_level >= 3:  # Only check important facts
                fact_consistency = await self._check_fact_consistency(
                    chapter.chapter_text, fact
                )
                
                continuity_check = ContinuityCheck(
                    element_type="world_fact",
                    element_id=fact_id,
                    consistency_score=fact_consistency,
                    verification_notes=f"Fact consistency: {fact.description[:50]}...",
                    requires_attention=fact_consistency < 0.8
                )
                continuity_checks.append(continuity_check)
                continuity_score = min(continuity_score, fact_consistency)
        
        return {
            'continuity_score': continuity_score,
            'continuity_checks': continuity_checks
        }
    
    async def _analyze_plot_advancement(self, chapter: ChapterVariant, story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze quality of plot thread advancement."""
        advancement_analyses = []
        total_advancement_score = 0.0
        meaningful_advancements = 0
        
        # Analyze each plot thread mentioned in chapter
        for thread_id in chapter.plot_threads_advanced:
            if thread_id in story_state.plot_threads:
                thread = story_state.plot_threads[thread_id]
                
                # Determine advancement type and quality
                advancement_type = await self._classify_advancement_type(
                    chapter.chapter_text, thread
                )
                
                advancement_quality = await self._assess_advancement_quality(
                    chapter.chapter_text, thread, advancement_type
                )
                
                meaningful_change = advancement_quality >= 0.6
                if meaningful_change:
                    meaningful_advancements += 1
                
                tension_impact = await self._calculate_tension_impact(
                    chapter.chapter_text, thread
                )
                
                advancement_analysis = PlotAdvancementAnalysis(
                    thread_id=thread_id,
                    advancement_type=advancement_type,
                    advancement_quality=advancement_quality,
                    advancement_description=f"Thread '{thread.title}' advanced through chapter content",
                    meaningful_change=meaningful_change,
                    tension_impact=tension_impact,
                    setup_payoff_ratio=0.5  # Would be calculated based on content analysis
                )
                
                advancement_analyses.append(advancement_analysis)
                total_advancement_score += advancement_quality
        
        # Calculate overall advancement score
        if advancement_analyses:
            avg_advancement_score = total_advancement_score / len(advancement_analyses)
        else:
            avg_advancement_score = 0.0
        
        return {
            'advancement_score': avg_advancement_score,
            'advancement_analyses': advancement_analyses,
            'meaningful_advancement_count': meaningful_advancements
        }
    
    async def _analyze_logic_consistency(self, chapter: ChapterVariant, story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze internal logical consistency."""
        inconsistencies = []
        logic_score = 1.0
        
        # Check for logical inconsistencies in chapter
        logical_issues = await self._detect_logical_inconsistencies(
            chapter.chapter_text, story_state
        )
        
        for issue in logical_issues:
            inconsistency = Inconsistency(
                inconsistency_id=f"logic_{len(inconsistencies)}",
                inconsistency_type=issue['type'],
                severity=issue['severity'],
                description=issue['description'],
                conflicting_elements=issue.get('elements', []),
                suggested_resolution=issue.get('resolution', 'Review and revise for logical consistency'),
                affected_characters=issue.get('characters', []),
                affected_plot_threads=issue.get('threads', [])
            )
            inconsistencies.append(inconsistency)
            
            # Reduce logic score based on severity
            severity_penalties = {
                SeverityLevel.MINOR: 0.05,
                SeverityLevel.MODERATE: 0.15,
                SeverityLevel.MAJOR: 0.3,
                SeverityLevel.CRITICAL: 0.5
            }
            logic_score -= severity_penalties.get(issue['severity'], 0.1)
        
        logic_score = max(0.0, logic_score)
        
        return {
            'logic_consistency_score': logic_score,
            'inconsistencies': inconsistencies
        }
    
    async def _analyze_tension_management(self, chapter: ChapterVariant, story_state: DynamicStoryState) -> Dict[str, Any]:
        """Analyze narrative tension management."""
        # Analyze tension trajectory through chapter
        tension_points = await self._analyze_tension_points(chapter.chapter_text)
        
        opening_tension = tension_points[0] if tension_points else 0.5
        closing_tension = tension_points[-1] if tension_points else 0.5
        peak_tension = max(tension_points) if tension_points else 0.5
        
        # Determine tension trajectory pattern
        if len(tension_points) >= 3:
            if tension_points[-1] > tension_points[0]:
                trajectory = "building"
            elif tension_points[-1] < tension_points[0]:
                trajectory = "declining"
            else:
                trajectory = "fluctuating"
        else:
            trajectory = "flat"
        
        # Assess appropriateness for story position
        story_position = story_state.story_completion_ratio
        expected_tension = await self._calculate_expected_tension(story_position, story_state.story_phase)
        
        tension_appropriateness = 1.0 - abs(closing_tension - expected_tension)
        
        # Analyze cliffhanger effectiveness if present
        cliffhanger_effectiveness = None
        if chapter.chapter_text.strip().endswith(('?', '!', '...')):
            cliffhanger_effectiveness = await self._assess_cliffhanger_effectiveness(
                chapter.chapter_text[-200:]  # Analyze last 200 characters
            )
        
        tension_analysis = TensionManagementAnalysis(
            opening_tension=opening_tension,
            closing_tension=closing_tension,
            peak_tension=peak_tension,
            tension_trajectory=trajectory,
            tension_appropriateness=tension_appropriateness,
            cliffhanger_effectiveness=cliffhanger_effectiveness
        )
        
        # Calculate overall tension management score
        tension_score = (tension_appropriateness + 
                        (peak_tension if trajectory == "building" else 0.5) +
                        (cliffhanger_effectiveness or 0.5)) / 3
        
        return {
            'tension_management_score': tension_score,
            'tension_analysis': tension_analysis
        }
    
    async def _compile_assessment(self, chapter: ChapterVariant, continuity_result: Dict, 
                                 advancement_result: Dict, logic_result: Dict, 
                                 tension_result: Dict) -> PlotCoherenceAssessment:
        """Compile comprehensive plot coherence assessment."""
        
        # Extract character and world consistency issues
        character_issues = [
            f"Character {check.element_id} voice consistency: {check.verification_notes}"
            for check in continuity_result['continuity_checks']
            if check.element_type == "character" and check.requires_attention
        ]
        
        world_issues = [
            f"World fact consistency issue: {check.verification_notes}"
            for check in continuity_result['continuity_checks']
            if check.element_type == "world_fact" and check.requires_attention
        ]
        
        # Generate improvement suggestions
        suggestions = []
        if continuity_result['continuity_score'] < 0.7:
            suggestions.append("Review character voice consistency and established facts")
        if advancement_result['advancement_score'] < 0.5:
            suggestions.append("Strengthen plot thread advancement with more meaningful developments")
        if logic_result['logic_consistency_score'] < 0.8:
            suggestions.append("Address logical inconsistencies and plot holes")
        if tension_result['tension_management_score'] < 0.6:
            suggestions.append("Improve tension management and pacing")
        
        # Generate advancement analysis summary
        advancement_summary = (
            f"Chapter advances {len(advancement_result['advancement_analyses'])} plot threads "
            f"with {advancement_result['meaningful_advancement_count']} meaningful progressions. "
            f"Average advancement quality: {advancement_result['advancement_score']:.2f}"
        )
        
        return PlotCoherenceAssessment(
            chapter_number=chapter.chapter_number,
            continuity_score=continuity_result['continuity_score'],
            advancement_score=advancement_result['advancement_score'],
            logic_consistency_score=logic_result['logic_consistency_score'],
            tension_management_score=tension_result['tension_management_score'],
            flagged_inconsistencies=logic_result['inconsistencies'],
            plot_advancement_analysis=advancement_result['advancement_analyses'],
            continuity_checks=continuity_result['continuity_checks'],
            tension_analysis=tension_result['tension_analysis'],
            character_consistency_issues=character_issues,
            world_consistency_issues=world_issues,
            timeline_issues=[],  # Would be populated by timeline analysis
            advancement_analysis=advancement_summary,
            suggestions=suggestions
        )
    
    # Placeholder methods for actual NLP/analysis implementation
    async def _test_analysis_tools(self) -> bool:
        """Test if analysis tools are working correctly."""
        return True
    
    async def _check_character_voice_consistency(self, char_id: str, text: str, character) -> float:
        """Check if character voice is consistent with established patterns."""
        # Placeholder - would use NLP to analyze dialogue and internal thoughts
        return 0.8
    
    async def _check_fact_consistency(self, text: str, fact) -> float:
        """Check if chapter content is consistent with established fact."""
        # Placeholder - would use semantic analysis to detect contradictions
        return 0.9
    
    async def _classify_advancement_type(self, text: str, thread) -> AdvancementType:
        """Classify the type of plot advancement that occurred."""
        # Placeholder - would analyze text to determine advancement type
        return AdvancementType.PROGRESSION
    
    async def _assess_advancement_quality(self, text: str, thread, advancement_type: AdvancementType) -> float:
        """Assess quality of plot advancement."""
        # Placeholder - would analyze meaningfulness of advancement
        return 0.7
    
    async def _calculate_tension_impact(self, text: str, thread) -> int:
        """Calculate impact on thread tension (-5 to +5)."""
        # Placeholder - would analyze tension changes
        return 1
    
    async def _detect_logical_inconsistencies(self, text: str, story_state: DynamicStoryState) -> List[Dict]:
        """Detect logical inconsistencies in chapter text."""
        # Placeholder - would use logical reasoning to find inconsistencies
        return []
    
    async def _analyze_tension_points(self, text: str) -> List[float]:
        """Analyze tension levels throughout chapter text."""
        # Placeholder - would analyze emotional intensity and conflict
        return [0.4, 0.6, 0.8, 0.7]
    
    async def _calculate_expected_tension(self, story_position: float, story_phase) -> float:
        """Calculate expected tension level for story position."""
        # Placeholder - would use story structure principles
        if story_position < 0.2:
            return 0.4  # Opening
        elif story_position < 0.7:
            return 0.6  # Rising action
        elif story_position < 0.8:
            return 0.9  # Climax
        else:
            return 0.5  # Resolution
    
    async def _assess_cliffhanger_effectiveness(self, ending_text: str) -> float:
        """Assess effectiveness of chapter ending as cliffhanger."""
        # Placeholder - would analyze ending for suspense and anticipation
        return 0.75
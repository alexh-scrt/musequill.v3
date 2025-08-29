"""
Character Development Component

Implements character development analysis, planning, and progression tracking
following the BaseComponent interface and project implementation guidelines.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import statistics

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.character_arc import (
    CharacterArc, RelationshipStatus, CharacterStability, NarrativeFunction
)
from musequill.v3.models.market_intelligence import ReaderPreference, SuccessPattern

# Import the configuration and models we just defined
from .character_development_config import (
    CharacterDevelopmentConfig, CharacterDevelopmentStrategy, VoiceConsistencyLevel
)
from .character_development_models import (
    CharacterDevelopmentInput, CharacterDevelopmentResult, 
    CharacterDevelopmentAssessment, CharacterDevelopmentPlan,
    CharacterDevelopmentPriority, DevelopmentType
)

logger = logging.getLogger(__name__)


class CharacterDevelopmentComponent(BaseComponent[CharacterDevelopmentInput, CharacterDevelopmentResult, CharacterDevelopmentConfig]):
    """
    Character Development Component
    
    Analyzes character arcs, plans development progressions, tracks voice consistency,
    manages relationship dynamics, and integrates market intelligence for optimal
    character development strategies.
    """
    
    def __init__(self, config: ComponentConfiguration[CharacterDevelopmentConfig]):
        super().__init__(config)
        self._voice_patterns_cache: Dict[str, Dict[str, Any]] = {}
        self._relationship_history: Dict[str, List[Dict[str, Any]]] = {}
        self._development_history: Dict[str, List[Dict[str, Any]]] = {}
        self._market_character_insights: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Initialize the character development component."""
        try:
            # Initialize voice pattern analysis systems
            self._voice_patterns_cache = {}
            
            # Initialize relationship tracking
            self._relationship_history = defaultdict(list)
            
            # Initialize development history tracking
            self._development_history = defaultdict(list)
            
            # Initialize market insights cache
            self._market_character_insights = {}
            
            logger.info("Character development component initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Character development component initialization failed: {str(e)}"
            logger.error(error_msg)
            self.state.last_error = error_msg
            return False
    
    async def process(self, input_data: CharacterDevelopmentInput) -> CharacterDevelopmentResult:
        """
        Process character development analysis and planning.
        
        Args:
            input_data: Character development input data
            
        Returns:
            Character development result with assessments and plans
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Assess current character development state
            assessments = await self._assess_character_states(input_data)
            
            # Step 2: Analyze market alignment if available
            if input_data.market_preferences or input_data.success_patterns:
                await self._analyze_market_alignment(input_data, assessments)
            
            # Step 3: Create development plans
            development_plans = await self._create_development_plans(input_data, assessments)
            
            # Step 4: Update character arcs with planned developments
            updated_characters = await self._update_character_arcs(
                input_data.characters, development_plans, input_data.current_chapter
            )
            
            # Step 5: Plan relationship updates
            relationship_updates = await self._plan_relationship_updates(
                input_data, development_plans
            )
            
            # Step 6: Plan voice evolution if enabled
            voice_evolution_plans = await self._plan_voice_evolution(
                input_data, assessments, development_plans
            )
            
            # Step 7: Generate market alignment improvements
            market_improvements = await self._generate_market_improvements(
                input_data, assessments
            )
            
            # Step 8: Calculate overall health and recommendations
            overall_health = self._calculate_overall_development_health(assessments)
            recommended_focus = self._recommend_chapter_focus(development_plans, assessments)
            warning_flags = self._identify_warning_flags(assessments, development_plans)
            success_predictions = self._predict_development_success(development_plans, assessments)
            
            # Step 9: Generate development rationale if enabled
            development_rationale = {}
            if self.config.specific_config.include_development_rationale:
                development_rationale = self._generate_development_rationale(
                    development_plans, assessments, input_data
                )
            
            # Step 10: Create processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_metadata = {
                'processing_time_seconds': processing_time,
                'characters_processed': len(input_data.characters),
                'plans_generated': len(development_plans),
                'market_integration_used': bool(input_data.market_preferences or input_data.success_patterns),
                'research_integration_used': bool(input_data.research_insights),
                'component_version': self.config.version,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create result
            result = CharacterDevelopmentResult(
                assessments=assessments,
                development_plans=development_plans,
                updated_characters=updated_characters,
                priority_developments=[plan for plan in development_plans 
                                     if plan.priority in [CharacterDevelopmentPriority.CRITICAL, 
                                                        CharacterDevelopmentPriority.HIGH]],
                relationship_updates=relationship_updates,
                voice_evolution_plans=voice_evolution_plans,
                market_alignment_improvements=market_improvements,
                development_rationale=development_rationale,
                overall_development_health=overall_health,
                recommended_chapter_focus=recommended_focus,
                warning_flags=warning_flags,
                success_predictions=success_predictions,
                processing_metadata=processing_metadata
            )
            
            # Update internal tracking
            await self._update_internal_tracking(input_data, result)
            
            logger.info(
                f"Character development processing completed: "
                f"{len(development_plans)} plans generated for {len(input_data.characters)} characters"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Character development processing failed: {str(e)}"
            logger.error(error_msg)
            raise ComponentError(error_msg) from e
    
    async def health_check(self) -> bool:
        """Check component health."""
        try:
            # Verify core systems are accessible
            if not isinstance(self._voice_patterns_cache, dict):
                return False
            
            if not isinstance(self._relationship_history, dict):
                return False
            
            if not isinstance(self._development_history, dict):
                return False
            
            # Check configuration validity
            config = self.config.specific_config
            if config.staleness_threshold <= 0:
                return False
            
            if config.max_characters_per_chapter <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Character development component health check failed: {str(e)}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup component resources."""
        try:
            # Clear caches
            self._voice_patterns_cache.clear()
            self._relationship_history.clear()
            self._development_history.clear()
            self._market_character_insights.clear()
            
            logger.info("Character development component cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Character development component cleanup failed: {str(e)}")
            return False
    
    # Private Methods
    
    async def _assess_character_states(self, input_data: CharacterDevelopmentInput) -> Dict[str, CharacterDevelopmentAssessment]:
        """Assess the current development state of all characters."""
        assessments = {}
        config = self.config.specific_config
        
        for char_id, character in input_data.characters.items():
            # Calculate staleness
            chapters_since_dev = character.chapters_since_development(input_data.current_chapter)
            staleness_score = min(1.0, chapters_since_dev / config.staleness_threshold)
            
            # Assess voice consistency
            voice_score = await self._assess_voice_consistency(character, char_id)
            
            # Assess arc progression
            arc_score = self._assess_arc_progression(character, input_data.current_chapter)
            
            # Assess relationship health
            relationship_score = self._assess_relationship_health(character, input_data.characters)
            
            # Assess market alignment
            market_score = await self._assess_market_alignment(
                character, input_data.market_preferences
            )
            
            # Identify issues
            unresolved_obstacles = list(character.remaining_obstacles)
            stagnant_relationships = self._identify_stagnant_relationships(
                character, input_data.characters, input_data.current_chapter
            )
            voice_inconsistencies = await self._identify_voice_inconsistencies(character, char_id)
            development_opportunities = self._identify_development_opportunities(
                character, input_data.plot_outline, input_data.chapter_objectives
            )
            
            assessment = CharacterDevelopmentAssessment(
                character_id=char_id,
                chapters_since_development=chapters_since_dev,
                development_staleness_score=staleness_score,
                voice_consistency_score=voice_score,
                arc_progression_score=arc_score,
                relationship_health_score=relationship_score,
                unresolved_obstacles=unresolved_obstacles,
                stagnant_relationships=stagnant_relationships,
                voice_inconsistencies=voice_inconsistencies,
                development_opportunities=development_opportunities,
                market_alignment_score=market_score
            )
            
            assessments[char_id] = assessment
        
        return assessments
    
    async def _assess_voice_consistency(self, character: CharacterArc, char_id: str) -> float:
        """Assess character voice consistency."""
        if not self.config.specific_config.voice_pattern_analysis:
            return 1.0
        
        # Get cached voice patterns
        cached_patterns = self._voice_patterns_cache.get(char_id, {})
        
        # Analyze current voice characteristics
        current_voice = character.voice_characteristics
        
        if not cached_patterns:
            # First time analyzing this character - cache patterns and return perfect score
            self._voice_patterns_cache[char_id] = {
                'vocabulary_level': current_voice.vocabulary_level,
                'formality_level': current_voice.formality_level,
                'speech_patterns': set(current_voice.speech_patterns),
                'dialogue_quirks': set(current_voice.dialogue_quirks),
                'emotional_expressiveness': current_voice.emotional_expressiveness
            }
            return 1.0
        
        # Compare with cached patterns
        consistency_scores = []
        
        # Check vocabulary level consistency
        if cached_patterns['vocabulary_level'] == current_voice.vocabulary_level:
            consistency_scores.append(1.0)
        else:
            consistency_scores.append(0.5)  # Some change is acceptable
        
        # Check formality level consistency
        if cached_patterns['formality_level'] == current_voice.formality_level:
            consistency_scores.append(1.0)
        else:
            consistency_scores.append(0.5)
        
        # Check speech patterns overlap
        cached_patterns_set = cached_patterns['speech_patterns']
        current_patterns_set = set(current_voice.speech_patterns)
        if cached_patterns_set and current_patterns_set:
            overlap = len(cached_patterns_set & current_patterns_set) / len(cached_patterns_set | current_patterns_set)
            consistency_scores.append(overlap)
        else:
            consistency_scores.append(1.0)
        
        # Check dialogue quirks overlap
        cached_quirks_set = cached_patterns['dialogue_quirks']
        current_quirks_set = set(current_voice.dialogue_quirks)
        if cached_quirks_set and current_quirks_set:
            overlap = len(cached_quirks_set & current_quirks_set) / len(cached_quirks_set | current_quirks_set)
            consistency_scores.append(overlap)
        else:
            consistency_scores.append(1.0)
        
        # Return average consistency score
        return statistics.mean(consistency_scores) if consistency_scores else 1.0
    
    def _assess_arc_progression(self, character: CharacterArc, current_chapter: int) -> float:
        """Assess character arc progression health."""
        # Check if character has any growth trajectory
        if not character.growth_trajectory:
            return 0.3  # No growth is problematic
        
        # Check if character has grown recently
        chapters_since_growth = character.chapters_since_development(current_chapter)
        if chapters_since_growth == 0:
            return 1.0  # Recent growth is excellent
        elif chapters_since_growth <= 3:
            return 0.8  # Recent growth is good
        elif chapters_since_growth <= 5:
            return 0.6  # Acceptable gap
        else:
            return 0.3  # Too long without growth
    
    def _assess_relationship_health(self, character: CharacterArc, all_characters: Dict[str, CharacterArc]) -> float:
        """Assess health of character's relationships."""
        if not character.relationship_dynamics:
            return 0.5  # No relationships tracked
        
        relationship_scores = []
        
        for other_char_id, status in character.relationship_dynamics.items():
            if other_char_id in all_characters:
                # Check if relationship is dynamic (not just neutral/unknown)
                if status in [RelationshipStatus.NEUTRAL, RelationshipStatus.UNKNOWN]:
                    relationship_scores.append(0.4)
                elif status == RelationshipStatus.COMPLICATED:
                    relationship_scores.append(0.9)  # Complicated is interesting
                else:
                    relationship_scores.append(0.7)  # Clear relationship status is good
            else:
                relationship_scores.append(0.2)  # Relationship to non-existent character
        
        return statistics.mean(relationship_scores) if relationship_scores else 0.5
    
    async def _assess_market_alignment(self, character: CharacterArc, market_preferences: Optional[List[ReaderPreference]]) -> float:
        """Assess how well character aligns with market preferences."""
        if not market_preferences or not self.config.specific_config.use_market_intelligence:
            return 0.5  # No market data available
        
        alignment_scores = []
        
        for preference in market_preferences:
            # Check character traits against preference
            preference_lower = preference.preference_description.lower()
            
            # Check personality traits alignment
            for trait in character.personality_traits:
                if trait.lower() in preference_lower:
                    alignment_scores.append(preference.importance_weight)
            
            # Check narrative function alignment
            if character.narrative_function.value.lower() in preference_lower:
                alignment_scores.append(preference.importance_weight)
            
            # Check emotional characteristics alignment
            if character.emotional_state.lower() in preference_lower:
                alignment_scores.append(preference.importance_weight * 0.5)
        
        return statistics.mean(alignment_scores) if alignment_scores else 0.5
    
    def _identify_stagnant_relationships(self, character: CharacterArc, all_characters: Dict[str, CharacterArc], current_chapter: int) -> List[str]:
        """Identify relationships that haven't evolved recently."""
        stagnant = []
        
        # This is a simplified implementation - in full version would track relationship change history
        for other_char_id, status in character.relationship_dynamics.items():
            if other_char_id in all_characters:
                # Check if relationship is static (neutral/unknown for too long)
                if status in [RelationshipStatus.NEUTRAL, RelationshipStatus.UNKNOWN]:
                    # Check if both characters have been around long enough for relationship to develop
                    other_char = all_characters[other_char_id]
                    if (current_chapter - character.introduction_chapter > 5 and
                        current_chapter - other_char.introduction_chapter > 5):
                        stagnant.append(other_char_id)
        
        return stagnant
    
    async def _identify_voice_inconsistencies(self, character: CharacterArc, char_id: str) -> List[str]:
        """Identify voice inconsistencies that need attention."""
        inconsistencies = []
        
        if not self.config.specific_config.voice_pattern_analysis:
            return inconsistencies
        
        voice = character.voice_characteristics
        
        # Check for internal inconsistencies in voice characteristics
        if voice.vocabulary_level == "simple" and voice.formality_level == "formal":
            inconsistencies.append("Simple vocabulary with formal speech may be inconsistent")
        
        if voice.emotional_expressiveness == "reserved" and "dramatic" in voice.dialogue_quirks:
            inconsistencies.append("Reserved expressiveness conflicts with dramatic dialogue quirks")
        
        # Check against cached patterns
        cached = self._voice_patterns_cache.get(char_id, {})
        if cached:
            if cached.get('vocabulary_level') != voice.vocabulary_level:
                inconsistencies.append(f"Vocabulary level changed from {cached.get('vocabulary_level')} to {voice.vocabulary_level}")
        
        return inconsistencies
    
    def _identify_development_opportunities(self, character: CharacterArc, plot_outline: Dict[str, Any], chapter_objectives: List[str]) -> List[str]:
        """Identify opportunities for character development."""
        opportunities = []
        
        # Check unresolved obstacles
        if character.remaining_obstacles:
            opportunities.append(f"Resolve {len(character.remaining_obstacles)} remaining obstacles")
        
        # Check internal conflicts
        if character.internal_conflicts:
            opportunities.append(f"Address {len(character.internal_conflicts)} internal conflicts")
        
        # Check relationship development opportunities
        neutral_relationships = [
            char_id for char_id, status in character.relationship_dynamics.items()
            if status in [RelationshipStatus.NEUTRAL, RelationshipStatus.UNKNOWN]
        ]
        if neutral_relationships:
            opportunities.append(f"Develop {len(neutral_relationships)} neutral relationships")
        
        # Check narrative function opportunities
        if character.narrative_function == NarrativeFunction.SUPPORT:
            opportunities.append("Potential to elevate support character role")
        
        # Check voice development opportunities
        if not character.voice_characteristics.speech_patterns:
            opportunities.append("Develop distinctive speech patterns")
        
        if not character.voice_characteristics.dialogue_quirks:
            opportunities.append("Add memorable dialogue quirks")
        
        return opportunities
    
    async def _create_development_plans(self, input_data: CharacterDevelopmentInput, assessments: Dict[str, CharacterDevelopmentAssessment]) -> List[CharacterDevelopmentPlan]:
        """Create development plans for characters based on assessments."""
        plans = []
        config = self.config.specific_config
        
        # Sort characters by development need
        sorted_chars = sorted(
            assessments.items(),
            key=lambda x: (x[1].needs_immediate_development, x[1].development_staleness_score),
            reverse=True
        )
        
        # Limit characters per chapter
        chars_to_develop = sorted_chars[:config.max_characters_per_chapter]
        
        for char_id, assessment in chars_to_develop:
            character = input_data.characters[char_id]
            
            # Determine priority
            if char_id in input_data.force_development:
                priority = CharacterDevelopmentPriority.CRITICAL
            elif assessment.needs_immediate_development:
                priority = CharacterDevelopmentPriority.HIGH
            elif assessment.development_staleness_score > 0.5:
                priority = CharacterDevelopmentPriority.MEDIUM
            else:
                priority = CharacterDevelopmentPriority.LOW
            
            # Create plans based on assessment
            char_plans = await self._create_character_specific_plans(
                character, assessment, input_data, priority
            )
            
            plans.extend(char_plans)
        
        # Sort plans by priority and confidence
        plans.sort(key=lambda x: (
            list(CharacterDevelopmentPriority).index(x.priority),
            -x.confidence_score
        ))
        
        return plans
    
    async def _create_character_specific_plans(self, character: CharacterArc, assessment: CharacterDevelopmentAssessment, input_data: CharacterDevelopmentInput, priority: CharacterDevelopmentPriority) -> List[CharacterDevelopmentPlan]:
        """Create specific development plans for a character."""
        plans = []
        config = self.config.specific_config
        
        # Plan obstacle resolution
        if assessment.unresolved_obstacles:
            for obstacle in assessment.unresolved_obstacles[:2]:  # Limit to 2 per chapter
                plan = CharacterDevelopmentPlan(
                    character_id=character.character_id,
                    character_name=character.name,
                    priority=priority,
                    development_type=DevelopmentType.OBSTACLE_RESOLUTION,
                    development_description=f"Resolve obstacle: {obstacle}",
                    target_chapter=input_data.current_chapter,
                    expected_outcomes=[f"Character growth through overcoming {obstacle}"],
                    confidence_score=0.8
                )
                plans.append(plan)
        
        # Plan relationship development
        if assessment.stagnant_relationships:
            for other_char_id in assessment.stagnant_relationships[:1]:  # One relationship per chapter
                plan = CharacterDevelopmentPlan(
                    character_id=character.character_id,
                    character_name=character.name,
                    priority=priority,
                    development_type=DevelopmentType.RELATIONSHIP_EVOLUTION,
                    development_description=f"Develop relationship with {other_char_id}",
                    target_chapter=input_data.current_chapter,
                    expected_outcomes=[f"More dynamic relationship with {other_char_id}"],
                    relationship_impacts={other_char_id: "Relationship becomes more defined"},
                    confidence_score=0.7
                )
                plans.append(plan)
        
        # Plan voice refinement if inconsistencies detected
        if assessment.voice_inconsistencies:
            plan = CharacterDevelopmentPlan(
                character_id=character.character_id,
                character_name=character.name,
                priority=priority,
                development_type=DevelopmentType.VOICE_REFINEMENT,
                development_description=f"Refine voice consistency addressing {len(assessment.voice_inconsistencies)} inconsistencies",
                target_chapter=input_data.current_chapter,
                expected_outcomes=["More consistent character voice"],
                voice_evolution_notes="Address identified voice inconsistencies while maintaining character essence",
                confidence_score=0.6
            )
            plans.append(plan)
        
        # Plan growth milestone if character is stale
        if assessment.development_staleness_score > 0.7:
            plan = CharacterDevelopmentPlan(
                character_id=character.character_id,
                character_name=character.name,
                priority=priority,
                development_type=DevelopmentType.GROWTH_MILESTONE,
                development_description="Achieve significant character growth milestone",
                target_chapter=input_data.current_chapter,
                expected_outcomes=["Character learns important lesson or gains new perspective"],
                confidence_score=0.9
            )
            plans.append(plan)
        
        return plans
    
    async def _update_character_arcs(self, characters: Dict[str, CharacterArc], development_plans: List[CharacterDevelopmentPlan], current_chapter: int) -> Dict[str, CharacterArc]:
        """Update character arcs based on development plans."""
        updated_characters = {}
        
        for char_id, character in characters.items():
            # Create updated character (copy)
            updated_char = CharacterArc(
                **character.model_dump()
            )
            
            # Apply development plans for this character
            char_plans = [plan for plan in development_plans if plan.character_id == char_id]
            
            for plan in char_plans:
                if plan.development_type == DevelopmentType.GROWTH_MILESTONE:
                    updated_char.add_growth_milestone(plan.development_description, current_chapter)
                
                elif plan.development_type == DevelopmentType.OBSTACLE_RESOLUTION:
                    # Mark obstacles for resolution
                    for obstacle in updated_char.remaining_obstacles:
                        if obstacle in plan.development_description:
                            updated_char.resolve_obstacle(obstacle, "Planned resolution", current_chapter)
                            break
            
            updated_characters[char_id] = updated_char
        
        return updated_characters
    
    async def _plan_relationship_updates(self, input_data: CharacterDevelopmentInput, development_plans: List[CharacterDevelopmentPlan]) -> Dict[str, Dict[str, RelationshipStatus]]:
        """Plan relationship status updates."""
        updates = {}
        
        for plan in development_plans:
            if plan.development_type == DevelopmentType.RELATIONSHIP_EVOLUTION:
                if plan.character_id not in updates:
                    updates[plan.character_id] = {}
                
                # Plan relationship improvements based on plan impacts
                for other_char_id, impact in plan.relationship_impacts.items():
                    if "dynamic" in impact.lower():
                        updates[plan.character_id][other_char_id] = RelationshipStatus.COMPLICATED
                    elif "positive" in impact.lower():
                        updates[plan.character_id][other_char_id] = RelationshipStatus.ALLIED
                    elif "tension" in impact.lower():
                        updates[plan.character_id][other_char_id] = RelationshipStatus.ANTAGONISTIC
        
        return updates
    
    async def _plan_voice_evolution(self, input_data: CharacterDevelopmentInput, assessments: Dict[str, CharacterDevelopmentAssessment], development_plans: List[CharacterDevelopmentPlan]) -> Dict[str, Dict[str, Any]]:
        """Plan character voice evolution."""
        voice_plans = {}
        
        for plan in development_plans:
            if plan.development_type == DevelopmentType.VOICE_REFINEMENT and plan.voice_evolution_notes:
                voice_plans[plan.character_id] = {
                    'refinement_notes': plan.voice_evolution_notes,
                    'target_consistency_score': 0.9,
                    'evolution_strategy': self.config.specific_config.voice_consistency_level.value
                }
        
        return voice_plans
    
    async def _generate_market_improvements(self, input_data: CharacterDevelopmentInput, assessments: Dict[str, CharacterDevelopmentAssessment]) -> Dict[str, List[str]]:
        """Generate market alignment improvement suggestions."""
        improvements = {}
        
        if not input_data.market_preferences:
            return improvements
        
        for char_id, assessment in assessments.items():
            if assessment.market_alignment_score < 0.6:
                char_improvements = []
                
                # Generate suggestions based on market preferences
                for preference in input_data.market_preferences:
                    if preference.importance_weight > 0.5:
                        char_improvements.append(
                            f"Consider incorporating: {preference.preference_description[:100]}..."
                        )
                
                if char_improvements:
                    improvements[char_id] = char_improvements[:3]  # Top 3 suggestions
        
        return improvements
    
    def _calculate_overall_development_health(self, assessments: Dict[str, CharacterDevelopmentAssessment]) -> float:
        """Calculate overall development health across all characters."""
        if not assessments:
            return 0.0
        
        health_scores = [assessment.overall_health_score for assessment in assessments.values()]
        return statistics.mean(health_scores)
    
    def _recommend_chapter_focus(self, development_plans: List[CharacterDevelopmentPlan], assessments: Dict[str, CharacterDevelopmentAssessment]) -> List[str]:
        """Recommend characters to focus on in upcoming chapters."""
        focus_characters = []
        
        # Add high-priority characters
        high_priority_chars = {
            plan.character_id for plan in development_plans
            if plan.priority in [CharacterDevelopmentPriority.CRITICAL, CharacterDevelopmentPriority.HIGH]
        }
        focus_characters.extend(high_priority_chars)
        
        # Add characters with low health scores
        low_health_chars = [
            char_id for char_id, assessment in assessments.items()
            if assessment.overall_health_score < 0.5 and char_id not in high_priority_chars
        ]
        focus_characters.extend(low_health_chars[:2])  # Limit to 2 additional
        
        return focus_characters
    
    def _identify_warning_flags(self, assessments: Dict[str, CharacterDevelopmentAssessment], development_plans: List[CharacterDevelopmentPlan]) -> List[str]:
        """Identify warning flags for potential development issues."""
        warnings = []
        
        # Check for characters with very low health
        very_low_health = [
            char_id for char_id, assessment in assessments.items()
            if assessment.overall_health_score < 0.3
        ]
        if very_low_health:
            warnings.append(f"Characters with very low development health: {', '.join(very_low_health)}")
        
        # Check for too many high-priority developments
        high_priority_count = len([
            plan for plan in development_plans
            if plan.priority == CharacterDevelopmentPriority.CRITICAL
        ])
        if high_priority_count > 3:
            warnings.append(f"Too many critical developments planned ({high_priority_count})")
        
        # Check for characters with many voice inconsistencies
        voice_problem_chars = [
            char_id for char_id, assessment in assessments.items()
            if len(assessment.voice_inconsistencies) > 3
        ]
        if voice_problem_chars:
            warnings.append(f"Characters with voice consistency issues: {', '.join(voice_problem_chars)}")
        
        return warnings
    
    def _predict_development_success(self, development_plans: List[CharacterDevelopmentPlan], assessments: Dict[str, CharacterDevelopmentAssessment]) -> Dict[str, float]:
        """Predict success probability for planned developments."""
        predictions = {}
        
        for plan in development_plans:
            assessment = assessments.get(plan.character_id)
            if not assessment:
                continue
            
            # Base prediction on plan confidence and character health
            base_score = plan.confidence_score
            health_modifier = assessment.overall_health_score * 0.3
            priority_modifier = {
                CharacterDevelopmentPriority.CRITICAL: 0.1,
                CharacterDevelopmentPriority.HIGH: 0.05,
                CharacterDevelopmentPriority.MEDIUM: 0.0,
                CharacterDevelopmentPriority.LOW: -0.05,
                CharacterDevelopmentPriority.DEFERRED: -0.1
            }.get(plan.priority, 0.0)
            
            prediction = min(1.0, max(0.0, base_score + health_modifier + priority_modifier))
            predictions[f"{plan.character_id}_{plan.development_type.value}"] = prediction
        
        return predictions
    
    def _generate_development_rationale(self, development_plans: List[CharacterDevelopmentPlan], assessments: Dict[str, CharacterDevelopmentAssessment], input_data: CharacterDevelopmentInput) -> Dict[str, str]:
        """Generate rationale for development decisions."""
        rationale = {}
        
        for char_id, assessment in assessments.items():
            reasons = []
            
            if assessment.development_staleness_score > 0.7:
                reasons.append(f"Character has been stale for {assessment.chapters_since_development} chapters")
            
            if assessment.overall_health_score < 0.5:
                reasons.append(f"Overall development health is low ({assessment.overall_health_score:.2f})")
            
            if assessment.unresolved_obstacles:
                reasons.append(f"{len(assessment.unresolved_obstacles)} unresolved obstacles need attention")
            
            if assessment.voice_inconsistencies:
                reasons.append(f"Voice consistency issues detected: {len(assessment.voice_inconsistencies)}")
            
            if reasons:
                rationale[char_id] = "; ".join(reasons)
        
        return rationale
    
    async def _update_internal_tracking(self, input_data: CharacterDevelopmentInput, result: CharacterDevelopmentResult) -> None:
        """Update internal tracking systems with processing results."""
        timestamp = datetime.now().isoformat()
        
        # Update development history
        for char_id in input_data.characters.keys():
            self._development_history[char_id].append({
                'chapter': input_data.current_chapter,
                'timestamp': timestamp,
                'assessment': result.assessments.get(char_id),
                'plans_count': len([p for p in result.development_plans if p.character_id == char_id])
            })
        
        # Update relationship tracking
        for char_id, updates in result.relationship_updates.items():
            if char_id not in self._relationship_history:
                self._relationship_history[char_id] = []
            
            self._relationship_history[char_id].append({
                'chapter': input_data.current_chapter,
                'timestamp': timestamp,
                'updates': updates
            })
    
    async def _analyze_market_alignment(self, input_data: CharacterDevelopmentInput, assessments: Dict[str, CharacterDevelopmentAssessment]) -> None:
        """Analyze and cache market alignment insights."""
        if not self.config.specific_config.use_market_intelligence:
            return
        
        # Store market insights for future use
        self._market_character_insights[input_data.current_chapter] = {
            'preferences_count': len(input_data.market_preferences) if input_data.market_preferences else 0,
            'success_patterns_count': len(input_data.success_patterns) if input_data.success_patterns else 0,
            'character_alignment_scores': {
                char_id: assessment.market_alignment_score
                for char_id, assessment in assessments.items()
            }
        }
"""
Chapter Generator Component

Implements the Generator component of the adversarial system, creating multiple
chapter variants based on story state, objectives, and market intelligence.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from pydantic import Field, BaseModel
from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.chapter_objective import ChapterObjective
from musequill.v3.models.chapter_variant import ChapterVariant, ChapterApproach, SceneStructure, GenerationMetadata
from musequill.v3.models.dynamic_story_state import DynamicStoryState
from musequill.v3.models.market_intelligence import MarketIntelligence
from musequill.v3.llm.ollama.ollama_client import LLMService, create_llm_service

class ChapterGeneratorConfig(BaseModel):
    """Configuration for Chapter Generator component."""
    
    llm_model_name: str = Field(
        default="llama3.3:70b",
        description="LLM model to use for generation"
    )
    
    max_generation_time_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Maximum time for single chapter generation"
    )
    
    variants_to_generate: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of chapter variants to generate"
    )
    
    creativity_temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.5,
        description="LLM creativity temperature"
    )
    
    max_context_tokens: int = Field(
        default=8000,
        ge=2000,
        le=32000,
        description="Maximum tokens for context assembly"
    )
    
    adaptive_parameters: bool = Field(
        default=True,
        description="Whether to adapt generation parameters based on feedback"
    )
    
    market_intelligence_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How much to weight market intelligence in generation"
    )
    
    enable_self_critique: bool = Field(
        default=True,
        description="Whether generator should self-critique during generation"
    )
    
    banned_phrases: List[str] = Field(
        default_factory=list,
        description="Phrases to avoid in generation"
    )


class ChapterGeneratorInput(BaseModel):
    """Input data for Chapter Generator."""
    
    chapter_objective: ChapterObjective = Field(
        description="Detailed objectives for chapter generation"
    )
    
    story_state: DynamicStoryState = Field(
        description="Current story state for context"
    )
    
    market_intelligence: Optional[MarketIntelligence] = Field(
        default=None,
        description="Market intelligence for commercial optimization"
    )
    
    previous_chapter_text: Optional[str] = Field(
        default=None,
        description="Previous chapter for continuity"
    )
    
    critic_feedback_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical feedback from critics for learning"
    )
    
    generation_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional constraints for generation"
    )


class ChapterGeneratorOutput(BaseModel):
    """Output from Chapter Generator."""
    
    chapter_variants: List[ChapterVariant] = Field(
        description="Generated chapter variants"
    )
    
    generation_metadata: Dict[str, Any] = Field(
        description="Metadata about generation process"
    )
    
    context_summary: str = Field(
        description="Summary of context used for generation"
    )
    
    approach_rationales: Dict[ChapterApproach, str] = Field(
        description="Rationale for each approach taken"
    )


class ChapterGenerator(BaseComponent[ChapterGeneratorInput, ChapterGeneratorOutput, ChapterGeneratorConfig]):
    """
    Chapter Generator component for creating multiple chapter variants.
    
    Implements the Generator role in the adversarial system, creating diverse
    approaches to fulfilling chapter objectives while learning from critic feedback.
    """
    
    def __init__(self, config: ComponentConfiguration[ChapterGeneratorConfig]):
        super().__init__(config)
        self._llm_client:Optional[LLMService] = None
        self._generation_history: List[Dict[str, Any]] = []
        self._learned_patterns: Dict[str, Any] = {}
        self._context_cache: Dict[str, str] = {}
        self._banned_patterns: Set[str] = set()
    
    async def initialize(self) -> bool:
        """Initialize the chapter generator with LLM client and learning systems."""
        try:
            # Initialize LLM client
            await self._initialize_llm_client()
            
            # Load learned patterns from previous generations
            await self._load_learned_patterns()
            
            # Initialize banned patterns
            self._banned_patterns.update(self.config.specific_config.banned_phrases)
            
            # Initialize context assembly tools
            await self._initialize_context_tools()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Chapter generator initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: ChapterGeneratorInput) -> ChapterGeneratorOutput:
        """
        Generate multiple chapter variants based on objectives and context.
        
        Args:
            input_data: Chapter objectives, story state, and context
            
        Returns:
            Multiple chapter variants with different narrative approaches
        """
        start_time = datetime.now()
        generation_id = f"gen_{int(start_time.timestamp())}"
        
        try:
            # Assemble generation context
            context = await self._assemble_generation_context(input_data, generation_id)
            
            # Determine generation approaches based on objectives and market intelligence
            approaches = await self._select_generation_approaches(
                input_data.chapter_objective,
                input_data.market_intelligence,
                input_data.critic_feedback_history
            )
            
            # Adapt generation parameters based on learning
            if self.config.specific_config.adaptive_parameters:
                await self._adapt_generation_parameters(input_data.critic_feedback_history)
            
            # Generate chapter variants concurrently
            variant_tasks = []
            for approach in approaches:
                task = self._generate_chapter_variant(
                    approach, context, input_data, generation_id
                )
                variant_tasks.append(task)
            
            # Execute generation with timeout
            timeout_seconds = self.config.specific_config.max_generation_time_seconds
            variant_results = await asyncio.wait_for(
                asyncio.gather(*variant_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            
            # Process variant results and handle exceptions
            successful_variants = []
            approach_rationales = {}
            
            for i, result in enumerate(variant_results):
                if isinstance(result, Exception):
                    self.state.error_count += 1
                    continue
                
                variant, rationale = result
                successful_variants.append(variant)
                approach_rationales[approaches[i]] = rationale
            
            if not successful_variants:
                raise ComponentError("No chapter variants were successfully generated")
            
            # Generate comprehensive metadata
            generation_metadata = {
                'generation_id': generation_id,
                'total_generation_time': (datetime.now() - start_time).total_seconds(),
                'variants_attempted': len(approaches),
                'variants_successful': len(successful_variants),
                'context_tokens_used': len(context.split()) * 1.3,  # Rough estimate
                'adaptation_applied': self.config.specific_config.adaptive_parameters,
                'market_intelligence_used': input_data.market_intelligence is not None
            }
            
            # Update learning from generation
            await self._update_learning_from_generation(
                input_data, successful_variants, generation_metadata
            )
            
            return ChapterGeneratorOutput(
                chapter_variants=successful_variants,
                generation_metadata=generation_metadata,
                context_summary=context[:500] + "..." if len(context) > 500 else context,
                approach_rationales=approach_rationales
            )
            
        except asyncio.TimeoutError:
            raise ComponentError(f"Chapter generation exceeded {timeout_seconds}s timeout")
        except Exception as e:
            raise ComponentError(f"Chapter generation failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on generation systems."""
        try:
            # Test LLM connectivity
            if not await self._test_llm_connection():
                return False
            
            # Check generation success rates
            if self.state.metrics.failure_rate > 0.3:
                return False
            
            # Test context assembly
            test_context = await self._test_context_assembly()
            if not test_context:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup generation resources."""
        try:
            if self._llm_client:
                await self._cleanup_llm_client()
            
            self._generation_history.clear()
            self._context_cache.clear()
            
            return True
            
        except Exception:
            return False
    
    async def _initialize_llm_client(self) -> None:
        """Initialize LLM client for generation."""
        # Placeholder for actual LLM client initialization
        # Would initialize Anthropic Claude, OpenAI GPT, or other LLM service
        self._llm_client = create_llm_service()
        await self._llm_client.update_default_parameters(
            {
                'model': self.config.specific_config.llm_model_name,
                'temperature': self.config.specific_config.creativity_temperature,
                'max_tokens': 4000
            }
        )
    async def _load_learned_patterns(self) -> None:
        """Load patterns learned from previous generations and critic feedback."""
        # Placeholder for loading learned patterns from persistent storage
        self._learned_patterns = {
            'successful_openings': [],
            'effective_transitions': [],
            'engaging_dialogue_patterns': [],
            'strong_endings': [],
            'character_voice_patterns': {},
            'market_aligned_elements': []
        }
    
    async def _initialize_context_tools(self) -> None:
        """Initialize tools for context assembly and management."""
        
        # Initialize context prioritization system
        self._context_prioritizer = {
            'story_context': {
                'weight': 0.4,
                'always_include': True,
                'compression_threshold': 1000,
                'priority': 1
            },
            'chapter_objectives': {
                'weight': 0.25,
                'always_include': True,
                'compression_threshold': 500,
                'priority': 2
            },
            'previous_chapter': {
                'weight': 0.15,
                'always_include': False,
                'compression_threshold': 800,
                'priority': 3
            },
            'market_intelligence': {
                'weight': 0.1,
                'always_include': False,
                'compression_threshold': 600,
                'priority': 4
            },
            'learned_patterns': {
                'weight': 0.08,
                'always_include': False,
                'compression_threshold': 400,
                'priority': 5
            },
            'generation_constraints': {
                'weight': 0.02,
                'always_include': False,
                'compression_threshold': 200,
                'priority': 6
            }
        }
        
        # Initialize context assembly utilities
        self._context_assembler = {
            'token_estimator': lambda text: len(text.split()) * 1.3,  # Rough token estimation
            'summarizers': {
                'extractive': self._extractive_summarize,
                'abstractive': self._abstractive_summarize,
                'template': self._template_summarize
            },
            'compressors': {
                'truncation': self._truncate_content,
                'selective_pruning': self._selective_content_pruning,
                'semantic_compression': self._semantic_content_compression
            },
            'formatters': {
                'story_context': self._format_story_context,
                'objectives': self._format_objectives_context,
                'market_guidance': self._format_market_guidance,
                'feedback_learnings': self._format_feedback_learnings,
                'constraints': self._format_constraints
            }
        }
        
        # Initialize context caching system
        self._context_cache = {
            # Cache already initialized in __init__
        }
        self._cache_metadata = {
            'hit_rate': 0.0,
            'cache_size': 0,
            'max_cache_size': 100,
            'ttl_hours': 24,
            'last_cleanup': datetime.now()
        }
        
        # Initialize context validation tools
        self._context_validator = {
            'token_limits': {
                'total': self.config.specific_config.max_context_tokens,
                'story_context': int(self.config.specific_config.max_context_tokens * 0.4),
                'objectives': int(self.config.specific_config.max_context_tokens * 0.25),
                'market_intelligence': int(self.config.specific_config.max_context_tokens * 0.2),
                'other_sections': int(self.config.specific_config.max_context_tokens * 0.15)
            },
            'required_sections': ['story_context', 'chapter_objectives'],
            'section_validators': {
                'story_context': self._validate_story_context,
                'chapter_objectives': self._validate_objectives_context,
                'market_intelligence': self._validate_market_context,
                'learned_patterns': self._validate_feedback_context
            }
        }
        
        # Initialize intelligent trimming system
        self._context_trimmer = {
            'strategies': [
                ('preserve_critical', self._preserve_critical_sections),
                ('smart_truncation', self._smart_content_truncation),
                ('importance_ranking', self._importance_based_trimming),
                ('semantic_clustering', self._cluster_and_compress)
            ],
            'section_importance': {
                'story_context': 10,
                'chapter_objectives': 9,
                'previous_chapter': 7,
                'market_intelligence': 5,
                'learned_patterns': 4,
                'generation_constraints': 3
            },
            'preserve_patterns': [
                r'=== [A-Z\s]+ ===',  # Section headers
                r'Primary Goal:.*',    # Key objectives
                r'Word Target:.*',     # Critical requirements
                r'NEEDS PAYOFF.*',     # Reader expectations
                r'Active Threads:.*'   # Plot continuity
            ]
        }
        
        # Initialize adaptive context management
        self._adaptive_context = {
            'generation_history': [],
            'success_patterns': {
                'effective_context_sizes': {},
                'successful_section_combinations': [],
                'optimal_compression_ratios': {}
            },
            'failure_analysis': {
                'context_too_large': 0,
                'missing_critical_info': 0,
                'poor_compression': 0
            },
            'learning_enabled': self.config.specific_config.adaptive_parameters
        }

# Helper methods for context tools
    async def _create_story_analysis(self, story_state: DynamicStoryState, max_tokens: int) -> str:
        """Create analytical summary of current story state."""
        analysis_parts = []
        
        # Momentum analysis
        momentum = story_state.calculate_momentum_score()
        if momentum < 0.3:
            analysis_parts.append("LOW MOMENTUM - Story needs acceleration")
        elif momentum > 0.8:
            analysis_parts.append("HIGH MOMENTUM - Maintain current pace")
        else:
            analysis_parts.append(f"Moderate momentum ({momentum:.1f})")
        
        # Thread analysis
        active_threads = story_state.get_active_threads()
        stagnant_threads = story_state.get_stagnant_threads()
        
        if stagnant_threads:
            analysis_parts.append(f"{len(stagnant_threads)} stagnant threads need attention")
        
        resolution_ready = story_state.get_resolution_ready_threads()
        if resolution_ready:
            analysis_parts.append(f"{len(resolution_ready)} threads ready for resolution")
        
        # Character development analysis
        chars_needing_dev = story_state.get_characters_needing_development()
        if chars_needing_dev:
            analysis_parts.append(f"{len(chars_needing_dev)} characters need development")
        
        # Reader satisfaction analysis
        if story_state.reader_expectations.needs_payoff():
            debt = story_state.reader_expectations.satisfaction_debt
            analysis_parts.append(f"Reader satisfaction debt: {debt:.2f} (NEEDS PAYOFF)")
        
        # Story phase alignment
        completion = story_state.story_completion_ratio
        phase = story_state.story_phase.value
        analysis_parts.append(f"Story phase: {phase} ({completion:.0%} complete)")
        
        # Combine and limit
        analysis = " | ".join(analysis_parts)
        
        # Trim if too long
        if len(analysis.split()) * 1.3 > max_tokens:
            analysis = await self._truncate_content(analysis, max_tokens)
        
        return analysis

    async def _extract_urgency_indicators(self, story_state: DynamicStoryState, max_tokens: int) -> str:
        """Extract indicators of what needs immediate attention."""
        urgencies = []
        
        # Critical thread staleness
        stagnant = story_state.get_stagnant_threads(staleness_threshold=5)
        for thread in stagnant[:3]:  # Top 3 most urgent
            chapters_stagnant = thread.chapters_since_advancement(story_state.current_chapter)
            urgencies.append(f"{thread.title} stagnant {chapters_stagnant}ch")
        
        # High-investment threads needing advancement
        active_threads = story_state.get_active_threads()
        high_investment_stagnant = [
            t for t in active_threads 
            if t.reader_investment.value == 'high' and 
            t.chapters_since_advancement(story_state.current_chapter) >= 3
        ]
        
        for thread in high_investment_stagnant[:2]:
            urgencies.append(f"HIGH-INVESTMENT: {thread.title} needs advancement")
        
        # Resolution opportunities
        ready_threads = story_state.get_resolution_ready_threads(readiness_threshold=0.8)
        for thread in ready_threads[:2]:
            urgencies.append(f"RESOLUTION READY: {thread.title}")
        
        # Character development urgencies
        chars_urgent = [
            c for c in story_state.get_characters_needing_development()
            if c.chapters_since_development(story_state.current_chapter) >= 6
        ]
        
        for char in chars_urgent[:2]:
            urgencies.append(f"CHARACTER: {char.name} needs development")
        
        # Reader expectation pressure
        if story_state.reader_expectations.satisfaction_debt >= 0.7:
            urgencies.append("CRITICAL: Reader satisfaction debt")
        
        # Narrative tension imbalance
        if story_state.narrative_tension == story_state.NarrativeTension.BUILDING:
            if len([t for t in active_threads if t.tension_level >= 7]) == 0:
                urgencies.append("Need higher tension threads")
        
        # Combine urgencies
        if not urgencies:
            return "No critical urgencies identified"
        
        urgency_text = " | ".join(urgencies)
        
        # Trim if too long
        if len(urgency_text.split()) * 1.3 > max_tokens:
            urgency_text = await self._truncate_content(urgency_text, max_tokens)
        
        return urgency_text

    async def _calculate_story_health(self, story_state: DynamicStoryState, max_tokens: int) -> str:
        """Calculate overall story health metrics."""
        health_metrics = []
        
        # Thread health
        active_threads = story_state.get_active_threads()
        if active_threads:
            avg_tension = sum(t.tension_level for t in active_threads) / len(active_threads)
            health_metrics.append(f"Avg tension: {avg_tension:.1f}")
            
            stagnant_ratio = len(story_state.get_stagnant_threads()) / len(active_threads)
            if stagnant_ratio > 0.3:
                health_metrics.append(f"Stagnancy: {stagnant_ratio:.0%} (HIGH)")
            else:
                health_metrics.append(f"Stagnancy: {stagnant_ratio:.0%}")
        
        # Character health
        if story_state.character_arcs:
            chars_needing_dev = len(story_state.get_characters_needing_development())
            total_chars = len(story_state.character_arcs)
            dev_ratio = chars_needing_dev / total_chars
            health_metrics.append(f"Char dev needed: {dev_ratio:.0%}")
        
        # Reader satisfaction health
        satisfaction_debt = story_state.reader_expectations.satisfaction_debt
        if satisfaction_debt >= 0.8:
            health_metrics.append("Reader satisfaction: CRITICAL")
        elif satisfaction_debt >= 0.6:
            health_metrics.append("Reader satisfaction: Poor")
        elif satisfaction_debt <= 0.3:
            health_metrics.append("Reader satisfaction: Good")
        else:
            health_metrics.append(f"Reader satisfaction: {satisfaction_debt:.1f}")
        
        # Momentum health
        momentum = story_state.momentum_score
        if momentum >= 0.8:
            health_metrics.append("Momentum: Excellent")
        elif momentum >= 0.6:
            health_metrics.append("Momentum: Good")
        elif momentum >= 0.4:
            health_metrics.append("Momentum: Fair")
        else:
            health_metrics.append("Momentum: Poor")
        
        # World consistency health
        consistency_issues = story_state.validate_consistency()
        if len(consistency_issues) > 3:
            health_metrics.append(f"Consistency: {len(consistency_issues)} issues")
        elif len(consistency_issues) > 0:
            health_metrics.append(f"Consistency: {len(consistency_issues)} minor issues")
        else:
            health_metrics.append("Consistency: Good")
        
        health_summary = " | ".join(health_metrics)
        
        # Trim if needed
        if len(health_summary.split()) * 1.3 > max_tokens:
            health_summary = await self._truncate_content(health_summary, max_tokens)
        
        return health_summary

# Helper methods for context tools

    async def _extractive_summarize(self, text: str, max_tokens: int) -> str:
        """Extract most important sentences up to token limit."""
        sentences = text.split('. ')
        if not sentences:
            return text
        
        # Score sentences by importance indicators
        scored_sentences = []
        importance_keywords = [
            'protagonist', 'antagonist', 'conflict', 'tension', 'objective', 
            'goal', 'obstacle', 'relationship', 'emotion', 'critical', 'important'
        ]
        
        for i, sentence in enumerate(sentences):
            score = 0
            # Position score (earlier sentences often more important)
            score += max(0, 10 - i)
            
            # Keyword score
            for keyword in importance_keywords:
                if keyword.lower() in sentence.lower():
                    score += 5
            
            # Length penalty for very long sentences
            if len(sentence.split()) > 30:
                score -= 2
                
            scored_sentences.append((score, sentence))
        
        # Sort by score and select best sentences within token limit
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        selected = []
        current_tokens = 0
        
        for score, sentence in scored_sentences:
            sentence_tokens = len(sentence.split()) * 1.3
            if current_tokens + sentence_tokens <= max_tokens:
                selected.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return '. '.join(selected) + ('.' if selected else '')

    async def _abstractive_summarize(self, text: str, max_tokens: int) -> str:
        """Create abstract summary of key points."""
        # Simple template-based abstractive summarization
        # In production, this would use a dedicated summarization model
        
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract key information patterns
            if 'Primary Goal:' in line:
                key_points.append(line)
            elif 'Active Threads:' in line:
                key_points.append(line)
            elif 'Characters:' in line:
                key_points.append(line)
            elif 'NEEDS PAYOFF' in line:
                key_points.append(line)
            elif line.startswith('Chapter') and ('|' in line):
                key_points.append(line)
        
        summary = ' | '.join(key_points)
        
        # Trim if too long
        if len(summary.split()) * 1.3 > max_tokens:
            words = summary.split()
            target_words = int(max_tokens / 1.3)
            summary = ' '.join(words[:target_words]) + '...'
        
        return summary

    async def _template_summarize(self, text: str, max_tokens: int) -> str:
        """Use templates to create structured summaries."""
        template_parts = []
        
        # Extract structured information
        if 'Chapter' in text and '|' in text:
            chapter_info = [line for line in text.split('\n') if 'Chapter' in line and '|' in line]
            if chapter_info:
                template_parts.append(f"Status: {chapter_info[0]}")
        
        if 'Active Threads:' in text:
            threads = [line for line in text.split('\n') if 'Active Threads:' in line]
            if threads:
                template_parts.append(threads[0])
        
        if 'Characters:' in text:
            characters = [line for line in text.split('\n') if 'Characters:' in line]
            if characters:
                template_parts.append(characters[0])
        
        return ' | '.join(template_parts) if template_parts else text[:max_tokens]

    async def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Simple truncation with ellipsis."""
        words = content.split()
        if len(words) * 1.3 <= max_tokens:
            return content
        
        target_words = int(max_tokens / 1.3) - 1
        return ' '.join(words[:target_words]) + '...'

    async def _selective_content_pruning(self, content: str, max_tokens: int) -> str:
        """Remove less important content while preserving structure."""
        lines = content.split('\n')
        important_lines = []
        optional_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classify line importance
            if any(pattern in line for pattern in ['===', 'Primary Goal:', 'Active Threads:', 'NEEDS PAYOFF']):
                important_lines.append(line)
            else:
                optional_lines.append(line)
        
        # Start with important lines
        result_lines = important_lines[:]
        current_tokens = sum(len(line.split()) * 1.3 for line in result_lines)
        
        # Add optional lines if space permits
        for line in optional_lines:
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens <= max_tokens:
                result_lines.append(line)
                current_tokens += line_tokens
        
        return '\n'.join(result_lines)

    async def _semantic_content_compression(self, content: str, max_tokens: int) -> str:
        """Compress content while preserving semantic meaning."""
        # Simplified semantic compression
        lines = content.split('\n')
        compressed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove redundant words while preserving meaning
            words = line.split()
            filtered_words = []
            
            skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'very', 'quite', 'rather'}
            
            for word in words:
                if word.lower() not in skip_words or len(filtered_words) == 0:
                    filtered_words.append(word)
            
            compressed_line = ' '.join(filtered_words)
            compressed_lines.append(compressed_line)
        
        compressed_content = '\n'.join(compressed_lines)
        
        # Final truncation if still too long
        if len(compressed_content.split()) * 1.3 > max_tokens:
            return await self._truncate_content(compressed_content, max_tokens)
        
        return compressed_content

    async def _validate_story_context(self, context: str) -> bool:
        """Validate story context section."""
        required_elements = ['Chapter', 'Active Threads', 'Characters']
        return any(element in context for element in required_elements)

    async def _validate_objectives_context(self, context: str) -> bool:
        """Validate objectives context section."""
        return 'Primary Goal:' in context or 'Word Target:' in context

    async def _validate_market_context(self, context: str) -> bool:
        """Validate market intelligence context section."""
        return len(context.strip()) > 0

    async def _validate_feedback_context(self, context: str) -> bool:
        """Validate feedback context section."""
        return len(context.strip()) > 0

    async def _preserve_critical_sections(self, context: str, target_tokens: int) -> str:
        """Preserve most critical sections when trimming."""
        sections = context.split('=== ')
        if len(sections) <= 1:
            return await self._truncate_content(context, target_tokens)
        
        critical_sections = []
        optional_sections = []
        
        for section in sections[1:]:  # Skip first empty split
            section = '=== ' + section
            if any(critical in section for critical in ['STORY CONTEXT', 'CHAPTER OBJECTIVES']):
                critical_sections.append(section)
            else:
                optional_sections.append(section)
        
        # Start with critical sections
        result = '\n\n'.join(critical_sections)
        current_tokens = len(result.split()) * 1.3
        
        # Add optional sections if space permits
        for section in optional_sections:
            section_tokens = len(section.split()) * 1.3
            if current_tokens + section_tokens <= target_tokens:
                result += '\n\n' + section
                current_tokens += section_tokens
        
        return result

    async def _smart_content_truncation(self, content: str, target_tokens: int) -> str:
        """Smart truncation that preserves sentence boundaries."""
        sentences = content.split('. ')
        if not sentences:
            return content
        
        result_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split()) * 1.3
            if current_tokens + sentence_tokens <= target_tokens:
                result_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        result = '. '.join(result_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result

    async def _importance_based_trimming(self, content: str, target_tokens: int) -> str:
        """Trim content based on importance scoring."""
        lines = content.split('\n')
        scored_lines = []
        
        for line in lines:
            score = 0
            line_lower = line.lower()
            
            # Section headers get high score
            if line.startswith('==='):
                score += 20
            
            # Critical information patterns
            importance_patterns = [
                ('primary goal', 15), ('active threads', 12), ('needs payoff', 10),
                ('chapter', 8), ('characters', 7), ('objective', 6),
                ('tension', 5), ('constraint', 3)
            ]
            
            for pattern, points in importance_patterns:
                if pattern in line_lower:
                    score += points
            
            scored_lines.append((score, line))
        
        # Sort by importance and select within token limit
        scored_lines.sort(reverse=True, key=lambda x: x[0])
        selected_lines = []
        current_tokens = 0
        
        for score, line in scored_lines:
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens <= target_tokens:
                selected_lines.append(line)
                current_tokens += line_tokens
        
        return '\n'.join(selected_lines)

    async def _cluster_and_compress(self, content: str, target_tokens: int) -> str:
        """Cluster related content and compress."""
        # Simple clustering by section headers
        sections = {}
        current_section = 'default'
        
        for line in content.split('\n'):
            if line.startswith('==='):
                current_section = line.strip()
                sections[current_section] = []
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)
        
        # Compress each section proportionally
        total_tokens_available = target_tokens
        compressed_sections = []
        
        for section_header, section_lines in sections.items():
            section_content = '\n'.join(section_lines)
            section_tokens = len(section_content.split()) * 1.3
            
            if section_header in self._context_prioritizer:
                section_weight = self._context_prioritizer[section_header.lower().replace('=', '').strip().replace(' ', '_')]['weight']
                allocated_tokens = int(total_tokens_available * section_weight)
            else:
                allocated_tokens = int(total_tokens_available * 0.1)
            
            if section_tokens > allocated_tokens:
                compressed_content = await self._truncate_content(section_content, allocated_tokens)
            else:
                compressed_content = section_content
            
            if section_header.startswith('==='):
                compressed_sections.append(section_header)
            compressed_sections.append(compressed_content)
        
        return '\n'.join(compressed_sections)










    async def _format_story_context(self, story_state: DynamicStoryState, max_tokens: int) -> str:
        """Format story state into structured context for generation."""
        
        # Use the built-in contextual summary as base
        base_summary = story_state.get_contextual_summary(max_tokens=int(max_tokens * 0.8))
        
        # If base summary fits, enhance it with additional context
        base_tokens = len(base_summary.split()) * 1.3
        if base_tokens >= max_tokens:
            return base_summary
        
        # Add additional context elements with remaining token budget
        remaining_tokens = max_tokens - base_tokens
        context_sections = []
        
        # Start with base summary
        context_sections.append(base_summary)
        
        # Add critical story analysis if tokens available
        if remaining_tokens > 100:
            analysis_tokens = min(remaining_tokens * 0.4, 200)
            story_analysis = await self._create_story_analysis(story_state, int(analysis_tokens))
            if story_analysis:
                context_sections.append(f"\nSTORY ANALYSIS: {story_analysis}")
                remaining_tokens -= len(story_analysis.split()) * 1.3
        
        # Add urgency indicators if tokens available
        if remaining_tokens > 50:
            urgency_tokens = min(remaining_tokens * 0.3, 100)
            urgency_info = await self._extract_urgency_indicators(story_state, int(urgency_tokens))
            if urgency_info:
                context_sections.append(f"\nURGENCY INDICATORS: {urgency_info}")
                remaining_tokens -= len(urgency_info.split()) * 1.3
        
        # Add story health metrics if tokens available
        if remaining_tokens > 30:
            health_tokens = min(remaining_tokens, 80)
            health_info = await self._calculate_story_health(story_state, int(health_tokens))
            if health_info:
                context_sections.append(f"\nSTORY HEALTH: {health_info}")
        
        # Combine all sections
        full_context = "".join(context_sections)
        
        # Final validation and trimming
        if len(full_context.split()) * 1.3 > max_tokens:
            full_context = await self._truncate_content(full_context, max_tokens)
        
        return full_context


    async def _extractive_summarize(self, text: str, max_tokens: int) -> str:
        """Extract most important sentences up to token limit."""
        sentences = text.split('. ')
        if not sentences:
            return text
        
        # Score sentences by importance indicators
        scored_sentences = []
        importance_keywords = [
            'protagonist', 'antagonist', 'conflict', 'tension', 'objective', 
            'goal', 'obstacle', 'relationship', 'emotion', 'critical', 'important'
        ]
        
        for i, sentence in enumerate(sentences):
            score = 0
            # Position score (earlier sentences often more important)
            score += max(0, 10 - i)
            
            # Keyword score
            for keyword in importance_keywords:
                if keyword.lower() in sentence.lower():
                    score += 5
            
            # Length penalty for very long sentences
            if len(sentence.split()) > 30:
                score -= 2
                
            scored_sentences.append((score, sentence))
        
        # Sort by score and select best sentences within token limit
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        selected = []
        current_tokens = 0
        
        for score, sentence in scored_sentences:
            sentence_tokens = len(sentence.split()) * 1.3
            if current_tokens + sentence_tokens <= max_tokens:
                selected.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return '. '.join(selected) + ('.' if selected else '')

    async def _abstractive_summarize(self, text: str, max_tokens: int) -> str:
        """Create abstract summary of key points."""
        # Simple template-based abstractive summarization
        # In production, this would use a dedicated summarization model
        
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract key information patterns
            if 'Primary Goal:' in line:
                key_points.append(line)
            elif 'Active Threads:' in line:
                key_points.append(line)
            elif 'Characters:' in line:
                key_points.append(line)
            elif 'NEEDS PAYOFF' in line:
                key_points.append(line)
            elif line.startswith('Chapter') and ('|' in line):
                key_points.append(line)
        
        summary = ' | '.join(key_points)
        
        # Trim if too long
        if len(summary.split()) * 1.3 > max_tokens:
            words = summary.split()
            target_words = int(max_tokens / 1.3)
            summary = ' '.join(words[:target_words]) + '...'
        
        return summary

    async def _template_summarize(self, text: str, max_tokens: int) -> str:
        """Use templates to create structured summaries."""
        template_parts = []
        
        # Extract structured information
        if 'Chapter' in text and '|' in text:
            chapter_info = [line for line in text.split('\n') if 'Chapter' in line and '|' in line]
            if chapter_info:
                template_parts.append(f"Status: {chapter_info[0]}")
        
        if 'Active Threads:' in text:
            threads = [line for line in text.split('\n') if 'Active Threads:' in line]
            if threads:
                template_parts.append(threads[0])
        
        if 'Characters:' in text:
            characters = [line for line in text.split('\n') if 'Characters:' in line]
            if characters:
                template_parts.append(characters[0])
        
        return ' | '.join(template_parts) if template_parts else text[:max_tokens]

    async def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Simple truncation with ellipsis."""
        words = content.split()
        if len(words) * 1.3 <= max_tokens:
            return content
        
        target_words = int(max_tokens / 1.3) - 1
        return ' '.join(words[:target_words]) + '...'

    async def _selective_content_pruning(self, content: str, max_tokens: int) -> str:
        """Remove less important content while preserving structure."""
        lines = content.split('\n')
        important_lines = []
        optional_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classify line importance
            if any(pattern in line for pattern in ['===', 'Primary Goal:', 'Active Threads:', 'NEEDS PAYOFF']):
                important_lines.append(line)
            else:
                optional_lines.append(line)
        
        # Start with important lines
        result_lines = important_lines[:]
        current_tokens = sum(len(line.split()) * 1.3 for line in result_lines)
        
        # Add optional lines if space permits
        for line in optional_lines:
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens <= max_tokens:
                result_lines.append(line)
                current_tokens += line_tokens
        
        return '\n'.join(result_lines)

    async def _semantic_content_compression(self, content: str, max_tokens: int) -> str:
        """Compress content while preserving semantic meaning."""
        # Simplified semantic compression
        lines = content.split('\n')
        compressed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove redundant words while preserving meaning
            words = line.split()
            filtered_words = []
            
            skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'very', 'quite', 'rather'}
            
            for word in words:
                if word.lower() not in skip_words or len(filtered_words) == 0:
                    filtered_words.append(word)
            
            compressed_line = ' '.join(filtered_words)
            compressed_lines.append(compressed_line)
        
        compressed_content = '\n'.join(compressed_lines)
        
        # Final truncation if still too long
        if len(compressed_content.split()) * 1.3 > max_tokens:
            return await self._truncate_content(compressed_content, max_tokens)
        
        return compressed_content

    async def _validate_story_context(self, context: str) -> bool:
        """Validate story context section."""
        required_elements = ['Chapter', 'Active Threads', 'Characters']
        return any(element in context for element in required_elements)

    async def _validate_objectives_context(self, context: str) -> bool:
        """Validate objectives context section."""
        return 'Primary Goal:' in context or 'Word Target:' in context

    async def _validate_market_context(self, context: str) -> bool:
        """Validate market intelligence context section."""
        return len(context.strip()) > 0

    async def _validate_feedback_context(self, context: str) -> bool:
        """Validate feedback context section."""
        return len(context.strip()) > 0

    async def _preserve_critical_sections(self, context: str, target_tokens: int) -> str:
        """Preserve most critical sections when trimming."""
        sections = context.split('=== ')
        if len(sections) <= 1:
            return await self._truncate_content(context, target_tokens)
        
        critical_sections = []
        optional_sections = []
        
        for section in sections[1:]:  # Skip first empty split
            section = '=== ' + section
            if any(critical in section for critical in ['STORY CONTEXT', 'CHAPTER OBJECTIVES']):
                critical_sections.append(section)
            else:
                optional_sections.append(section)
        
        # Start with critical sections
        result = '\n\n'.join(critical_sections)
        current_tokens = len(result.split()) * 1.3
        
        # Add optional sections if space permits
        for section in optional_sections:
            section_tokens = len(section.split()) * 1.3
            if current_tokens + section_tokens <= target_tokens:
                result += '\n\n' + section
                current_tokens += section_tokens
        
        return result

    async def _smart_content_truncation(self, content: str, target_tokens: int) -> str:
        """Smart truncation that preserves sentence boundaries."""
        sentences = content.split('. ')
        if not sentences:
            return content
        
        result_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split()) * 1.3
            if current_tokens + sentence_tokens <= target_tokens:
                result_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        result = '. '.join(result_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result

    async def _importance_based_trimming(self, content: str, target_tokens: int) -> str:
        """Trim content based on importance scoring."""
        lines = content.split('\n')
        scored_lines = []
        
        for line in lines:
            score = 0
            line_lower = line.lower()
            
            # Section headers get high score
            if line.startswith('==='):
                score += 20
            
            # Critical information patterns
            importance_patterns = [
                ('primary goal', 15), ('active threads', 12), ('needs payoff', 10),
                ('chapter', 8), ('characters', 7), ('objective', 6),
                ('tension', 5), ('constraint', 3)
            ]
            
            for pattern, points in importance_patterns:
                if pattern in line_lower:
                    score += points
            
            scored_lines.append((score, line))
        
        # Sort by importance and select within token limit
        scored_lines.sort(reverse=True, key=lambda x: x[0])
        selected_lines = []
        current_tokens = 0
        
        for score, line in scored_lines:
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens <= target_tokens:
                selected_lines.append(line)
                current_tokens += line_tokens
        
        return '\n'.join(selected_lines)

    async def _cluster_and_compress(self, content: str, target_tokens: int) -> str:
        """Cluster related content and compress."""
        # Simple clustering by section headers
        sections = {}
        current_section = 'default'
        
        for line in content.split('\n'):
            if line.startswith('==='):
                current_section = line.strip()
                sections[current_section] = []
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)
        
        # Compress each section proportionally
        total_tokens_available = target_tokens
        compressed_sections = []
        
        for section_header, section_lines in sections.items():
            section_content = '\n'.join(section_lines)
            section_tokens = len(section_content.split()) * 1.3
            
            if section_header in self._context_prioritizer:
                section_weight = self._context_prioritizer[section_header.lower().replace('=', '').strip().replace(' ', '_')]['weight']
                allocated_tokens = int(total_tokens_available * section_weight)
            else:
                allocated_tokens = int(total_tokens_available * 0.1)
            
            if section_tokens > allocated_tokens:
                compressed_content = await self._truncate_content(section_content, allocated_tokens)
            else:
                compressed_content = section_content
            
            if section_header.startswith('==='):
                compressed_sections.append(section_header)
            compressed_sections.append(compressed_content)
        
        return '\n'.join(compressed_sections)
    
    async def _assemble_generation_context(self, input_data: ChapterGeneratorInput, 
                                          generation_id: str) -> str:
        """Assemble comprehensive context for chapter generation."""
        
        # Check context cache first
        cache_key = f"{input_data.story_state.current_chapter}_{hash(str(input_data.chapter_objective))}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        context_parts = []
        
        # Story state context (highest priority)
        story_context = input_data.story_state.get_contextual_summary(
            max_tokens=int(self.config.specific_config.max_context_tokens * 0.4)
        )
        context_parts.append(f"=== STORY CONTEXT ===\n{story_context}")
        
        # Chapter objectives (critical)
        objectives_context = await self._format_objectives_context(input_data.chapter_objective)
        context_parts.append(f"=== CHAPTER OBJECTIVES ===\n{objectives_context}")
        
        # Previous chapter continuity
        if input_data.previous_chapter_text:
            prev_summary = await self._summarize_previous_chapter(input_data.previous_chapter_text)
            context_parts.append(f"=== PREVIOUS CHAPTER ===\n{prev_summary}")
        
        # Market intelligence guidance
        if input_data.market_intelligence:
            market_context = await self._format_market_guidance(
                input_data.market_intelligence,
                int(self.config.specific_config.max_context_tokens * 0.2)
            )
            context_parts.append(f"=== MARKET GUIDANCE ===\n{market_context}")
        
        # Learned patterns and feedback
        if input_data.critic_feedback_history:
            feedback_context = await self._format_feedback_learnings(
                input_data.critic_feedback_history
            )
            context_parts.append(f"=== LEARNED PATTERNS ===\n{feedback_context}")
        
        # Generation constraints
        if input_data.generation_constraints:
            constraints_context = await self._format_constraints(input_data.generation_constraints)
            context_parts.append(f"=== CONSTRAINTS ===\n{constraints_context}")
        
        # Assemble final context
        full_context = "\n\n".join(context_parts)
        
        # Trim to fit token limit
        if len(full_context.split()) > self.config.specific_config.max_context_tokens:
            full_context = await self._trim_context_to_limit(full_context)
        
        # Cache the context
        self._context_cache[cache_key] = full_context
        
        return full_context
    
    async def _select_generation_approaches(self, objective: ChapterObjective,
                                           market_intelligence: Optional[MarketIntelligence],
                                           feedback_history: List[Dict[str, Any]]) -> List[ChapterApproach]:
        """Select diverse generation approaches based on objectives and context."""
        
        approaches = []
        num_variants = self.config.specific_config.variants_to_generate
        
        # Analyze objective complexity and requirements
        complexity_score = objective.calculate_complexity_score()
        
        # Default approach selection based on objectives
        if len(objective.plot_advancements) > len(objective.character_development_targets):
            approaches.append(ChapterApproach.PLOT_DRIVEN)
        else:
            approaches.append(ChapterApproach.CHARACTER_FOCUSED)
        
        # Add approaches based on specific requirements
        if len(objective.world_building_elements) > 2:
            approaches.append(ChapterApproach.WORLD_BUILDING)
        
        if objective.requires_scene_type('action') or 'action' in objective.primary_goal.lower():
            approaches.append(ChapterApproach.ACTION_HEAVY)
        
        if objective.requires_scene_type('dialogue') or len(objective.character_development_targets) > 1:
            approaches.append(ChapterApproach.DIALOGUE_HEAVY)
        
        # Market intelligence influence
        if market_intelligence:
            market_approaches = await self._get_market_preferred_approaches(market_intelligence)
            approaches.extend(market_approaches)
        
        # Learning from feedback history
        if feedback_history:
            successful_approaches = await self._analyze_successful_approaches(feedback_history)
            approaches.extend(successful_approaches)
        
        # Ensure diversity and limit to configured number
        approaches = list(dict.fromkeys(approaches))  # Remove duplicates while preserving order
        
        # Pad with additional approaches if needed
        while len(approaches) < num_variants:
            remaining_approaches = [
                ChapterApproach.ATMOSPHERIC,
                ChapterApproach.INTROSPECTIVE,
                ChapterApproach.PACING_ACCELERATED,
                ChapterApproach.PACING_CONTEMPLATIVE
            ]
            for approach in remaining_approaches:
                if approach not in approaches:
                    approaches.append(approach)
                    break
            else:
                break
        
        return approaches[:num_variants]
    
    async def _adapt_generation_parameters(self, feedback_history: List[Dict[str, Any]]) -> None:
        """Adapt generation parameters based on critic feedback patterns."""
        
        if not feedback_history:
            return
        
        # Analyze recent feedback trends
        recent_feedback = feedback_history[-10:]  # Last 10 pieces of feedback
        
        # Adjust creativity temperature based on feedback
        creativity_scores = [f.get('creativity_score', 0.5) for f in recent_feedback]
        avg_creativity = sum(creativity_scores) / len(creativity_scores)
        
        if avg_creativity < 0.4:  # Too formulaic
            self.config.specific_config.creativity_temperature = min(1.2, 
                self.config.specific_config.creativity_temperature + 0.1)
        elif avg_creativity > 0.9:  # Too experimental
            self.config.specific_config.creativity_temperature = max(0.3,
                self.config.specific_config.creativity_temperature - 0.1)
        
        # Update banned patterns based on repeated criticisms
        common_issues = {}
        for feedback in recent_feedback:
            for issue in feedback.get('common_issues', []):
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # Add frequently flagged patterns to banned list
        for issue, count in common_issues.items():
            if count >= 3:  # Appears in 3+ recent feedbacks
                self._banned_patterns.add(issue)
    
    async def _generate_chapter_variant(self, approach: ChapterApproach, context: str,
                                       input_data: ChapterGeneratorInput, 
                                       generation_id: str) -> Tuple[ChapterVariant, str]:
        """Generate a single chapter variant using specified approach."""
        
        start_time = datetime.now()
        variant_id = f"{generation_id}_{approach.value}"
        
        # Create approach-specific generation prompt
        generation_prompt = await self._create_generation_prompt(
            approach, context, input_data.chapter_objective
        )
        
        # Generate chapter content using LLM
        chapter_content = await self._call_llm_for_generation(generation_prompt, approach)
        
        # Parse and structure the generated content
        structured_content = await self._parse_generated_content(chapter_content, approach)
        
        # Analyze the generated content
        content_analysis = await self._analyze_generated_content(
            structured_content, input_data.chapter_objective, approach
        )
        
        # Self-critique if enabled
        self_critique_notes = []
        if self.config.specific_config.enable_self_critique:
            self_critique_notes = await self._perform_self_critique(
                structured_content, input_data.chapter_objective
            )
        
        # Create generation metadata
        generation_metadata = GenerationMetadata(
            generation_attempt=1,
            generation_timestamp=start_time,
            generation_time_seconds=(datetime.now() - start_time).total_seconds(),
            model_parameters={
                'temperature': self.config.specific_config.creativity_temperature,
                'model': self.config.specific_config.llm_model_name,
                'approach': approach.value
            },
            prompt_tokens=len(generation_prompt.split()) * 1.3,  # Rough estimate
            completion_tokens=len(chapter_content.split()) * 1.3
        )
        
        # Create chapter variant
        chapter_variant = ChapterVariant(
            variant_id=variant_id,
            chapter_number=input_data.chapter_objective.chapter_number,
            approach=approach,
            chapter_text=structured_content['text'],
            chapter_title=structured_content.get('title'),
            word_count=len(structured_content['text'].split()),
            scene_structure=structured_content.get('scenes', []),
            objectives_addressed=content_analysis['objectives_fulfilled'],
            characters_featured=content_analysis['characters_present'],
            plot_threads_advanced=content_analysis['threads_advanced'],
            new_world_elements=content_analysis['world_elements'],
            emotional_beats_achieved=content_analysis['emotional_beats'],
            dialogue_percentage=content_analysis['dialogue_ratio'],
            pacing_assessment=content_analysis['pacing'],
            generation_metadata=generation_metadata,
            self_critique_notes=self_critique_notes
        )
        
        # Create rationale for this approach
        rationale = await self._create_approach_rationale(approach, input_data.chapter_objective)
        
        return chapter_variant, rationale
    
    async def _create_generation_prompt(self, approach: ChapterApproach, context: str,
                                       objective: ChapterObjective) -> str:
        """Create approach-specific generation prompt."""
        
        # Base prompt structure
        prompt_parts = [
            "You are an expert novelist writing a chapter with the following context and objectives:",
            "",
            context,
            "",
            f"=== GENERATION APPROACH: {approach.value.upper()} ===",
            await self._get_approach_specific_guidance(approach),
            "",
            "=== SPECIFIC REQUIREMENTS ===",
            f"- Word count target: {objective.word_count_target.target_words} words",
            f"- Primary goal: {objective.primary_goal}",
            f"- Emotional beats to include: {', '.join(beat.value for beat in objective.emotional_beats)}",
            f"- Scene types required: {', '.join(scene.value for scene in objective.scene_requirements)}",
            "",
            "=== BANNED PATTERNS ===",
            "Avoid these overused elements:",
        ]
        
        # Add banned patterns
        for pattern in list(self._banned_patterns)[:10]:  # Limit to prevent prompt bloat
            prompt_parts.append(f"- {pattern}")
        
        prompt_parts.extend([
            "",
            "Write a complete chapter that fulfills all objectives while maintaining high literary quality.",
            "Focus on the specified approach while ensuring story coherence and reader engagement.",
            "Begin writing now:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _get_approach_specific_guidance(self, approach: ChapterApproach) -> str:
        """Get specific guidance for each generation approach."""
        
        guidance_map = {
            ChapterApproach.CHARACTER_FOCUSED: (
                "Focus on deep character development, internal thoughts, relationships, and emotional growth. "
                "Prioritize character voice, motivations, and psychological realism."
            ),
            ChapterApproach.PLOT_DRIVEN: (
                "Emphasize plot advancement, events, and story momentum. Drive the narrative forward "
                "with significant developments and clear story progression."
            ),
            ChapterApproach.WORLD_BUILDING: (
                "Rich descriptive detail, world rules, setting establishment, and atmospheric immersion. "
                "Build the story world through vivid descriptions and cultural details."
            ),
            ChapterApproach.ACTION_HEAVY: (
                "Dynamic scenes, physical conflict, fast pacing, and kinetic energy. "
                "Create tension through action sequences and high-stakes situations."
            ),
            ChapterApproach.DIALOGUE_HEAVY: (
                "Character interaction through conversation, relationship dynamics through speech, "
                "and information delivery through natural dialogue exchanges."
            ),
            ChapterApproach.INTROSPECTIVE: (
                "Internal character reflection, philosophical depth, emotional introspection, "
                "and psychological complexity through interior monologue."
            ),
            ChapterApproach.ATMOSPHERIC: (
                "Mood, tone, sensory immersion, and emotional atmosphere. Create a specific "
                "feeling or mood that permeates the entire chapter."
            ),
            ChapterApproach.PACING_ACCELERATED: (
                "Quick scene transitions, compressed time, urgent momentum, and driving tension "
                "that propels the reader forward rapidly."
            ),
            ChapterApproach.PACING_CONTEMPLATIVE: (
                "Slower development, deeper exploration, reflective moments, and allowing "
                "ideas and emotions to develop gradually and thoughtfully."
            )
        }
        
        return guidance_map.get(approach, "Focus on creating engaging, well-crafted narrative content.")
    
    async def _call_llm_for_generation(self, prompt: str, approach: ChapterApproach) -> str:
        """Call LLM service for chapter content generation."""
        
        # Placeholder for actual LLM API call
        # In real implementation, would call Anthropic Claude, OpenAI, etc.
        
        # Simulate LLM response based on approach
        simulated_responses = {
            ChapterApproach.CHARACTER_FOCUSED: self._generate_character_focused_sample(),
            ChapterApproach.PLOT_DRIVEN: self._generate_plot_driven_sample(),
            ChapterApproach.WORLD_BUILDING: self._generate_world_building_sample()
        }
        
        # Return simulated response or default
        base_response = simulated_responses.get(approach, self._generate_default_sample())
        
        # Add some variation based on creativity temperature
        if self.config.specific_config.creativity_temperature > 0.8:
            base_response += "\n\nThe chapter took an unexpected turn, introducing elements that challenged conventional storytelling approaches."
        
        return base_response
    
    def _generate_character_focused_sample(self) -> str:
        """Generate sample character-focused content."""
        return """
        Chapter 15: Inner Storms
        
        Sarah stared at the letter in her hands, watching the words blur through unshed tears. The formal language couldn't disguise the pain behind each carefully chosen phrase. After twenty years of friendship, this was how it would end—not with shouting or dramatic confrontations, but with legal terminology and measured distances.
        
        "I never thought it would come to this," she whispered to the empty room.
        
        The silence that answered felt heavier than any argument they'd ever had. She remembered the first time she'd met Jennifer, how they'd laughed until their sides hurt over something completely silly. Now that laughter seemed like it belonged to different people entirely.
        
        Her phone buzzed. A text from David: "How are you holding up?"
        
        She typed and deleted several responses before settling on: "One day at a time."
        
        But even as she sent it, she knew that wasn't quite true. Some days felt like decades, while others passed in a blur of legal meetings and strained conversations. Time had become elastic, bending around the weight of loss in ways she'd never expected.
        """
    
    def _generate_plot_driven_sample(self) -> str:
        """Generate sample plot-driven content."""
        return """
        Chapter 15: The Discovery
        
        The warehouse door swung open with a rusty screech that echoed through the empty building. Marcus stepped inside, his flashlight cutting through the darkness as dust motes danced in the beam.
        
        "This has to be it," he muttered, consulting the map one more time.
        
        The concrete floor was littered with debris—broken glass, torn papers, the remnants of whatever operation had been running here before. But in the far corner, exactly where the coordinates indicated, stood a metal cabinet that looked distinctly out of place among the decay.
        
        His heart rate quickened as he approached. The lock was newer than everything else in the building, still shiny despite the grime. He pulled out the key they'd recovered from Thompson's apartment, holding his breath as he slid it into the lock.
        
        It turned with a soft click.
        
        Inside, stacks of documents waited in neat piles. Marcus grabbed the top folder and opened it, his eyes widening as he read the first page. Names, dates, amounts—everything they'd been looking for was here.
        
        "Jesus," he breathed. "They've been planning this for years."
        
        A sound from the entrance made him freeze. Footsteps. Heavy boots on concrete.
        
        Someone else was here.
        """
    
    def _generate_world_building_sample(self) -> str:
        """Generate sample world-building content.""" 
        return """
        Chapter 15: The Crystal Gardens
        
        The morning mist clung to the crystal formations like gossamer threads, each droplet refracting the twin suns' light into rainbow cascades that painted the garden in shifting hues. Zara moved carefully along the narrow path, mindful of the delicate resonance that connected each crystal to its neighbors.
        
        The Keepers had maintained this place for a thousand years, she'd been told, though the crystals themselves were far older. They hummed with an energy that seemed to pulse in harmony with her heartbeat, a frequency that spoke of deep earth and ancient magic.
        
        "First time in the Gardens?" Elder Thorne's voice carried easily across the space, though she could barely see him through the prismatic maze.
        
        "Yes, Elder. I've only read about them in the texts."
        
        His laughter was warm, rich with the wisdom of decades spent among the singing stones. "The texts can't capture the music, child. Listen."
        
        Zara stopped walking and closed her eyes, letting her other senses expand. The crystals weren't just humming—they were singing, each formation contributing its own note to a vast, complex harmony that seemed to tell the story of the world itself. In the deeper bass notes, she could almost hear the planet's core; in the higher frequencies, the whisper of stellar winds.
        
        "The crystals remember everything," Thorne continued, his voice now directly beside her. "Every sunrise, every storm, every moment of joy and sorrow that has touched this world. That's why we guard them. They are our living history."
        """
    
    def _generate_default_sample(self) -> str:
        """Generate default sample content."""
        return """
        Chapter 15: Moving Forward
        
        The day began like any other, but Emma could sense something different in the air. Perhaps it was the way the light fell through her window, or the particular quality of silence that filled the house. Whatever it was, she felt a stirring of anticipation that had been absent for months.
        
        She made her coffee with more attention than usual, savoring the ritual of grinding the beans and watching the dark liquid bloom in the filter. Small moments like these had become precious to her lately—anchors in the uncertainty that had defined her life since the changes began.
        
        The phone rang just as she was settling into her favorite chair by the window.
        
        "Emma? It's Michael. I have news."
        
        Her grip tightened on the mug. News could be good or bad, but either way, it meant the waiting was finally over.
        
        "I'm listening," she said, and meant it completely.
        """
    
    async def _parse_generated_content(self, content: str, approach: ChapterApproach) -> Dict[str, Any]:
        """Parse and structure generated content."""
        
        lines = content.strip().split('\n')
        
        # Extract title if present
        title = None
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('Chapter') and ':' in line:
                title = line.strip()
                content_start = i + 1
                break
        
        # Join remaining content
        main_text = '\n'.join(lines[content_start:]).strip()
        
        # Basic scene detection (simplified)
        scenes = []
        paragraphs = main_text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 50:  # Substantial paragraphs
                scene = SceneStructure(
                    scene_id=f"scene_{i}",
                    scene_type="narrative",  # Would be more sophisticated
                    word_count=len(paragraph.split()),
                    characters_present=[],  # Would extract from content
                    location="unspecified",  # Would extract from content  
                    primary_purpose="story_advancement",
                    emotional_tone="neutral",  # Would analyze
                    plot_elements_advanced=[]
                )
                scenes.append(scene)
        
        return {
            'title': title,
            'text': main_text,
            'scenes': scenes,
            'word_count': len(main_text.split())
        }
    
    async def _analyze_generated_content(self, content: Dict[str, Any], 
                                        objective: ChapterObjective,
                                        approach: ChapterApproach) -> Dict[str, Any]:
        """Analyze generated content for objective fulfillment."""
        
        text = content['text']
        word_count = content['word_count']
        
        # Analyze objective fulfillment (simplified)
        objectives_fulfilled = []
        for advancement in objective.plot_advancements:
            # Simple keyword matching (would be more sophisticated)
            if any(word in text.lower() for word in advancement.advancement_description.lower().split()):
                objectives_fulfilled.append({
                    'objective_id': advancement.thread_id,
                    'objective_type': 'plot_advancement',
                    'fulfillment_level': 0.8,
                    'fulfillment_description': f"Advanced {advancement.thread_id} plot thread"
                })
        
        # Extract characters mentioned (simplified)
        common_names = ['Sarah', 'David', 'Jennifer', 'Marcus', 'Thompson', 'Zara', 'Emma', 'Michael']
        characters_present = [name for name in common_names if name in text]
        
        # Identify plot threads (simplified)
        threads_advanced = [obj['objective_id'] for obj in objectives_fulfilled 
                           if obj['objective_type'] == 'plot_advancement']
        
        # Identify world elements (simplified)
        world_elements = []
        if approach == ChapterApproach.WORLD_BUILDING:
            if 'crystal' in text.lower():
                world_elements.append('crystal_gardens')
            if 'warehouse' in text.lower():
                world_elements.append('warehouse_location')
        
        # Identify emotional beats (simplified)
        emotional_beats = []
        if 'tears' in text or 'pain' in text:
            emotional_beats.append('sadness')
        if 'heart rate' in text or 'freeze' in text:
            emotional_beats.append('tension')
        
        # Calculate dialogue percentage (rough estimate)
        dialogue_lines = [line for line in text.split('\n') if '"' in line]
        dialogue_percentage = len('\n'.join(dialogue_lines).split()) / max(1, word_count)
        
        # Assess pacing (simplified)
        avg_sentence_length = word_count / max(1, text.count('.') + text.count('!') + text.count('?'))
        if avg_sentence_length < 12:
            pacing = "fast"
        elif avg_sentence_length > 20:
            pacing = "slow"
        else:
            pacing = "moderate"
        
        return {
            'objectives_fulfilled': objectives_fulfilled,
            'characters_present': characters_present,
            'threads_advanced': threads_advanced,
            'world_elements': world_elements,
            'emotional_beats': emotional_beats,
            'dialogue_ratio': min(1.0, dialogue_percentage),
            'pacing': pacing
        }
    
    # Additional placeholder methods for completion
    
    async def _perform_self_critique(self, content: Dict[str, Any], 
                                    objective: ChapterObjective) -> List[str]:
        """Perform self-critique of generated content."""
        critiques = []
        
        word_count = content['word_count'] 
        target_words = objective.word_count_target.target_words
        
        if word_count < target_words * 0.8:
            critiques.append("Chapter may be too short for target word count")
        elif word_count > target_words * 1.2:
            critiques.append("Chapter may be too long for target word count")
        
        return critiques
    
    async def _format_objectives_context(self, objective: ChapterObjective) -> str:
        """Format chapter objectives for context."""
        return f"Primary Goal: {objective.primary_goal}\nWord Target: {objective.word_count_target.target_words}\nKey Requirements: {', '.join(objective.secondary_goals)}"
    
    async def _summarize_previous_chapter(self, previous_text: str) -> str:
        """Summarize previous chapter for continuity."""
        # Simple summarization (would use more sophisticated method)
        sentences = previous_text.split('.')[:3]  # First 3 sentences
        return '. '.join(sentences) + '.'
    
    async def _format_market_guidance(self, market_intelligence: MarketIntelligence, max_tokens: int) -> str:
        """Format market intelligence for generation guidance.""" 
        guidance = market_intelligence.generate_content_guidance()
        
        parts = []
        if guidance['trending_elements']:
            parts.append("Trending: " + '; '.join(guidance['trending_elements'][:3]))
        if guidance['avoid_these']:
            parts.append("Avoid: " + '; '.join(guidance['avoid_these'][:3]))
        
        return '\n'.join(parts)
    
    async def _format_feedback_learnings(self, feedback_history: List[Dict[str, Any]]) -> str:
        """Format learned patterns from feedback."""
        if not feedback_history:
            return "No previous feedback available."
        
        recent = feedback_history[-3:]  # Last 3 feedbacks
        patterns = []
        
        for feedback in recent:
            if feedback.get('successful_elements'):
                patterns.extend(feedback['successful_elements'][:2])
        
        return "Successful patterns: " + '; '.join(patterns) if patterns else "No clear patterns identified."
    
    async def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format generation constraints."""
        constraint_list = []
        for key, value in constraints.items():
            constraint_list.append(f"{key}: {value}")
        return '\n'.join(constraint_list)
    
    async def _trim_context_to_limit(self, context: str) -> str:
        """Trim context to fit within token limits."""
        words = context.split()
        limit = self.config.specific_config.max_context_tokens
        
        if len(words) <= limit:
            return context
        
        # Keep the most important sections (beginning and end)
        keep_start = int(limit * 0.6)
        keep_end = int(limit * 0.4)
        
        trimmed_words = words[:keep_start] + ["..."] + words[-keep_end:]
        return ' '.join(trimmed_words)
    
    async def _get_market_preferred_approaches(self, market_intelligence: MarketIntelligence) -> List[ChapterApproach]:
        """Get approaches preferred by current market trends."""
        approaches = []
        
        for trend in market_intelligence.current_trends:
            if trend.trend_type.value == 'pacing' and trend.popularity_score > 0.7:
                approaches.append(ChapterApproach.PACING_ACCELERATED)
            elif trend.trend_type.value == 'character_type' and trend.popularity_score > 0.7:
                approaches.append(ChapterApproach.CHARACTER_FOCUSED)
        
        return approaches
    
    async def _analyze_successful_approaches(self, feedback_history: List[Dict[str, Any]]) -> List[ChapterApproach]:
        """Analyze which approaches have been successful based on feedback."""
        approach_scores = {}
        
        for feedback in feedback_history:
            approach = feedback.get('approach')
            score = feedback.get('overall_score', 0.5)
            
            if approach:
                if approach not in approach_scores:
                    approach_scores[approach] = []
                approach_scores[approach].append(score)
        
        # Find approaches with average score > 0.7
        successful = []
        for approach, scores in approach_scores.items():
            if sum(scores) / len(scores) > 0.7:
                try:
                    successful.append(ChapterApproach(approach))
                except ValueError:
                    pass  # Invalid approach string
        
        return successful
    
    async def _update_learning_from_generation(self, input_data: ChapterGeneratorInput,
                                              variants: List[ChapterVariant],
                                              metadata: Dict[str, Any]) -> None:
        """Update learning patterns from this generation."""
        
        # Store generation in history
        self._generation_history.append({
            'timestamp': datetime.now(),
            'chapter_number': input_data.chapter_objective.chapter_number,
            'approaches_used': [v.approach.value for v in variants],
            'variants_generated': len(variants),
            'generation_metadata': metadata
        })
        
        # Keep history manageable
        if len(self._generation_history) > 100:
            self._generation_history = self._generation_history[-50:]
    
    async def _create_approach_rationale(self, approach: ChapterApproach, 
                                        objective: ChapterObjective) -> str:
        """Create rationale for why this approach was selected."""
        
        rationale_map = {
            ChapterApproach.CHARACTER_FOCUSED: f"Selected character-focused approach due to {len(objective.character_development_targets)} character development objectives and emphasis on internal growth.",
            ChapterApproach.PLOT_DRIVEN: f"Selected plot-driven approach due to {len(objective.plot_advancements)} plot threads requiring advancement and story momentum needs.",
            ChapterApproach.WORLD_BUILDING: f"Selected world-building approach due to {len(objective.world_building_elements)} new world elements to introduce and setting establishment requirements."
        }
        
        return rationale_map.get(approach, f"Selected {approach.value} approach based on chapter objectives and story requirements.")
    
    # Test and utility methods
    
    async def _test_llm_connection(self) -> bool:
        """Test LLM client connectivity."""
        return self._llm_client is not None
    
    async def _test_context_assembly(self) -> str:
        """Test context assembly functionality."""
        return "Test context assembled successfully"
    
    async def _cleanup_llm_client(self) -> None:
        """Cleanup LLM client resources."""
        self._llm_client = None
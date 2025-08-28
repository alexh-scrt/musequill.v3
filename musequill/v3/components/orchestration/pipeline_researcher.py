"""
Enhanced Pipeline Researcher Integration

This module provides on-demand web search and data extraction capabilities
that can be integrated into any pipeline component for real-time research.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from tavily import TavilyClient

# Import your existing components
from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.components.researcher.researcher_agent import ResearcherComponent
from musequill.v3.models.researcher_agent_model import ResearchQuery, ResearchResults

logger = logging.getLogger(__name__)


class ResearchPriority(Enum):
    """Priority levels for research requests."""
    URGENT = "urgent"        # Immediate execution required
    HIGH = "high"           # Execute before normal priority
    NORMAL = "normal"       # Standard execution priority  
    LOW = "low"            # Execute when resources available
    BACKGROUND = "background"  # Execute during idle time


class ResearchScope(Enum):
    """Scope of research to perform."""
    QUICK = "quick"         # Single search, basic analysis
    STANDARD = "standard"   # Multiple searches, moderate analysis
    DEEP = "deep"          # Comprehensive multi-query research
    TARGETED = "targeted"   # Focused research on specific aspects


@dataclass
class ResearchContext:
    """Context information for research requests."""
    pipeline_stage: str
    component_name: str
    story_state: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None
    previous_research: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class ResearchRequest:
    """A request for research to be performed."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    scope: ResearchScope = ResearchScope.STANDARD
    priority: ResearchPriority = ResearchPriority.NORMAL
    context: Optional[ResearchContext] = None
    specific_questions: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    max_sources: int = 10
    timeout_seconds: int = 300
    callback_component: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchResponse(BaseModel):
    """Response from research operations."""
    request_id: str
    status: str  # "completed", "failed", "timeout", "cancelled"
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    sources_found: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    recommendations: Optional[List[str]] = None
    follow_up_queries: Optional[List[str]] = None
    completed_at: Optional[datetime] = None


class PipelineResearcherConfig(BaseModel):
    """Configuration for the pipeline researcher."""
    tavily_api_key: str = Field(description="Tavily API key")
    max_concurrent_requests: int = Field(default=5, description="Max concurrent research requests")
    default_timeout: int = Field(default=300, description="Default timeout in seconds")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    quality_threshold: float = Field(default=0.7, description="Minimum quality threshold")
    
    # Integration settings
    auto_trigger_conditions: Dict[str, Any] = Field(
        default_factory=lambda: {
            "market_data_age_hours": 6,
            "plot_inconsistency_threshold": 0.8,
            "character_development_gaps": True,
            "trend_analysis_frequency": "daily"
        }
    )


class PipelineResearcher:
    """
    Enhanced researcher that integrates with the pipeline orchestration system.
    Provides on-demand web search and data extraction for any pipeline component.
    """
    
    def __init__(self, config: PipelineResearcherConfig):
        self.config = config
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.researcher_component = None  # Will be initialized lazily
        self.active_requests: Dict[str, ResearchRequest] = {}
        self.request_queue: List[ResearchRequest] = []
        self.results_cache: Dict[str, Tuple[ResearchResponse, datetime]] = {}
        self.running = False
        
    async def initialize(self):
        """Initialize the researcher component and start processing."""
        # Initialize the core researcher component
        from musequill.v3.components.researcher.researcher_agent import create_researcher_component
        self.researcher_component = create_researcher_component("Pipeline Researcher")
        self.running = True
        
        # Start the request processing loop
        asyncio.create_task(self._process_request_queue())
        logger.info("Pipeline researcher initialized and running")
    
    async def shutdown(self):
        """Shutdown the researcher and cleanup resources."""
        self.running = False
        # Cancel any pending requests
        for request_id in list(self.active_requests.keys()):
            await self.cancel_request(request_id)
        logger.info("Pipeline researcher shut down")
    
    # PUBLIC API METHODS
    
    async def research(
        self,
        query: str,
        scope: ResearchScope = ResearchScope.STANDARD,
        priority: ResearchPriority = ResearchPriority.NORMAL,
        context: Optional[ResearchContext] = None,
        **kwargs
    ) -> ResearchResponse:
        """
        Perform research with specified parameters.
        
        Args:
            query: The research question or topic
            scope: How comprehensive the research should be
            priority: Priority level for execution
            context: Additional context from the calling component
            **kwargs: Additional parameters
            
        Returns:
            ResearchResponse with results and metadata
        """
        request = ResearchRequest(
            query=query,
            scope=scope,
            priority=priority,
            context=context,
            specific_questions=kwargs.get('specific_questions'),
            filters=kwargs.get('filters'),
            max_sources=kwargs.get('max_sources', 10),
            timeout_seconds=kwargs.get('timeout_seconds', self.config.default_timeout)
        )
        
        return await self.submit_request(request)
    
    async def quick_search(self, query: str, max_sources: int = 5) -> ResearchResponse:
        """Perform a quick search for immediate results."""
        return await self.research(
            query=query,
            scope=ResearchScope.QUICK,
            priority=ResearchPriority.HIGH,
            max_sources=max_sources,
            timeout_seconds=60
        )
    
    async def deep_research(
        self,
        topic: str,
        specific_questions: List[str],
        context: Optional[ResearchContext] = None
    ) -> ResearchResponse:
        """Perform comprehensive research on a topic."""
        return await self.research(
            query=topic,
            scope=ResearchScope.DEEP,
            priority=ResearchPriority.NORMAL,
            context=context,
            specific_questions=specific_questions,
            max_sources=20,
            timeout_seconds=600
        )
    
    async def market_intelligence(
        self,
        genre: str,
        market_aspect: str,
        context: Optional[ResearchContext] = None
    ) -> ResearchResponse:
        """Gather market intelligence for a specific genre and aspect."""
        query = f"{genre} fiction {market_aspect} current market trends 2025"
        
        market_questions = [
            f"What are readers currently looking for in {genre} fiction?",
            f"What are common complaints about {genre} books?",
            f"What {genre} books are currently successful and why?",
            f"What trends are emerging in {genre} publishing?"
        ]
        
        return await self.research(
            query=query,
            scope=ResearchScope.TARGETED,
            priority=ResearchPriority.HIGH,
            context=context,
            specific_questions=market_questions,
            filters={"time_range": "recent", "source_types": ["publishing", "reviews", "industry"]},
            max_sources=15
        )
    
    async def plot_research(
        self,
        plot_element: str,
        genre: str,
        story_context: Dict[str, Any]
    ) -> ResearchResponse:
        """Research plot elements and story development techniques."""
        context = ResearchContext(
            pipeline_stage="plot_development",
            component_name="plot_researcher",
            story_state=story_context
        )
        
        query = f"{plot_element} techniques {genre} fiction effective storytelling"
        
        plot_questions = [
            f"How do successful {genre} authors handle {plot_element}?",
            f"What are common mistakes with {plot_element} in {genre}?",
            f"What techniques create engaging {plot_element}?",
            f"How does {plot_element} affect reader engagement?"
        ]
        
        return await self.research(
            query=query,
            scope=ResearchScope.TARGETED,
            context=context,
            specific_questions=plot_questions,
            max_sources=12
        )
    
    # REQUEST MANAGEMENT
    
    async def submit_request(self, request: ResearchRequest) -> ResearchResponse:
        """Submit a research request for processing."""
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for request {request.request_id}")
            return cached_result
        
        # Add to queue
        self.active_requests[request.request_id] = request
        self.request_queue.append(request)
        self.request_queue.sort(key=lambda r: self._get_priority_weight(r.priority))
        
        logger.info(f"Submitted research request {request.request_id}: {request.query}")
        
        # Wait for completion
        return await self._wait_for_completion(request.request_id)
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or active research request."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            self.request_queue = [r for r in self.request_queue if r.request_id != request_id]
            logger.info(f"Cancelled research request {request_id}")
            return True
        return False
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a research request."""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": "active" if request_id in [r.request_id for r in self.request_queue] else "completed",
                "query": request.query,
                "priority": request.priority.value,
                "created_at": request.created_at.isoformat()
            }
        return None
    
    # INTEGRATION HELPERS
    
    def create_story_research_context(
        self,
        pipeline_stage: str,
        component_name: str,
        story_state: Dict[str, Any]
    ) -> ResearchContext:
        """Create research context from story state."""
        return ResearchContext(
            pipeline_stage=pipeline_stage,
            component_name=component_name,
            story_state=story_state,
            market_context=story_state.get('market_intelligence', {}),
            previous_research=story_state.get('research_history', {}),
            constraints=story_state.get('constraints', {})
        )
    
    def should_trigger_research(self, trigger_conditions: Dict[str, Any]) -> bool:
        """Determine if research should be automatically triggered."""
        conditions = self.config.auto_trigger_conditions
        
        # Check market data freshness
        if 'last_market_update' in trigger_conditions:
            last_update = trigger_conditions['last_market_update']
            hours_since = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
            if hours_since > conditions['market_data_age_hours']:
                return True
        
        # Check plot inconsistency threshold
        if 'plot_consistency_score' in trigger_conditions:
            if trigger_conditions['plot_consistency_score'] < conditions['plot_inconsistency_threshold']:
                return True
        
        # Check for character development gaps
        if conditions['character_development_gaps'] and 'character_gaps' in trigger_conditions:
            if trigger_conditions['character_gaps']:
                return True
        
        return False
    
    # PRIVATE METHODS
    
    async def _process_request_queue(self):
        """Main processing loop for research requests."""
        while self.running:
            if not self.request_queue:
                await asyncio.sleep(1)
                continue
            
            # Process up to max_concurrent_requests
            batch_size = min(len(self.request_queue), self.config.max_concurrent_requests)
            current_batch = self.request_queue[:batch_size]
            self.request_queue = self.request_queue[batch_size:]
            
            # Execute batch concurrently
            tasks = [self._execute_request(request) for request in current_batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            await asyncio.sleep(0.1)  # Small delay to prevent tight loop
    
    async def _execute_request(self, request: ResearchRequest) -> ResearchResponse:
        """Execute a single research request."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Convert request to ResearchQuery format
            research_query = self._convert_to_research_query(request)
            
            # Execute using the core researcher component
            researcher_input = {
                'research_id': request.request_id,
                'queries': [research_query],
                'force_refresh': True
            }
            
            result = await self.researcher_component.execute(researcher_input)
            
            # Convert result to ResearchResponse
            response = self._convert_to_response(request, result, start_time)
            
            # Cache successful results
            if response.status == "completed" and self.config.enable_caching:
                cache_key = self._generate_cache_key(request)
                self._cache_result(cache_key, response)
            
            # Cleanup
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            return response
            
        except Exception as e:
            logger.error(f"Research request {request.request_id} failed: {e}")
            response = ResearchResponse(
                request_id=request.request_id,
                status="failed",
                error_message=str(e),
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                completed_at=datetime.now(timezone.utc)
            )
            
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            return response
    
    def _convert_to_research_query(self, request: ResearchRequest) -> ResearchQuery:
        """Convert ResearchRequest to ResearchQuery format."""
        return ResearchQuery(
            category=request.context.component_name if request.context else "general",
            questions=request.specific_questions or [request.query],
            priority="High" if request.priority in [ResearchPriority.URGENT, ResearchPriority.HIGH] else "Medium"
        )
    
    def _convert_to_response(
        self,
        request: ResearchRequest,
        result: Any,
        start_time: datetime
    ) -> ResearchResponse:
        """Convert researcher component result to ResearchResponse."""
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        if hasattr(result, 'error') and result.error:
            return ResearchResponse(
                request_id=request.request_id,
                status="failed",
                error_message=result.error,
                execution_time=execution_time,
                completed_at=datetime.now(timezone.utc)
            )
        
        # Extract meaningful data from result
        summary = self._generate_summary(result)
        sources_found = getattr(result, 'total_sources', 0)
        
        return ResearchResponse(
            request_id=request.request_id,
            status="completed",
            results=result.__dict__ if hasattr(result, '__dict__') else {"raw_result": result},
            summary=summary,
            sources_found=sources_found,
            execution_time=execution_time,
            completed_at=datetime.now(timezone.utc)
        )
    
    def _generate_summary(self, result: Any) -> str:
        """Generate a human-readable summary of research results."""
        if hasattr(result, 'updated_queries') and result.updated_queries:
            # Extract key findings from queries
            findings = []
            for query in result.updated_queries:
                if hasattr(query, 'category'):
                    findings.append(f"Research on {query.category} completed")
            
            return f"Research completed with {len(findings)} areas investigated: " + ", ".join(findings)
        
        return "Research completed successfully"
    
    def _generate_cache_key(self, request: ResearchRequest) -> str:
        """Generate a cache key for a research request."""
        key_data = {
            'query': request.query,
            'scope': request.scope.value,
            'specific_questions': request.specific_questions,
            'filters': request.filters
        }
        return f"research:{hash(json.dumps(key_data, sort_keys=True))}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ResearchResponse]:
        """Get cached result if available and not expired."""
        if not self.config.enable_caching or cache_key not in self.results_cache:
            return None
        
        result, cached_at = self.results_cache[cache_key]
        cache_age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        
        if cache_age_hours > self.config.cache_ttl_hours:
            del self.results_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, response: ResearchResponse):
        """Cache a research result."""
        self.results_cache[cache_key] = (response, datetime.now(timezone.utc))
    
    def _get_priority_weight(self, priority: ResearchPriority) -> int:
        """Get numeric weight for priority sorting."""
        weights = {
            ResearchPriority.URGENT: 0,
            ResearchPriority.HIGH: 1,
            ResearchPriority.NORMAL: 2,
            ResearchPriority.LOW: 3,
            ResearchPriority.BACKGROUND: 4
        }
        return weights.get(priority, 2)
    
    async def _wait_for_completion(self, request_id: str) -> ResearchResponse:
        """Wait for a request to complete and return the result."""
        timeout_seconds = self.active_requests[request_id].timeout_seconds
        start_time = datetime.now(timezone.utc)
        
        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout_seconds:
            if request_id not in self.active_requests:
                # Request completed, find the result
                # This is simplified - in a real implementation, you'd store results
                break
            await asyncio.sleep(0.5)
        
        # Timeout case
        await self.cancel_request(request_id)
        return ResearchResponse(
            request_id=request_id,
            status="timeout",
            error_message=f"Request timed out after {timeout_seconds} seconds",
            execution_time=timeout_seconds,
            completed_at=datetime.now(timezone.utc)
        )


# PIPELINE INTEGRATION HELPERS

class ResearchableMixin:
    """
    Mixin class that can be added to any pipeline component to provide
    research capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.researcher: Optional[PipelineResearcher] = None
    
    def set_researcher(self, researcher: PipelineResearcher):
        """Inject the researcher instance."""
        self.researcher = researcher
    
    async def quick_research(self, query: str) -> Optional[ResearchResponse]:
        """Perform quick research if researcher is available."""
        if self.researcher:
            return await self.researcher.quick_search(query)
        return None
    
    async def contextual_research(
        self,
        query: str,
        story_state: Dict[str, Any],
        scope: ResearchScope = ResearchScope.STANDARD
    ) -> Optional[ResearchResponse]:
        """Perform research with story context."""
        if not self.researcher:
            return None
        
        context = self.researcher.create_story_research_context(
            pipeline_stage=getattr(self, 'pipeline_stage', 'unknown'),
            component_name=getattr(self, 'component_name', self.__class__.__name__),
            story_state=story_state
        )
        
        return await self.researcher.research(query, scope=scope, context=context)


# FACTORY FUNCTION

def create_pipeline_researcher(config: PipelineResearcherConfig) -> PipelineResearcher:
    """Create and initialize a pipeline researcher instance."""
    return PipelineResearcher(config)


# EXAMPLE USAGE

async def example_usage():
    """Example of how to use the enhanced pipeline researcher."""
    
    # Create configuration
    config = PipelineResearcherConfig(
        tavily_api_key="your-api-key-here",
        max_concurrent_requests=3,
        enable_caching=True
    )
    
    # Create and initialize researcher
    researcher = create_pipeline_researcher(config)
    await researcher.initialize()
    
    try:
        # Quick search example
        response = await researcher.quick_search("current romance fiction trends 2025")
        print(f"Quick search completed: {response.summary}")
        
        # Market intelligence example
        market_response = await researcher.market_intelligence(
            genre="thriller",
            market_aspect="reader preferences",
            context=None
        )
        print(f"Market research: {market_response.sources_found} sources found")
        
        # Deep research example
        deep_response = await researcher.deep_research(
            topic="character development techniques",
            specific_questions=[
                "What makes characters memorable?",
                "How do readers connect with protagonists?",
                "What character flaws create engagement?"
            ]
        )
        print(f"Deep research completed in {deep_response.execution_time:.2f}s")
        
    finally:
        await researcher.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())
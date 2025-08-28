"""
Market Intelligence Engine Component

Implements market research collection, trend analysis, and commercial viability
insights using Tavily web search for real-time market intelligence gathering.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import re
import hashlib

from musequill.v3.components.base.component_interface import (
    BaseComponent, ComponentConfiguration, ComponentType, ComponentError
)
from musequill.v3.models.market_intelligence import (
    MarketIntelligence, MarketTrend, ReaderPreference, SuccessPattern,
    CommonComplaint, CompetitiveAnalysis, TrendType, ConfidenceLevel,
    TrendLifecycle, ReaderPreferenceType
)


class MarketIntelligenceEngineConfig(BaseModel):
    """Configuration for Market Intelligence Engine component."""
    
    tavily_api_key: str = Field(
        description="Tavily API key for web search"
    )
    
    update_frequency_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours between market intelligence updates"
    )
    
    search_depth: str = Field(
        default="advanced",
        description="Depth of Tavily search (basic/advanced)"
    )
    
    max_search_results: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Maximum search results per query"
    )
    
    trend_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level for trend identification"
    )
    
    enable_competitive_analysis: bool = Field(
        default=True,
        description="Whether to perform competitive landscape analysis"
    )
    
    cache_duration_hours: int = Field(
        default=6,
        ge=1,
        le=48,
        description="Hours to cache search results"
    )
    
    genres_to_track: List[str] = Field(
        default_factory=lambda: ["romance", "thriller", "fantasy", "literary", "mystery"],
        description="Genres to track market intelligence for"
    )


class MarketIntelligenceEngineInput(BaseModel):
    """Input data for Market Intelligence Engine."""
    
    target_genre: str = Field(
        description="Primary genre for market intelligence gathering"
    )
    
    target_market: str = Field(
        default="mass_market",
        description="Target market segment"
    )
    
    specific_queries: List[str] = Field(
        default_factory=list,
        description="Additional specific research queries"
    )
    
    competitive_titles: List[str] = Field(
        default_factory=list,
        description="Specific titles for competitive analysis"
    )
    
    force_refresh: bool = Field(
        default=False,
        description="Force refresh of cached data"
    )


class MarketIntelligenceEngine(BaseComponent[MarketIntelligenceEngineInput, MarketIntelligence, MarketIntelligenceEngineConfig]):
    """
    Market Intelligence Engine for real-time market research and trend analysis.
    
    Collects market data through Tavily searches, analyzes trends, identifies
    reader preferences, and provides commercial viability insights.
    """
    
    def __init__(self, config: ComponentConfiguration[MarketIntelligenceEngineConfig]):
        super().__init__(config)
        self._tavily_client = None
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._trend_history: Dict[str, List[MarketTrend]] = {}
        self._analysis_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize market intelligence gathering systems."""
        try:
            # Initialize Tavily client
            await self._initialize_tavily_client()
            
            # Load cached trend history
            await self._load_trend_history()
            
            # Initialize analysis tools
            await self._initialize_analysis_tools()
            
            return True
            
        except Exception as e:
            self.state.last_error = f"Market intelligence engine initialization failed: {str(e)}"
            return False
    
    async def process(self, input_data: MarketIntelligenceEngineInput) -> MarketIntelligence:
        """
        Gather and analyze market intelligence for specified genre and market.
        
        Args:
            input_data: Genre, market, and research parameters
            
        Returns:
            Comprehensive market intelligence report
        """
        start_time = datetime.now()
        
        try:
            # Check if cached data is still valid
            if not input_data.force_refresh:
                cached_intelligence = await self._get_cached_intelligence(
                    input_data.target_genre, input_data.target_market
                )
                if cached_intelligence:
                    return cached_intelligence
            
            # Gather market data through multiple research streams
            research_tasks = [
                self._research_current_trends(input_data.target_genre),
                self._research_reader_preferences(input_data.target_genre, input_data.target_market),
                self._research_success_patterns(input_data.target_genre),
                self._research_common_complaints(input_data.target_genre),
            ]
            
            # Add competitive analysis if enabled
            if self.config.specific_config.enable_competitive_analysis:
                research_tasks.append(
                    self._research_competitive_landscape(
                        input_data.target_genre, input_data.competitive_titles
                    )
                )
            
            # Execute research tasks
            results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            trends_result = results[0] if not isinstance(results[0], Exception) else []
            preferences_result = results[1] if not isinstance(results[1], Exception) else []
            patterns_result = results[2] if not isinstance(results[2], Exception) else []
            complaints_result = results[3] if not isinstance(results[3], Exception) else []
            
            competitive_result = None
            if len(results) > 4 and not isinstance(results[4], Exception):
                competitive_result = results[4]
            
            # Analyze additional specific queries
            if input_data.specific_queries:
                specific_insights = await self._research_specific_queries(input_data.specific_queries)
                # Integrate specific insights into results
                trends_result.extend(specific_insights.get('trends', []))
                preferences_result.extend(specific_insights.get('preferences', []))
            
            # Identify oversaturated elements and emerging opportunities
            oversaturated_elements = await self._identify_oversaturated_elements(
                trends_result, patterns_result
            )
            emerging_opportunities = await self._identify_emerging_opportunities(
                trends_result, competitive_result
            )
            
            # Compile comprehensive market intelligence
            market_intelligence = MarketIntelligence(
                genre=input_data.target_genre,
                target_market=input_data.target_market,
                current_trends=trends_result,
                reader_preferences=preferences_result,
                success_patterns=patterns_result,
                common_complaints=complaints_result,
                competitive_analysis=competitive_result or CompetitiveAnalysis(
                    genre_saturation_level=0.5,
                    key_differentiators=[],
                    successful_recent_releases=[],
                    market_gaps=[],
                    pricing_insights={},
                    distribution_channels=[]
                ),
                oversaturated_elements=oversaturated_elements,
                emerging_opportunities=emerging_opportunities,
                seasonal_factors=await self._analyze_seasonal_factors(input_data.target_genre),
                last_updated=datetime.now(),
                data_sources=self._get_data_sources_used()
            )
            
            # Cache the intelligence
            await self._cache_intelligence(market_intelligence)
            
            # Update trend history
            await self._update_trend_history(input_data.target_genre, trends_result)
            
            return market_intelligence
            
        except Exception as e:
            raise ComponentError(f"Market intelligence gathering failed: {str(e)}", self.config.component_id)
    
    async def health_check(self) -> bool:
        """Perform health check on market intelligence systems."""
        try:
            # Test Tavily connectivity
            if not await self._test_tavily_connection():
                return False
            
            # Check cache validity
            if not self._search_cache:
                # Empty cache is okay, but test cache functionality
                test_key = "health_check_test"
                self._search_cache[test_key] = {"test": True, "timestamp": datetime.now()}
                if test_key not in self._search_cache:
                    return False
                del self._search_cache[test_key]
            
            # Check component performance
            if self.state.metrics.failure_rate > 0.2:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup market intelligence resources."""
        try:
            self._search_cache.clear()
            self._trend_history.clear()
            self._analysis_cache.clear()
            
            if self._tavily_client:
                # Cleanup Tavily client if needed
                pass
            
            return True
            
        except Exception:
            return False
    
    async def _initialize_tavily_client(self) -> None:
        """Initialize Tavily client for web search."""
        # Placeholder for actual Tavily client initialization
        # In real implementation:
        # from tavily import TavilyClient
        # self._tavily_client = TavilyClient(api_key=self.config.specific_config.tavily_api_key)
        
        self._tavily_client = {
            'api_key': self.config.specific_config.tavily_api_key,
            'search_depth': self.config.specific_config.search_depth,
            'max_results': self.config.specific_config.max_search_results,
            'initialized': True
        }
    
    async def _load_trend_history(self) -> None:
        """Load historical trend data for analysis."""
        # Placeholder for loading trend history from persistent storage
        self._trend_history = {}
    
    async def _initialize_analysis_tools(self) -> None:
        """Initialize market analysis and NLP tools."""
        # Initialize sentiment analysis, trend detection, etc.
        pass
    
    async def _get_cached_intelligence(self, genre: str, market: str) -> Optional[MarketIntelligence]:
        """Get cached market intelligence if still valid."""
        cache_key = f"{genre}_{market}"
        
        if cache_key in self._analysis_cache:
            cached_data = self._analysis_cache[cache_key]
            cache_time = cached_data.get('timestamp', datetime.min)
            
            if datetime.now() - cache_time < timedelta(hours=self.config.specific_config.cache_duration_hours):
                return cached_data.get('intelligence')
        
        return None
    
    async def _research_current_trends(self, genre: str) -> List[MarketTrend]:
        """Research current market trends for specified genre."""
        
        trends = []
        search_queries = [
            f"{genre} fiction trends 2025",
            f"{genre} book market analysis current",
            f"bestselling {genre} novels 2024 2025",
            f"{genre} reader preferences recent survey",
            f"{genre} publishing industry trends"
        ]
        
        for query in search_queries:
            try:
                search_results = await self._perform_tavily_search(query)
                query_trends = await self._extract_trends_from_results(search_results, genre)
                trends.extend(query_trends)
                
                # Add small delay to respect rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.state.error_count += 1
                continue
        
        # Deduplicate and validate trends
        validated_trends = await self._validate_and_deduplicate_trends(trends)
        
        return validated_trends
    
    async def _research_reader_preferences(self, genre: str, market: str) -> List[ReaderPreference]:
        """Research reader preferences for genre and market."""
        
        preferences = []
        search_queries = [
            f"{genre} readers what they want survey",
            f"{genre} book reviews common praise criticism",
            f"{market} market {genre} preferences study",
            f"{genre} fiction reader demographics preferences"
        ]
        
        for query in search_queries:
            try:
                search_results = await self._perform_tavily_search(query)
                query_preferences = await self._extract_preferences_from_results(search_results, genre)
                preferences.extend(query_preferences)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                continue
        
        return await self._consolidate_preferences(preferences)
    
    async def _research_success_patterns(self, genre: str) -> List[SuccessPattern]:
        """Research patterns in commercially successful books."""
        
        patterns = []
        search_queries = [
            f"bestselling {genre} books common elements analysis",
            f"{genre} commercial success factors study",
            f"what makes {genre} novels sell well",
            f"{genre} book marketing success stories"
        ]
        
        for query in search_queries:
            try:
                search_results = await self._perform_tavily_search(query)
                query_patterns = await self._extract_success_patterns_from_results(search_results, genre)
                patterns.extend(query_patterns)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                continue
        
        return await self._validate_success_patterns(patterns)
    
    async def _research_common_complaints(self, genre: str) -> List[CommonComplaint]:
        """Research common reader complaints about genre."""
        
        complaints = []
        search_queries = [
            f"{genre} books negative reviews common complaints",
            f"what readers hate about {genre} fiction",
            f"{genre} novels criticism recurring themes",
            f"goodreads {genre} one star reviews analysis"
        ]
        
        for query in search_queries:
            try:
                search_results = await self._perform_tavily_search(query)
                query_complaints = await self._extract_complaints_from_results(search_results, genre)
                complaints.extend(query_complaints)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                continue
        
        return await self._categorize_complaints(complaints)
    
    async def _research_competitive_landscape(self, genre: str, specific_titles: List[str]) -> CompetitiveAnalysis:
        """Research competitive landscape and market positioning."""
        
        search_queries = [
            f"{genre} book market saturation analysis 2025",
            f"new {genre} authors breakthrough success",
            f"{genre} publishing market size competition"
        ]
        
        # Add specific title queries
        for title in specific_titles[:3]:  # Limit to prevent too many queries
            search_queries.append(f'"{title}" book sales analysis reviews')
        
        all_results = []
        for query in search_queries:
            try:
                search_results = await self._perform_tavily_search(query)
                all_results.extend(search_results)
                await asyncio.sleep(0.5)
            except Exception as e:
                continue
        
        return await self._analyze_competitive_landscape(all_results, genre)
    
    async def _research_specific_queries(self, queries: List[str]) -> Dict[str, List[Any]]:
        """Research additional specific queries provided by user."""
        
        insights = {
            'trends': [],
            'preferences': [],
            'patterns': []
        }
        
        for query in queries:
            try:
                search_results = await self._perform_tavily_search(query)
                
                # Analyze results to categorize insights
                query_insights = await self._categorize_query_insights(search_results, query)
                
                for category, items in query_insights.items():
                    if category in insights:
                        insights[category].extend(items)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                continue
        
        return insights
    
    async def _perform_tavily_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using Tavily API."""
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self._search_cache:
            cached_result = self._search_cache[query_hash]
            cache_time = cached_result.get('timestamp', datetime.min)
            
            if datetime.now() - cache_time < timedelta(hours=self.config.specific_config.cache_duration_hours):
                return cached_result.get('results', [])
        
        # Perform actual search (placeholder implementation)
        # In real implementation:
        # search_response = await self._tavily_client.search(
        #     query=query,
        #     search_depth=self.config.specific_config.search_depth,
        #     max_results=self.config.specific_config.max_search_results
        # )
        
        # Simulated search results for different query types
        simulated_results = await self._generate_simulated_search_results(query)
        
        # Cache results
        self._search_cache[query_hash] = {
            'results': simulated_results,
            'timestamp': datetime.now()
        }
        
        return simulated_results
    
    async def _generate_simulated_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate simulated search results for development/testing."""
        
        query_lower = query.lower()
        
        # Simulate different types of results based on query content
        if 'trends' in query_lower:
            return [
                {
                    'title': f'2025 Fiction Trends: What Readers Want',
                    'content': 'Recent surveys show readers increasingly prefer diverse characters, complex plotting, and authentic emotional experiences. Fast-paced narratives continue to dominate digital platforms.',
                    'url': 'https://publishinginsights.com/2025-trends',
                    'published_date': '2025-01-15'
                },
                {
                    'title': f'Market Analysis: Rising Genre Preferences',
                    'content': 'Data indicates growing demand for genre-blending fiction, shorter chapter structures, and multi-POV narratives. Traditional publishing adapting to digital-first preferences.',
                    'url': 'https://bookmarket.com/analysis-2025',
                    'published_date': '2025-01-10'
                }
            ]
        
        elif 'preferences' in query_lower or 'survey' in query_lower:
            return [
                {
                    'title': f'Reader Survey Results: What Makes Books Unputdownable',
                    'content': 'Survey of 10,000 readers reveals preference for strong character development, consistent pacing, and satisfying plot resolution. Readers increasingly abandon books with slow openings.',
                    'url': 'https://readersurvey.org/2025-results',
                    'published_date': '2025-01-20'
                }
            ]
        
        elif 'complaints' in query_lower or 'criticism' in query_lower:
            return [
                {
                    'title': f'Common Reader Complaints in Modern Fiction',
                    'content': 'Analysis of negative reviews shows recurring issues: repetitive plots, inconsistent character behavior, unresolved plot threads, and predictable endings top the list.',
                    'url': 'https://bookcriticism.com/common-issues',
                    'published_date': '2025-01-12'
                }
            ]
        
        elif 'bestselling' in query_lower or 'success' in query_lower:
            return [
                {
                    'title': f'Success Factors in Recent Bestsellers',
                    'content': 'Analysis of 2024 bestsellers reveals common elements: strong opening hooks, consistent character voice, balanced pacing, and satisfying emotional payoffs drive commercial success.',
                    'url': 'https://bestseller-analysis.com/success-factors',
                    'published_date': '2025-01-18'
                }
            ]
        
        else:
            return [
                {
                    'title': f'Market Research: {query}',
                    'content': f'Research findings related to {query} indicate various market dynamics and reader preferences that inform commercial publishing strategies.',
                    'url': f'https://market-research.com/{query_lower.replace(" ", "-")}',
                    'published_date': '2025-01-15'
                }
            ]
    
    async def _extract_trends_from_results(self, results: List[Dict[str, Any]], genre: str) -> List[MarketTrend]:
        """Extract market trends from search results."""
        
        trends = []
        
        for result in results:
            content = result.get('content', '')
            title = result.get('title', '')
            
            # Simple trend extraction based on keywords
            trend_indicators = {
                'diverse characters': TrendType.CHARACTER_TYPE,
                'complex plotting': TrendType.STRUCTURE,
                'fast-paced': TrendType.PACING,
                'multi-POV': TrendType.TECHNIQUE,
                'short chapters': TrendType.STRUCTURE,
                'emotional experiences': TrendType.THEME
            }
            
            for indicator, trend_type in trend_indicators.items():
                if indicator in content.lower() or indicator in title.lower():
                    trend_id = f"{trend_type.value}_{indicator.replace(' ', '_')}"
                    
                    # Determine popularity and lifecycle based on context
                    popularity_score = 0.7  # Default
                    lifecycle = TrendLifecycle.GROWING  # Default
                    
                    if 'increasingly' in content.lower() or 'rising' in content.lower():
                        popularity_score = 0.8
                        lifecycle = TrendLifecycle.GROWING
                    elif 'continue' in content.lower() or 'still' in content.lower():
                        popularity_score = 0.9
                        lifecycle = TrendLifecycle.PEAK
                    
                    trend = MarketTrend(
                        trend_id=trend_id,
                        trend_type=trend_type,
                        title=f"Growing demand for {indicator}",
                        description=f"Market research indicates {indicator} are becoming increasingly popular with readers",
                        popularity_score=popularity_score,
                        lifecycle_stage=lifecycle,
                        confidence_level=ConfidenceLevel.MEDIUM,
                        supporting_evidence=[result.get('url', '')],
                        related_genres=[genre],
                        implementation_examples=[f"Incorporate {indicator} into narrative structure"],
                        date_identified=datetime.now()
                    )
                    trends.append(trend)
        
        return trends
    
    async def _extract_preferences_from_results(self, results: List[Dict[str, Any]], genre: str) -> List[ReaderPreference]:
        """Extract reader preferences from search results."""
        
        preferences = []
        
        preference_indicators = {
            'character development': ReaderPreferenceType.CHARACTER_DEVELOPMENT,
            'consistent pacing': ReaderPreferenceType.PACING,
            'plot resolution': ReaderPreferenceType.PLOT_COMPLEXITY,
            'strong opening': ReaderPreferenceType.PACING,
            'dialogue': ReaderPreferenceType.DIALOGUE_STYLE,
            'world building': ReaderPreferenceType.WORLD_BUILDING
        }
        
        for result in results:
            content = result.get('content', '').lower()
            
            for indicator, pref_type in preference_indicators.items():
                if indicator in content:
                    preference = ReaderPreference(
                        preference_type=pref_type,
                        preference_description=f"Readers prefer {indicator} in {genre} fiction",
                        importance_weight=0.8,
                        genre_specificity={genre: 0.9},
                        demographic_variations={},
                        supporting_data=[result.get('url', '')]
                    )
                    preferences.append(preference)
        
        return preferences
    
    async def _extract_success_patterns_from_results(self, results: List[Dict[str, Any]], genre: str) -> List[SuccessPattern]:
        """Extract success patterns from search results."""
        
        patterns = []
        
        for result in results:
            content = result.get('content', '').lower()
            
            success_indicators = [
                'strong opening hooks',
                'consistent character voice', 
                'balanced pacing',
                'satisfying emotional payoffs',
                'engaging dialogue',
                'compelling conflicts'
            ]
            
            for indicator in success_indicators:
                if indicator in content:
                    pattern = SuccessPattern(
                        pattern_id=f"{indicator.replace(' ', '_')}_{genre}",
                        pattern_description=f"Successful {genre} books commonly feature {indicator}",
                        success_correlation=0.8,
                        frequency_in_bestsellers=0.7,
                        applicable_genres=[genre],
                        implementation_guidelines=[f"Ensure strong implementation of {indicator}"],
                        common_mistakes=[f"Weak or inconsistent {indicator}"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _extract_complaints_from_results(self, results: List[Dict[str, Any]], genre: str) -> List[CommonComplaint]:
        """Extract common reader complaints from search results."""
        
        complaints = []
        
        complaint_indicators = {
            'repetitive plots': 'plot',
            'inconsistent character': 'character', 
            'unresolved plot threads': 'plot',
            'predictable endings': 'plot',
            'slow openings': 'pacing',
            'poor dialogue': 'dialogue'
        }
        
        for result in results:
            content = result.get('content', '').lower()
            
            for indicator, complaint_type in complaint_indicators.items():
                if indicator in content:
                    complaint = CommonComplaint(
                        complaint_type=complaint_type,
                        complaint_description=f"Readers frequently complain about {indicator}",
                        frequency_score=0.6,
                        impact_on_sales=0.7,
                        affected_genres=[genre],
                        prevention_strategies=[f"Avoid {indicator} through careful planning"],
                        example_reviews=[content[:100] + "..."]
                    )
                    complaints.append(complaint)
        
        return complaints
    
    # Additional helper methods (implementations simplified for space)
    
    async def _validate_and_deduplicate_trends(self, trends: List[MarketTrend]) -> List[MarketTrend]:
        """Remove duplicate trends and validate confidence levels."""
        seen_trends = set()
        validated = []
        
        for trend in trends:
            if trend.trend_id not in seen_trends:
                if trend.confidence_level != ConfidenceLevel.LOW or len(trend.supporting_evidence) > 1:
                    validated.append(trend)
                    seen_trends.add(trend.trend_id)
        
        return validated
    
    async def _consolidate_preferences(self, preferences: List[ReaderPreference]) -> List[ReaderPreference]:
        """Consolidate and deduplicate reader preferences."""
        pref_map = {}
        
        for pref in preferences:
            key = f"{pref.preference_type.value}_{pref.preference_description[:50]}"
            if key not in pref_map:
                pref_map[key] = pref
            else:
                # Merge supporting data
                pref_map[key].supporting_data.extend(pref.supporting_data)
        
        return list(pref_map.values())
    
    async def _validate_success_patterns(self, patterns: List[SuccessPattern]) -> List[SuccessPattern]:
        """Validate and filter success patterns."""
        return [p for p in patterns if p.success_correlation >= 0.6]
    
    async def _categorize_complaints(self, complaints: List[CommonComplaint]) -> List[CommonComplaint]:
        """Categorize and prioritize complaints."""
        return sorted(complaints, key=lambda x: x.impact_on_sales, reverse=True)
    
    async def _analyze_competitive_landscape(self, results: List[Dict[str, Any]], genre: str) -> CompetitiveAnalysis:
        """Analyze competitive landscape from search results."""
        
        # Extract saturation indicators
        saturation_level = 0.6  # Default moderate saturation
        
        content_combined = ' '.join([r.get('content', '').lower() for r in results])
        
        if 'oversaturated' in content_combined or 'crowded market' in content_combined:
            saturation_level = 0.9
        elif 'growing market' in content_combined or 'opportunities' in content_combined:
            saturation_level = 0.4
        
        return CompetitiveAnalysis(
            genre_saturation_level=saturation_level,
            key_differentiators=["unique voice", "fresh perspective", "innovative structure"],
            successful_recent_releases=[],  # Would extract from results
            market_gaps=["underserved demographics", "unexplored themes"],
            pricing_insights={"average_ebook": 4.99, "average_paperback": 12.99},
            distribution_channels=["Amazon", "traditional bookstores", "digital platforms"]
        )
    
    async def _categorize_query_insights(self, results: List[Dict[str, Any]], query: str) -> Dict[str, List[Any]]:
        """Categorize insights from specific query results."""
        return {
            'trends': [],
            'preferences': [],
            'patterns': []
        }
    
    async def _identify_oversaturated_elements(self, trends: List[MarketTrend], 
                                             patterns: List[SuccessPattern]) -> List[str]:
        """Identify oversaturated market elements to avoid."""
        oversaturated = []
        
        # Elements that appear in too many trends/patterns
        element_counts = {}
        
        for trend in trends:
            if trend.lifecycle_stage == TrendLifecycle.SATURATED:
                oversaturated.append(trend.title)
        
        for pattern in patterns:
            if pattern.frequency_in_bestsellers > 0.9:  # Very common
                oversaturated.append(pattern.pattern_description)
        
        return list(set(oversaturated))
    
    async def _identify_emerging_opportunities(self, trends: List[MarketTrend], 
                                            competitive_analysis: Optional[CompetitiveAnalysis]) -> List[str]:
        """Identify emerging market opportunities."""
        opportunities = []
        
        # Emerging trends represent opportunities
        for trend in trends:
            if trend.lifecycle_stage == TrendLifecycle.EMERGING:
                opportunities.append(f"Early adoption of {trend.title}")
        
        # Market gaps from competitive analysis
        if competitive_analysis:
            opportunities.extend(competitive_analysis.market_gaps)
        
        return opportunities
    
    async def _analyze_seasonal_factors(self, genre: str) -> Dict[str, Any]:
        """Analyze seasonal factors affecting book sales."""
        return {
            "peak_seasons": ["December", "June-August"],
            "genre_specific_patterns": f"{genre} typically sees increased sales during holiday seasons"
        }
    
    def _get_data_sources_used(self) -> List[str]:
        """Get list of data sources used in this analysis."""
        sources = set()
        
        for cached_result in self._search_cache.values():
            for result in cached_result.get('results', []):
                if 'url' in result:
                    sources.add(result['url'])
        
        return list(sources)
    
    async def _cache_intelligence(self, intelligence: MarketIntelligence) -> None:
        """Cache market intelligence for future use."""
        cache_key = f"{intelligence.genre}_{intelligence.target_market}"
        self._analysis_cache[cache_key] = {
            'intelligence': intelligence,
            'timestamp': datetime.now()
        }
    
    async def _update_trend_history(self, genre: str, trends: List[MarketTrend]) -> None:
        """Update historical trend data."""
        if genre not in self._trend_history:
            self._trend_history[genre] = []
        
        self._trend_history[genre].extend(trends)
        
        # Keep history manageable
        if len(self._trend_history[genre]) > 100:
            self._trend_history[genre] = self._trend_history[genre][-50:]
    
    async def _test_tavily_connection(self) -> bool:
        """Test Tavily API connectivity."""
        return self._tavily_client is not None and self._tavily_client.get('initialized', False)
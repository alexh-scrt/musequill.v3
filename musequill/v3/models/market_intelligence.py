"""
Market Intelligence Model

Manages market research data, trend analysis, and commercial viability insights
from Tavily searches for adaptive content generation optimization.
"""
# pylint: disable=locally-disabled, fixme, line-too-long, no-member

from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional, Any
from datetime import datetime, date
from enum import Enum


class TrendType(str, Enum):
    """Types of market trends tracked."""
    TECHNIQUE = "technique"
    THEME = "theme"
    STYLE = "style"
    STRUCTURE = "structure"
    CHARACTER_TYPE = "character_type"
    SETTING = "setting"
    PLOT_DEVICE = "plot_device"
    PACING = "pacing"


class ConfidenceLevel(str, Enum):
    """Confidence levels for market intelligence data."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrendLifecycle(str, Enum):
    """Lifecycle stage of market trends."""
    EMERGING = "emerging"
    GROWING = "growing"
    PEAK = "peak"
    DECLINING = "declining"
    SATURATED = "saturated"


class ReaderPreferenceType(str, Enum):
    """Types of reader preferences tracked."""
    PACING = "pacing"
    CHARACTER_DEVELOPMENT = "character_development"
    PLOT_COMPLEXITY = "plot_complexity"
    EMOTIONAL_INTENSITY = "emotional_intensity"
    WORLD_BUILDING = "world_building"
    DIALOGUE_STYLE = "dialogue_style"
    CHAPTER_LENGTH = "chapter_length"
    NARRATIVE_VOICE = "narrative_voice"


class MarketTrend(BaseModel):
    """A specific market trend identified through research."""
    
    trend_id: str = Field(
        description="Unique identifier for this trend"
    )
    
    trend_type: TrendType = Field(
        description="Category of trend"
    )
    
    title: str = Field(
        min_length=10,
        max_length=200,
        description="Descriptive title of the trend"
    )
    
    description: str = Field(
        min_length=50,
        max_length=1000,
        description="Detailed description of the trend"
    )
    
    popularity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Current popularity level (0.0-1.0)"
    )
    
    lifecycle_stage: TrendLifecycle = Field(
        description="Current lifecycle stage of this trend"
    )
    
    confidence_level: ConfidenceLevel = Field(
        description="Confidence in this trend analysis"
    )
    
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting this trend (URLs, examples, data points)"
    )
    
    related_genres: List[str] = Field(
        default_factory=list,
        description="Genres where this trend is most prominent"
    )
    
    implementation_examples: List[str] = Field(
        default_factory=list,
        description="Specific examples of how this trend is implemented"
    )
    
    date_identified: datetime = Field(
        default_factory=datetime.now,
        description="When this trend was first identified"
    )
    
    last_validated: datetime = Field(
        default_factory=datetime.now,
        description="When this trend was last validated with fresh data"
    )
    
    projected_duration: Optional[str] = Field(
        default=None,
        description="Projected timeline for trend relevance"
    )


class ReaderPreference(BaseModel):
    """Reader preference data for specific aspects of fiction."""
    
    preference_type: ReaderPreferenceType = Field(
        description="Type of preference being tracked"
    )
    
    preference_description: str = Field(
        min_length=20,
        max_length=300,
        description="Description of what readers prefer"
    )
    
    importance_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="How important this preference is to readers (0.0-1.0)"
    )
    
    genre_specificity: Dict[str, float] = Field(
        default_factory=dict,
        description="How much this preference varies by genre"
    )
    
    demographic_variations: Dict[str, float] = Field(
        default_factory=dict,
        description="How preference varies by reader demographics"
    )
    
    supporting_data: List[str] = Field(
        default_factory=list,
        description="Data sources supporting this preference analysis"
    )


class SuccessPattern(BaseModel):
    """Pattern identified in successful books/content."""
    
    pattern_id: str = Field(
        description="Unique identifier for this success pattern"
    )
    
    pattern_description: str = Field(
        min_length=30,
        max_length=500,
        description="Description of the successful pattern"
    )
    
    success_correlation: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of correlation with commercial success"
    )
    
    frequency_in_bestsellers: float = Field(
        ge=0.0,
        le=1.0,
        description="How often this pattern appears in bestsellers"
    )
    
    applicable_genres: List[str] = Field(
        default_factory=list,
        description="Genres where this pattern is most effective"
    )
    
    implementation_guidelines: List[str] = Field(
        default_factory=list,
        description="How to implement this pattern effectively"
    )
    
    common_mistakes: List[str] = Field(
        default_factory=list,
        description="Common ways this pattern is implemented poorly"
    )


class CommonComplaint(BaseModel):
    """Common reader complaint about fiction content."""
    
    complaint_type: str = Field(
        description="Category of complaint (pacing, character, plot, etc.)"
    )
    
    complaint_description: str = Field(
        min_length=20,
        max_length=300,
        description="Description of what readers complain about"
    )
    
    frequency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How frequently this complaint appears"
    )
    
    impact_on_sales: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated negative impact on book sales/ratings"
    )
    
    affected_genres: List[str] = Field(
        default_factory=list,
        description="Genres where this complaint is most common"
    )
    
    prevention_strategies: List[str] = Field(
        default_factory=list,
        description="Strategies to avoid this complaint"
    )
    
    example_reviews: List[str] = Field(
        default_factory=list,
        description="Example reader reviews mentioning this complaint"
    )


class CompetitiveAnalysis(BaseModel):
    """Analysis of competitive landscape and market positioning."""
    
    genre_saturation_level: float = Field(
        ge=0.0,
        le=1.0,
        description="How saturated the target genre market is"
    )
    
    key_differentiators: List[str] = Field(
        default_factory=list,
        description="Elements that help books stand out in this market"
    )
    
    successful_recent_releases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Analysis of recent successful books in genre"
    )
    
    market_gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps or opportunities in the market"
    )
    
    pricing_insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market pricing analysis and recommendations"
    )
    
    distribution_channels: List[str] = Field(
        default_factory=list,
        description="Most effective distribution channels for this genre"
    )


class MarketIntelligence(BaseModel):
    """
    Comprehensive market intelligence for adaptive content generation.
    
    Aggregates trend analysis, reader preferences, success patterns,
    and competitive insights for commercial optimization.
    """
    
    genre: str = Field(
        description="Primary genre this intelligence applies to"
    )
    
    target_market: str = Field(
        description="Target market segment (mass market, literary, YA, etc.)"
    )
    
    current_trends: List[MarketTrend] = Field(
        default_factory=list,
        description="Current market trends relevant to content generation"
    )
    
    reader_preferences: List[ReaderPreference] = Field(
        default_factory=list,
        description="Reader preference data for optimization"
    )
    
    success_patterns: List[SuccessPattern] = Field(
        default_factory=list,
        description="Patterns identified in commercially successful content"
    )
    
    common_complaints: List[CommonComplaint] = Field(
        default_factory=list,
        description="Common reader complaints to avoid"
    )
    
    competitive_analysis: CompetitiveAnalysis = Field(
        description="Analysis of competitive landscape"
    )
    
    oversaturated_elements: List[str] = Field(
        default_factory=list,
        description="Story elements that are overused in current market"
    )
    
    emerging_opportunities: List[str] = Field(
        default_factory=list,
        description="Emerging opportunities for differentiation"
    )
    
    seasonal_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Seasonal trends affecting book sales/preferences"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this intelligence was last updated"
    )
    
    data_sources: List[str] = Field(
        default_factory=list,
        description="Sources used to compile this intelligence"
    )
    
    @computed_field
    @property
    def high_confidence_trends(self) -> List[MarketTrend]:
        """Get trends with high or very high confidence levels."""
        return [trend for trend in self.current_trends 
                if trend.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]]
    
    @computed_field
    @property
    def emerging_trend_count(self) -> int:
        """Number of emerging trends to watch."""
        return len([trend for trend in self.current_trends 
                   if trend.lifecycle_stage == TrendLifecycle.EMERGING])
    
    @computed_field
    @property
    def critical_complaints_count(self) -> int:
        """Number of high-impact reader complaints."""
        return len([complaint for complaint in self.common_complaints 
                   if complaint.impact_on_sales >= 0.7])
    
    def get_trends_by_type(self, trend_type: TrendType) -> List[MarketTrend]:
        """Get all trends of specific type."""
        return [trend for trend in self.current_trends 
                if trend.trend_type == trend_type]
    
    def get_preferences_by_type(self, pref_type: ReaderPreferenceType) -> List[ReaderPreference]:
        """Get reader preferences of specific type."""
        return [pref for pref in self.reader_preferences 
                if pref.preference_type == pref_type]
    
    def get_high_impact_complaints(self, impact_threshold: float = 0.6) -> List[CommonComplaint]:
        """Get complaints with high impact on sales."""
        return [complaint for complaint in self.common_complaints 
                if complaint.impact_on_sales >= impact_threshold]
    
    def calculate_market_opportunity_score(self) -> float:
        """
        Calculate overall market opportunity score.
        
        Returns:
            Opportunity score from 0.0 (saturated/difficult) to 1.0 (high opportunity)
        """
        opportunity_factors = []
        
        # Market saturation (inverted - less saturation = more opportunity)
        saturation_penalty = self.competitive_analysis.genre_saturation_level
        opportunity_factors.append(1.0 - saturation_penalty)
        
        # Emerging trends create opportunities
        emerging_bonus = min(0.3, self.emerging_trend_count * 0.1)
        opportunity_factors.append(emerging_bonus)
        
        # Market gaps represent opportunities
        gap_bonus = min(0.3, len(self.competitive_analysis.market_gaps) * 0.1)
        opportunity_factors.append(gap_bonus)
        
        # Fewer critical complaints = easier to satisfy readers
        complaint_penalty = min(0.4, self.critical_complaints_count * 0.1)
        opportunity_factors.append(1.0 - complaint_penalty)
        
        # Base score
        base_score = sum(opportunity_factors) / len(opportunity_factors)
        
        # Boost for identified opportunities
        if self.emerging_opportunities:
            base_score += min(0.2, len(self.emerging_opportunities) * 0.05)
        
        return min(1.0, base_score)
    
    def generate_content_guidance(self) -> Dict[str, List[str]]:
        """
        Generate actionable content guidance based on market intelligence.
        
        Returns:
            Dict with guidance categories and specific recommendations
        """
        guidance = {
            "trending_elements": [],
            "reader_preferences": [],
            "avoid_these": [],
            "differentiation_opportunities": []
        }
        
        # Extract trending elements
        for trend in self.high_confidence_trends:
            if trend.lifecycle_stage in [TrendLifecycle.GROWING, TrendLifecycle.PEAK]:
                guidance["trending_elements"].append(
                    f"{trend.title}: {trend.description[:100]}..."
                )
        
        # High-importance reader preferences
        high_importance_prefs = [pref for pref in self.reader_preferences 
                                if pref.importance_weight >= 0.7]
        for pref in high_importance_prefs:
            guidance["reader_preferences"].append(pref.preference_description)
        
        # Things to avoid (oversaturated + common complaints)
        guidance["avoid_these"].extend(self.oversaturated_elements)
        for complaint in self.get_high_impact_complaints():
            guidance["avoid_these"].append(f"Avoid: {complaint.complaint_description}")
        
        # Differentiation opportunities
        guidance["differentiation_opportunities"].extend(self.emerging_opportunities)
        guidance["differentiation_opportunities"].extend(
            self.competitive_analysis.market_gaps
        )
        
        return guidance
    
    def assess_trend_alignment(self, content_elements: List[str]) -> float:
        """
        Assess how well content elements align with current trends.
        
        Args:
            content_elements: List of content elements to evaluate
            
        Returns:
            Alignment score from 0.0 (poor alignment) to 1.0 (excellent alignment)
        """
        if not content_elements or not self.current_trends:
            return 0.5  # Neutral if no data
        
        alignment_scores = []
        
        for element in content_elements:
            element_lower = element.lower()
            best_alignment = 0.0
            
            for trend in self.high_confidence_trends:
                # Simple keyword matching (would be more sophisticated in practice)
                trend_keywords = (trend.title + " " + trend.description).lower()
                
                # Check if element relates to trend
                element_words = set(element_lower.split())
                trend_words = set(trend_keywords.split())
                
                overlap = len(element_words.intersection(trend_words))
                if overlap > 0:
                    # Weight by trend popularity and lifecycle
                    lifecycle_weight = {
                        TrendLifecycle.EMERGING: 0.6,
                        TrendLifecycle.GROWING: 0.9,
                        TrendLifecycle.PEAK: 1.0,
                        TrendLifecycle.DECLINING: 0.4,
                        TrendLifecycle.SATURATED: 0.2
                    }
                    
                    trend_score = (
                        trend.popularity_score * 
                        lifecycle_weight.get(trend.lifecycle_stage, 0.5) *
                        (overlap / len(element_words))
                    )
                    
                    best_alignment = max(best_alignment, trend_score)
            
            alignment_scores.append(best_alignment)
        
        return sum(alignment_scores) / len(alignment_scores)
    
    def is_stale(self, staleness_threshold_days: int = 7) -> bool:
        """
        Check if market intelligence is stale and needs updating.
        
        Args:
            staleness_threshold_days: Days after which intelligence is considered stale
            
        Returns:
            True if intelligence needs updating
        """
        days_since_update = (datetime.now() - self.last_updated).days
        return days_since_update >= staleness_threshold_days
    
    def get_update_priorities(self) -> List[str]:
        """
        Get prioritized list of intelligence areas needing updates.
        
        Returns:
            List of update priorities
        """
        priorities = []
        
        # Check trend validation dates
        stale_trends = [trend for trend in self.current_trends 
                       if (datetime.now() - trend.last_validated).days >= 14]
        if stale_trends:
            priorities.append(f"Validate {len(stale_trends)} potentially stale trends")
        
        # Check for missing recent data
        if not self.current_trends:
            priorities.append("No current trends - need comprehensive trend analysis")
        
        if not self.reader_preferences:
            priorities.append("Missing reader preference data")
        
        if not self.success_patterns:
            priorities.append("Need analysis of recent success patterns")
        
        # Check competitive analysis freshness
        if not self.competitive_analysis.successful_recent_releases:
            priorities.append("Update competitive analysis with recent releases")
        
        return priorities[:5]  # Return top 5 priorities
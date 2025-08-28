"""
Reader Engagement Assessment Model

Evaluates emotional journey effectiveness, question/answer balance, satisfaction
potential, and cliffhanger quality for the Reader Engagement Critic.
"""

from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class EmotionalResponse(str, Enum):
    """Expected reader emotional responses."""
    CURIOSITY = "curiosity"
    TENSION = "tension"
    SATISFACTION = "satisfaction"
    SURPRISE = "surprise"
    EMPATHY = "empathy"
    ANTICIPATION = "anticipation"
    RELIEF = "relief"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    HOPE = "hope"
    CONFUSION = "confusion"
    BOREDOM = "boredom"


class EngagementRisk(str, Enum):
    """Types of reader engagement risks."""
    PACING_TOO_SLOW = "pacing_too_slow"
    INFORMATION_OVERLOAD = "information_overload"
    LACK_OF_STAKES = "lack_of_stakes"
    PREDICTABILITY = "predictability"
    CHARACTER_DISCONNECT = "character_disconnect"
    UNRESOLVED_THREADS = "unresolved_threads"
    WEAK_CLIFFHANGER = "weak_cliffhanger"
    REPETITIVE_CONTENT = "repetitive_content"


class CliffhangerType(str, Enum):
    """Types of chapter endings."""
    REVELATION = "revelation"
    DANGER = "danger"
    DECISION = "decision"
    ARRIVAL = "arrival"
    DISCOVERY = "discovery"
    CONFRONTATION = "confrontation"
    EMOTIONAL_PEAK = "emotional_peak"
    NONE = "none"


class EmotionalBeat(BaseModel):
    """Analysis of a specific emotional beat in the chapter."""
    
    beat_type: EmotionalResponse = Field(
        description="Type of emotional response targeted"
    )
    
    effectiveness: float = Field(
        ge=0.0,
        le=1.0,
        description="How effectively this emotion was evoked"
    )
    
    placement_appropriateness: float = Field(
        ge=0.0,
        le=1.0,
        description="How appropriate the placement of this beat is"
    )
    
    supporting_elements: List[str] = Field(
        default_factory=list,
        description="Story elements that support this emotional beat"
    )
    
    evidence_text: Optional[str] = Field(
        default=None,
        description="Specific text that creates this emotional response"
    )


class QuestionAnswerAnalysis(BaseModel):
    """Analysis of question planting and resolution balance."""
    
    new_questions_planted: int = Field(
        ge=0,
        description="Number of new questions/mysteries introduced"
    )
    
    existing_questions_answered: int = Field(
        ge=0,
        description="Number of existing questions resolved"
    )
    
    question_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of questions planted (intrigue level)"
    )
    
    answer_satisfaction_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Satisfaction level of answers provided"
    )
    
    balance_appropriateness: float = Field(
        ge=0.0,
        le=1.0,
        description="Appropriateness of question/answer balance for story position"
    )
    
    outstanding_question_pressure: float = Field(
        ge=0.0,
        le=1.0,
        description="Pressure from unresolved questions (reader urgency)"
    )


class SatisfactionPrediction(BaseModel):
    """Prediction of reader satisfaction with chapter."""
    
    immediate_satisfaction: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted immediate reader satisfaction"
    )
    
    long_term_satisfaction: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted satisfaction as part of overall story"
    )
    
    reread_value: float = Field(
        ge=0.0,
        le=1.0,
        description="Likelihood reader would reread this chapter"
    )
    
    recommendation_likelihood: float = Field(
        ge=0.0,
        le=1.0,
        description="Likelihood reader would recommend this chapter/book"
    )
    
    satisfaction_factors: List[str] = Field(
        default_factory=list,
        description="Elements contributing to reader satisfaction"
    )
    
    dissatisfaction_risks: List[str] = Field(
        default_factory=list,
        description="Elements that might reduce satisfaction"
    )


class CliffhangerAnalysis(BaseModel):
    """Analysis of chapter ending and cliffhanger effectiveness."""
    
    cliffhanger_type: CliffhangerType = Field(
        description="Type of chapter ending employed"
    )
    
    effectiveness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How effective the cliffhanger is at creating anticipation"
    )
    
    stakes_clarity: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear the stakes are for the cliffhanger"
    )
    
    emotional_impact: float = Field(
        ge=0.0,
        le=1.0,
        description="Emotional impact of the chapter ending"
    )
    
    next_chapter_compulsion: float = Field(
        ge=0.0,
        le=1.0,
        description="How much this ending compels reading the next chapter"
    )
    
    resolution_timeline: str = Field(
        description="Expected timeline for cliffhanger resolution (immediate, short-term, long-term)"
    )


class ReaderEngagementAssessment(BaseModel):
    """
    Comprehensive assessment of reader engagement and satisfaction potential.
    
    Evaluates emotional journey, question/answer balance, satisfaction prediction,
    and cliffhanger effectiveness for commercial viability.
    """
    
    chapter_number: int = Field(
        ge=1,
        description="Chapter being assessed"
    )
    
    emotional_journey_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Effectiveness of emotional journey throughout chapter"
    )
    
    question_answer_balance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Balance of planting questions vs providing answers"
    )
    
    satisfaction_potential_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted reader satisfaction with this chapter"
    )
    
    cliffhanger_effectiveness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Effectiveness of chapter ending at maintaining engagement"
    )
    
    emotional_beats: List[EmotionalBeat] = Field(
        default_factory=list,
        description="Analysis of emotional beats throughout chapter"
    )
    
    question_answer_analysis: QuestionAnswerAnalysis = Field(
        description="Detailed analysis of question/answer balance"
    )
    
    satisfaction_prediction: SatisfactionPrediction = Field(
        description="Prediction of reader satisfaction levels"
    )
    
    cliffhanger_analysis: CliffhangerAnalysis = Field(
        description="Analysis of chapter ending effectiveness"
    )
    
    engagement_risks: List[EngagementRisk] = Field(
        default_factory=list,
        description="Identified risks to reader engagement"
    )
    
    engagement_strengths: List[str] = Field(
        default_factory=list,
        description="Elements that strongly engage readers"
    )
    
    predicted_reader_response: str = Field(
        min_length=50,
        max_length=500,
        description="Predicted overall reader response to this chapter"
    )
    
    engagement_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific recommendations to improve reader engagement"
    )
    
    @computed_field
    @property
    def overall_engagement_score(self) -> float:
        """Calculate weighted overall reader engagement score."""
        return (
            self.emotional_journey_score * 0.3 +
            self.question_answer_balance_score * 0.25 +
            self.satisfaction_potential_score * 0.3 +
            self.cliffhanger_effectiveness_score * 0.15
        )
    
    @computed_field
    @property
    def high_risk_engagement_issues(self) -> int:
        """Count of high-risk engagement problems."""
        high_risk_types = [
            EngagementRisk.LACK_OF_STAKES,
            EngagementRisk.CHARACTER_DISCONNECT,
            EngagementRisk.PACING_TOO_SLOW
        ]
        return len([risk for risk in self.engagement_risks if risk in high_risk_types])
    
    @computed_field
    @property
    def emotional_variety_score(self) -> float:
        """Score based on variety of emotional beats."""
        unique_emotions = len(set(beat.beat_type for beat in self.emotional_beats))
        # Normalize to 0-1 scale (assume 6+ different emotions = max variety)
        return min(1.0, unique_emotions / 6.0)
    
    def calculate_page_turner_potential(self) -> float:
        """
        Calculate how likely this chapter is to keep readers turning pages.
        
        Returns:
            Page-turner score from 0.0 (puts reader to sleep) to 1.0 (unputdownable)
        """
        factors = []
        
        # Strong cliffhanger is crucial for page-turning
        factors.append(self.cliffhanger_effectiveness_score * 0.4)
        
        # Emotional engagement keeps readers invested
        factors.append(self.emotional_journey_score * 0.3)
        
        # Question planting creates curiosity
        question_factor = min(1.0, self.question_answer_analysis.new_questions_planted / 3.0)
        factors.append(question_factor * 0.2)
        
        # Avoid major engagement risks
        risk_penalty = min(1.0, len(self.engagement_risks) / 5.0)
        factors.append(max(0.0, 1.0 - risk_penalty) * 0.1)
        
        return sum(factors)
    
    def assess_binge_readability(self) -> Tuple[float, List[str]]:
        """
        Assess how well chapter supports binge-reading behavior.
        
        Returns:
            Tuple of (binge_score, improvement_suggestions)
        """
        binge_score = 0.0
        suggestions = []
        
        # Fast pacing supports binge reading
        if EngagementRisk.PACING_TOO_SLOW in self.engagement_risks:
            suggestions.append("Increase pacing to support binge reading")
        else:
            binge_score += 0.25
        
        # Strong cliffhangers encourage immediate next chapter
        if self.cliffhanger_analysis.next_chapter_compulsion >= 0.7:
            binge_score += 0.3
        else:
            suggestions.append("Strengthen chapter ending to compel next chapter reading")
        
        # Emotional investment keeps readers engaged
        if self.emotional_journey_score >= 0.7:
            binge_score += 0.25
        else:
            suggestions.append("Deepen emotional investment to maintain reader engagement")
        
        # Avoid information overload that slows reading
        if EngagementRisk.INFORMATION_OVERLOAD in self.engagement_risks:
            suggestions.append("Reduce information density for easier consumption")
        else:
            binge_score += 0.2
        
        return (binge_score, suggestions)
    
    def predict_review_sentiment(self) -> Dict[str, float]:
        """
        Predict likely reader review sentiment distribution.
        
        Returns:
            Dict with predicted percentages for positive/neutral/negative reviews
        """
        # Base predictions on satisfaction scores
        immediate_sat = self.satisfaction_prediction.immediate_satisfaction
        long_term_sat = self.satisfaction_prediction.long_term_satisfaction
        
        # Calculate overall satisfaction
        overall_sat = (immediate_sat + long_term_sat) / 2
        
        # Adjust for engagement risks
        risk_penalty = min(0.3, len(self.engagement_risks) * 0.05)
        adjusted_sat = max(0.0, overall_sat - risk_penalty)
        
        # Predict sentiment distribution
        if adjusted_sat >= 0.8:
            return {"positive": 0.75, "neutral": 0.20, "negative": 0.05}
        elif adjusted_sat >= 0.6:
            return {"positive": 0.60, "neutral": 0.30, "negative": 0.10}
        elif adjusted_sat >= 0.4:
            return {"positive": 0.40, "neutral": 0.35, "negative": 0.25}
        else:
            return {"positive": 0.20, "neutral": 0.30, "negative": 0.50}
    
    def get_genre_specific_engagement_feedback(self, genre: str) -> List[str]:
        """
        Provide genre-specific engagement recommendations.
        
        Args:
            genre: Target genre
            
        Returns:
            List of genre-tailored engagement suggestions
        """
        feedback = []
        genre_lower = genre.lower()
        
        if "romance" in genre_lower:
            # Romance needs strong emotional beats
            romance_emotions = [beat for beat in self.emotional_beats 
                              if beat.beat_type in [EmotionalResponse.EMPATHY, EmotionalResponse.ANTICIPATION, EmotionalResponse.SATISFACTION]]
            if len(romance_emotions) < 2:
                feedback.append("Romance readers expect strong emotional connection and romantic tension")
            
            if self.cliffhanger_analysis.cliffhanger_type not in [CliffhangerType.EMOTIONAL_PEAK, CliffhangerType.DECISION]:
                feedback.append("Romance chapters benefit from emotional or relationship-focused cliffhangers")
        
        elif "mystery" in genre_lower or "thriller" in genre_lower:
            # Mystery/thriller needs questions and tension
            if self.question_answer_analysis.new_questions_planted == 0:
                feedback.append("Mystery/thriller chapters should introduce new questions or deepen existing mysteries")
            
            tension_beats = [beat for beat in self.emotional_beats 
                           if beat.beat_type in [EmotionalResponse.TENSION, EmotionalResponse.FEAR, EmotionalResponse.SURPRISE]]
            if len(tension_beats) == 0:
                feedback.append("Mystery/thriller requires sustained tension and suspenseful moments")
        
        elif "fantasy" in genre_lower or "sci-fi" in genre_lower:
            # SFF needs wonder and discovery
            wonder_beats = [beat for beat in self.emotional_beats 
                          if beat.beat_type in [EmotionalResponse.CURIOSITY, EmotionalResponse.SURPRISE]]
            if len(wonder_beats) == 0:
                feedback.append("SFF readers expect moments of wonder, discovery, or world-building revelation")
        
        elif "literary" in genre_lower:
            # Literary fiction needs emotional depth
            if self.emotional_variety_score < 0.6:
                feedback.append("Literary fiction benefits from complex, varied emotional experiences")
            
            if self.satisfaction_prediction.reread_value < 0.6:
                feedback.append("Literary fiction should have depth that rewards rereading")
        
        return feedback
    
    def calculate_commercial_appeal(self, target_audience: str = "general") -> float:
        """
        Calculate commercial appeal score based on engagement factors.
        
        Args:
            target_audience: Target reader demographic
            
        Returns:
            Commercial appeal score (0.0-1.0)
        """
        appeal_factors = []
        
        # Page-turner potential is crucial for commercial success
        page_turner = self.calculate_page_turner_potential()
        appeal_factors.append(page_turner * 0.4)
        
        # Broad emotional appeal
        appeal_factors.append(self.emotional_journey_score * 0.25)
        
        # Satisfaction drives word-of-mouth
        appeal_factors.append(self.satisfaction_potential_score * 0.25)
        
        # Question/answer balance maintains interest
        appeal_factors.append(self.question_answer_balance_score * 0.1)
        
        base_appeal = sum(appeal_factors)
        
        # Adjust for audience
        if target_audience == "young_adult":
            # YA benefits from strong emotional beats and cliffhangers
            if self.cliffhanger_effectiveness_score >= 0.7:
                base_appeal += 0.05
            if self.emotional_variety_score >= 0.7:
                base_appeal += 0.05
        
        elif target_audience == "mass_market":
            # Mass market needs accessibility and pace
            if EngagementRisk.PACING_TOO_SLOW not in self.engagement_risks:
                base_appeal += 0.05
            if EngagementRisk.INFORMATION_OVERLOAD not in self.engagement_risks:
                base_appeal += 0.05
        
        return min(1.0, base_appeal)
    
    def is_engaging_enough_for_publication(self, genre: str = "general", threshold: float = 0.65) -> bool:
        """
        Determine if engagement level meets publication standards.
        
        Args:
            genre: Target genre
            threshold: Minimum engagement score
            
        Returns:
            True if engagement meets standards
        """
        # Must meet overall threshold
        if self.overall_engagement_score < threshold:
            return False
        
        # Cannot have too many high-risk engagement issues
        if self.high_risk_engagement_issues > 1:
            return False
        
        # Must have some emotional variety
        if len(self.emotional_beats) == 0:
            return False
        
        # Genre-specific requirements
        genre_lower = genre.lower()
        if "thriller" in genre_lower and self.cliffhanger_effectiveness_score < 0.6:
            return False
        
        if "romance" in genre_lower and self.emotional_journey_score < 0.7:
            return False
        
        return True
"""
Recursive Detection Aggregator for G₂-Level Reasoning

This module implements the recursive detection decision engine that combines evidence
from multiple feature modules using weighted aggregation and recursive refinement.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
try:
    from .modules.base_module import FeatureResult
except ImportError:
    from modules.base_module import FeatureResult


@dataclass
class EvidenceSignal:
    """Single piece of evidence from a feature module with polarity support"""
    name: str
    score: float
    weight: float
    polarity: str = "positive"  # "positive" or "negative"
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True


@dataclass
class AggregationResult:
    """Result of evidence aggregation with positive/negative evidence breakdown"""
    final_score: float
    confidence: float
    phi0_contribution: float
    feature_contribution: float
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    positive_evidence_score: float = 0.0
    negative_evidence_score: float = 0.0
    evidence_count: int = 0  # Total evidence count (backward compatibility)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingAggregationResult(AggregationResult):
    """Extended result for streaming aggregation with additional metadata"""
    modules_completed: int = 0
    modules_total: int = 0
    completion_percentage: float = 0.0
    streaming_confidence: float = 0.0  # Confidence adjusted for partial results
    early_decision_possible: bool = False
    next_module_priority: Optional[str] = None


class RecursiveDetectionAggregator:
    """
    G₂-level recursive geometric reasoning engine.
    
    Combines φ⁰ base score with evidence from multiple feature modules
    to make final detection decisions with recursive refinement capability.
    """
    
    def __init__(self, phi0_weight: float = 0.6, feature_weight: float = 0.4, polarity_preferences: Dict[str, str] = None):
        """
        Initialize the aggregator
        
        Args:
            phi0_weight: Weight for base φ⁰ score contribution
            feature_weight: Weight for feature evidence contribution
            polarity_preferences: Dict mapping feature names to polarity preferences ("positive", "negative", or None)
        """
        self.phi0_weight = phi0_weight
        self.feature_weight = feature_weight
        self.polarity_preferences = polarity_preferences or {}
        self.evidence_signals: List[EvidenceSignal] = []
        self.phi0_score = 0.0
        self.refinement_history: List[AggregationResult] = []
    
    def reset(self):
        """Reset aggregator for new detection"""
        self.evidence_signals.clear()
        self.phi0_score = 0.0
        self.refinement_history.clear()
    
    def set_phi0_score(self, score: float):
        """Set the base φ⁰ detection score"""
        self.phi0_score = max(0.0, min(1.0, score))
    
    def add_evidence(self, name: str, result: FeatureResult, weight: float = 1.0):
        """
        Add evidence signal from a feature module with dynamic polarity interpretation
        
        Args:
            name: Name of the feature module
            result: FeatureResult from the module
            weight: Weight for this evidence in aggregation
        """
        if result.valid:
            # Dynamic polarity interpretation for neutral features
            interpreted_polarity, adjusted_weight = self._interpret_polarity(
                name, result.score, result.metadata or {}, weight
            )
            
            signal = EvidenceSignal(
                name=name,
                score=max(0.0, min(1.0, result.score)),
                weight=adjusted_weight,
                polarity=interpreted_polarity,
                metadata=result.metadata or {},
                valid=result.valid
            )
            self.evidence_signals.append(signal)
    
    def aggregate(self) -> AggregationResult:
        """
        Aggregate all evidence signals with φ⁰ score using bidirectional reasoning
        
        Returns:
            AggregationResult with final detection decision incorporating negative evidence
        """
        if not self.evidence_signals:
            # Only φ⁰ score available
            return AggregationResult(
                final_score=self.phi0_score,
                confidence=self.phi0_score,
                phi0_contribution=1.0,
                feature_contribution=0.0,
                positive_evidence_count=0,
                negative_evidence_count=0,
                evidence_count=0,
                reason="Only φ⁰ score available",
                metadata={"phi0_score": self.phi0_score}
            )
        
        # Separate positive and negative evidence
        valid_signals = [s for s in self.evidence_signals if s.valid]
        positive_signals = [s for s in valid_signals if s.polarity == "positive"]
        negative_signals = [s for s in valid_signals if s.polarity == "negative"]
        
        if not valid_signals:
            return AggregationResult(
                final_score=self.phi0_score,
                confidence=self.phi0_score * 0.8,  # Reduced confidence due to failed features
                phi0_contribution=1.0,
                feature_contribution=0.0,
                positive_evidence_count=0,
                negative_evidence_count=0,
                evidence_count=0,
                reason="All feature modules failed",
                metadata={"phi0_score": self.phi0_score}
            )
        
        # Calculate weighted positive evidence
        pos_score = 0.0
        pos_weight_sum = 0.0
        if positive_signals:
            pos_weight_sum = sum(s.weight for s in positive_signals)
            pos_score = sum(s.score * s.weight for s in positive_signals) / pos_weight_sum
        
        # Calculate weighted negative evidence
        neg_score = 0.0
        neg_weight_sum = 0.0
        if negative_signals:
            neg_weight_sum = sum(s.weight for s in negative_signals)
            neg_score = sum(s.score * s.weight for s in negative_signals) / neg_weight_sum
        
        # Bidirectional feature combination: positive evidence builds confidence, negative evidence reduces it
        feature_score = 0.5 + (pos_score - neg_score) * 0.5
        feature_score = max(0.0, min(1.0, feature_score))
        
        # Combine φ⁰ and bidirectional feature scores
        final_score = (self.phi0_weight * self.phi0_score + 
                      self.feature_weight * feature_score)
        
        # Enhanced confidence calculation incorporating evidence quality
        phi0_feature_agreement = 1.0 - abs(self.phi0_score - feature_score)
        evidence_quality = (pos_weight_sum + neg_weight_sum) / (pos_weight_sum + neg_weight_sum + 2.0)  # Normalized quality
        confidence = final_score * (0.6 + 0.3 * phi0_feature_agreement + 0.1 * evidence_quality)
        
        # Generate detailed reason with bidirectional evidence breakdown
        pos_summary = ", ".join([f"{s.name}={s.score:.3f}" for s in positive_signals[:2]])
        neg_summary = ", ".join([f"{s.name}={s.score:.3f}" for s in negative_signals[:2]])
        
        reason_parts = [f"φ⁰={self.phi0_score:.3f}"]
        if pos_summary:
            reason_parts.append(f"✓ Positive: {pos_summary}")
        if neg_summary:
            reason_parts.append(f"✗ Negative: {neg_summary}")
        reason = ", ".join(reason_parts)
        
        # Collect comprehensive metadata
        metadata = {
            "phi0_score": self.phi0_score,
            "feature_score": feature_score,
            "positive_score": pos_score,
            "negative_score": neg_score,
            "agreement": phi0_feature_agreement,
            "evidence_quality": evidence_quality,
            "positive_details": {s.name: {"score": s.score, "weight": s.weight, "metadata": s.metadata} 
                               for s in positive_signals},
            "negative_details": {s.name: {"score": s.score, "weight": s.weight, "metadata": s.metadata} 
                               for s in negative_signals}
        }
        
        result = AggregationResult(
            final_score=final_score,
            confidence=confidence,
            phi0_contribution=self.phi0_weight,
            feature_contribution=self.feature_weight,
            positive_evidence_count=len(positive_signals),
            negative_evidence_count=len(negative_signals),
            positive_evidence_score=pos_score,
            negative_evidence_score=neg_score,
            evidence_count=len(valid_signals),
            reason=reason,
            metadata=metadata
        )
        
        self.refinement_history.append(result)
        return result
    
    def should_refine(self, result: AggregationResult, ambiguity_threshold: Tuple[float, float] = (0.45, 0.65)) -> bool:
        """
        Determine if recursive refinement is needed
        
        Args:
            result: Current aggregation result
            ambiguity_threshold: (low, high) range for ambiguous scores
            
        Returns:
            True if refinement should be attempted
        """
        low_thresh, high_thresh = ambiguity_threshold
        
        # Check if score is in ambiguous range
        if low_thresh <= result.final_score <= high_thresh:
            return True
        
        # Check if there's disagreement between φ⁰ and features
        if result.evidence_count > 0:
            disagreement = abs(self.phi0_score - result.metadata.get("feature_score", 0))
            if disagreement > 0.3:
                return True
        
        # Check if confidence is low
        if result.confidence < 0.6:
            return True
        
        return False
    
    def suggest_refinement_strategy(self, result: AggregationResult) -> Dict[str, Any]:
        """
        Suggest strategy for recursive refinement
        
        Args:
            result: Current aggregation result
            
        Returns:
            Dictionary with refinement suggestions
        """
        suggestions = {
            "increase_radius": False,
            "decrease_radius": False,
            "zoom_in": False,
            "different_center": False,
            "reasons": []
        }
        
        # Analyze disagreement patterns
        if result.evidence_count > 0:
            feature_score = result.metadata.get("feature_score", 0)
            
            if self.phi0_score > feature_score + 0.2:
                # φ⁰ confident but features disagree - might be too small radius
                suggestions["increase_radius"] = True
                suggestions["reasons"].append("φ⁰ confident but features uncertain - try larger radius")
            
            elif feature_score > self.phi0_score + 0.2:
                # Features confident but φ⁰ disagrees - might be wrong center or too large radius
                suggestions["different_center"] = True
                suggestions["decrease_radius"] = True
                suggestions["reasons"].append("Features confident but φ⁰ uncertain - try different center/smaller radius")
        
        # Low confidence suggests need for higher resolution
        if result.confidence < 0.5:
            suggestions["zoom_in"] = True
            suggestions["reasons"].append("Low confidence - try higher resolution analysis")
        
        return suggestions
    
    def get_refinement_history(self) -> List[AggregationResult]:
        """Get history of refinement attempts"""
        return self.refinement_history.copy()
    
    def _interpret_polarity(self, module_name: str, score: float, metadata: Dict[str, Any], weight: float) -> Tuple[str, float]:
        """
        Dynamically interpret polarity of neutral evidence based on context
        
        Args:
            module_name: Name of the feature module
            score: Raw feature score
            metadata: Feature metadata for context
            weight: Original weight
            
        Returns:
            Tuple of (interpreted_polarity, adjusted_weight)
        """
        # Check for explicit polarity preference from profile configuration
        if module_name in self.polarity_preferences:
            preferred_polarity = self.polarity_preferences[module_name]
            if preferred_polarity in ["positive", "negative"]:
                # Use configured polarity preference, but still adjust weight based on score strength
                score_strength = max(0.1, score)  # Avoid zero weight
                adjusted_weight = weight * score_strength
                return preferred_polarity, adjusted_weight
        
        # Very low scores are typically negative evidence
        if score < 0.1:
            return "negative", weight * 0.8
            
        # Context-aware polarity interpretation based on kernel.md specifications
        if module_name == "ElevationEntropy" or module_name == "EntropyAnalysis":
            entropy = metadata.get('combined_entropy', score)
            if entropy > 0.7:  # High entropy = chaos = negative evidence
                return "negative", weight
            elif entropy < 0.3:  # Low entropy = order = positive evidence  
                return "positive", weight * 0.8
            else:  # Medium entropy = inconclusive, slightly negative
                return "negative", weight * 0.5
                
        elif module_name == "Volume" or module_name == "VolumeAnalysis":
            volume = metadata.get('normalized_volume', score)
            if volume > 0.9:  # Excessive volume = likely natural feature
                return "negative", weight * 1.2
            elif volume > 0.4:  # Meaningful volume = structure (more discriminating)
                return "positive", weight
            else:  # Insufficient volume = negative
                return "negative", weight * 0.9
                
        elif module_name == "DropoffSharpness" or module_name == "EdgeAnalysis":
            sharpness = metadata.get('edge_sharpness', score)
            max_edge = metadata.get('max_edge_strength', 0)
            
            # Sharp edges only meaningful if there's actual elevation change
            if sharpness > 0.5 and max_edge > 0.3:  # Sharp edges with meaningful elevation
                return "positive", weight
            elif sharpness < 0.2:  # Very gradual edges
                return "negative", weight
            elif max_edge < 0.1:  # No meaningful edges (flat terrain)
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.4  # Medium sharpness = weak positive
                
        elif module_name == "Compactness":
            compactness = metadata.get('compactness', score)
            elevation_range = metadata.get('value_range', 0)
            mean_elevation = metadata.get('mean_elevation', 0)
            
            # High compactness only meaningful if there's significant elevation variation
            if compactness > 0.8 and elevation_range > 0.5 and mean_elevation > 0.3:
                return "positive", weight
            elif compactness > 0.7 and elevation_range < 0.2:  # High compactness but flat = not structural
                return "negative", weight * 0.8
            elif compactness < 0.4:  # Low compactness = chaos
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.5  # Medium compactness = uncertain
                
        elif module_name == "ElevationHistogram":
            phi0_sig = metadata.get('phi0_signature', score)
            if phi0_sig > 0.6:  # Strong φ⁰ signature
                return "positive", weight * 1.2  # Boost core φ⁰ evidence
            elif phi0_sig < 0.3:  # Weak φ⁰ signature
                return "negative", weight
            else:
                return "positive", weight
        
        # Default: treat high scores as positive, low as negative (more discriminating)
        if score > 0.7:
            return "positive", weight
        elif score < 0.3:
            return "negative", weight * 0.9
        else:
            return "positive", weight * 0.5  # Less confident for medium scores
    
    def get_best_result(self) -> Optional[AggregationResult]:
        """Get the best result from refinement history"""
        if not self.refinement_history:
            return None
    
    def _interpret_polarity(self, module_name: str, score: float, metadata: Dict[str, Any], weight: float) -> Tuple[str, float]:
        """
        Dynamically interpret polarity of neutral evidence based on context
        
        Args:
            module_name: Name of the feature module
            score: Raw feature score
            metadata: Feature metadata for context
            weight: Original weight
            
        Returns:
            Tuple of (interpreted_polarity, adjusted_weight)
        """
        # Very low scores are typically negative evidence
        if score < 0.1:
            return "negative", weight * 0.8
            
        # Context-aware polarity interpretation based on kernel.md specifications
        if module_name == "ElevationEntropy" or module_name == "EntropyAnalysis":
            entropy = metadata.get('combined_entropy', score)
            if entropy > 0.7:  # High entropy = chaos = negative evidence
                return "negative", weight
            elif entropy < 0.3:  # Low entropy = order = positive evidence  
                return "positive", weight * 0.8
            else:  # Medium entropy = inconclusive, slightly negative
                return "negative", weight * 0.5
                
        elif module_name == "Volume" or module_name == "VolumeAnalysis":
            volume = metadata.get('normalized_volume', score)
            if volume > 0.9:  # Excessive volume = likely natural feature
                return "negative", weight * 1.2
            elif volume > 0.4:  # Meaningful volume = structure (more discriminating)
                return "positive", weight
            else:  # Insufficient volume = negative
                return "negative", weight * 0.9
                
        elif module_name == "DropoffSharpness" or module_name == "EdgeAnalysis":
            sharpness = metadata.get('edge_sharpness', score)
            max_edge = metadata.get('max_edge_strength', 0)
            
            # Sharp edges only meaningful if there's actual elevation change
            if sharpness > 0.5 and max_edge > 0.3:  # Sharp edges with meaningful elevation
                return "positive", weight
            elif sharpness < 0.2:  # Very gradual edges
                return "negative", weight
            elif max_edge < 0.1:  # No meaningful edges (flat terrain)
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.4  # Medium sharpness = weak positive
                
        elif module_name == "Compactness":
            compactness = metadata.get('compactness', score)
            elevation_range = metadata.get('value_range', 0)
            mean_elevation = metadata.get('mean_elevation', 0)
            
            # High compactness only meaningful if there's significant elevation variation
            if compactness > 0.8 and elevation_range > 0.5 and mean_elevation > 0.3:
                return "positive", weight
            elif compactness > 0.7 and elevation_range < 0.2:  # High compactness but flat = not structural
                return "negative", weight * 0.8
            elif compactness < 0.4:  # Low compactness = chaos
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.5  # Medium compactness = uncertain
                
        elif module_name == "ElevationHistogram":
            phi0_sig = metadata.get('phi0_signature', score)
            if phi0_sig > 0.6:  # Strong φ⁰ signature
                return "positive", weight * 1.2  # Boost core φ⁰ evidence
            elif phi0_sig < 0.3:  # Weak φ⁰ signature
                return "negative", weight
            else:
                return "positive", weight
        
        # Default: treat high scores as positive, low as negative (more discriminating)
        if score > 0.7:
            return "positive", weight
        elif score < 0.3:
            return "negative", weight * 0.9
        else:
            return "positive", weight * 0.5  # Less confident for medium scores
        
        return max(self.refinement_history, key=lambda r: r.confidence)
    
    def stream_evidence(self, name: str, result: FeatureResult, weight: float = 1.0) -> StreamingAggregationResult:
        """
        Add evidence and immediately compute streaming aggregation result
        
        Args:
            name: Name of the feature module
            result: FeatureResult from the module
            weight: Weight for this evidence
            
        Returns:
            StreamingAggregationResult with current state
        """
        # Add the evidence
        self.add_evidence(name, result, weight)
        
        # Compute current aggregation state
        current_result = self.aggregate()
        
        # Calculate streaming-specific metrics
        modules_completed = len([s for s in self.evidence_signals if s.valid])
        
        # Determine if we should continue or can make early decision
        should_continue = self._should_continue_streaming(current_result)
        next_priority = self._suggest_next_priority_module() if should_continue else None
        
        # Calculate confidence evolution
        confidence_trend = self._calculate_confidence_trend()
        
        # Create comprehensive streaming result
        return StreamingAggregationResult(
            final_score=current_result.final_score,
            confidence=current_result.confidence,
            phi0_contribution=current_result.phi0_contribution,
            feature_contribution=current_result.feature_contribution,
            evidence_count=current_result.evidence_count,
            reason=current_result.reason,
            metadata=current_result.metadata,
            modules_completed=modules_completed,
            modules_total=0,  # Will be set by caller
            completion_percentage=0.0,  # Will be calculated by caller
            streaming_confidence=current_result.confidence,
            early_decision_possible=not should_continue,
            next_module_priority=next_priority
        )
    
    def _should_continue_streaming(self, current_result: AggregationResult) -> bool:
        """
        Determine if streaming should continue or if early decision is possible
        
        Args:
            current_result: Current aggregation result
            
        Returns:
            True if should continue, False if early decision possible
        """
        # Early stopping conditions
        
        # 1. High confidence - can stop early
        if current_result.confidence > 0.85:
            return False
        
        # 2. Very low confidence with multiple modules - unlikely to improve
        if current_result.confidence < 0.15 and len(self.evidence_signals) >= 3:
            return False
        
        # 3. Strong agreement between φ⁰ and features
        if len(self.evidence_signals) >= 2:
            feature_score = current_result.metadata.get("feature_score", 0)
            agreement = 1.0 - abs(self.phi0_score - feature_score)
            if agreement > 0.9 and current_result.confidence > 0.7:
                return False
        
        # 4. Clear structural pattern detected
        if current_result.final_score > 0.8 and current_result.confidence > 0.75:
            return False
        
        # 5. Clear non-structure pattern
        if current_result.final_score < 0.2 and current_result.confidence > 0.6:
            return False
        
        return True
    
    def _suggest_next_priority_module(self) -> Optional[str]:
        """
        Suggest which module should run next based on current evidence
        
        Returns:
            Name of next priority module or None if no preference
        """
        completed_names = {s.name for s in self.evidence_signals}
        
        # Priority logic based on current evidence
        if not completed_names:
            return "volume"  # Start with volume as it's often decisive
        
        # If φ⁰ score is high but we need confirmation
        if self.phi0_score > 0.6:
            if "compactness" not in completed_names:
                return "compactness"
            if "dropoff" not in completed_names:
                return "dropoff"
        
        # If φ⁰ score is low, check for vegetation discrimination
        if self.phi0_score < 0.3:
            if "entropy" not in completed_names:
                return "entropy"
        
        # Default priority order for remaining modules
        priority_order = ["volume", "compactness", "dropoff", "entropy", "planarity"]
        for module_name in priority_order:
            if module_name not in completed_names:
                return module_name
        
        return None
    
    def _interpret_polarity(self, module_name: str, score: float, metadata: Dict[str, Any], weight: float) -> Tuple[str, float]:
        """
        Dynamically interpret polarity of neutral evidence based on context
        
        Args:
            module_name: Name of the feature module
            score: Raw feature score
            metadata: Feature metadata for context
            weight: Original weight
            
        Returns:
            Tuple of (interpreted_polarity, adjusted_weight)
        """
        # Very low scores are typically negative evidence
        if score < 0.1:
            return "negative", weight * 0.8
            
        # Context-aware polarity interpretation based on kernel.md specifications
        if module_name == "ElevationEntropy" or module_name == "EntropyAnalysis":
            entropy = metadata.get('combined_entropy', score)
            if entropy > 0.7:  # High entropy = chaos = negative evidence
                return "negative", weight
            elif entropy < 0.3:  # Low entropy = order = positive evidence  
                return "positive", weight * 0.8
            else:  # Medium entropy = inconclusive, slightly negative
                return "negative", weight * 0.5
                
        elif module_name == "Volume" or module_name == "VolumeAnalysis":
            volume = metadata.get('normalized_volume', score)
            if volume > 0.9:  # Excessive volume = likely natural feature
                return "negative", weight * 1.2
            elif volume > 0.4:  # Meaningful volume = structure (more discriminating)
                return "positive", weight
            else:  # Insufficient volume = negative
                return "negative", weight * 0.9
                
        elif module_name == "DropoffSharpness" or module_name == "EdgeAnalysis":
            sharpness = metadata.get('edge_sharpness', score)
            max_edge = metadata.get('max_edge_strength', 0)
            
            # Sharp edges only meaningful if there's actual elevation change
            if sharpness > 0.5 and max_edge > 0.3:  # Sharp edges with meaningful elevation
                return "positive", weight
            elif sharpness < 0.2:  # Very gradual edges
                return "negative", weight
            elif max_edge < 0.1:  # No meaningful edges (flat terrain)
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.4  # Medium sharpness = weak positive
                
        elif module_name == "Compactness":
            compactness = metadata.get('compactness', score)
            elevation_range = metadata.get('value_range', 0)
            mean_elevation = metadata.get('mean_elevation', 0)
            
            # High compactness only meaningful if there's significant elevation variation
            if compactness > 0.8 and elevation_range > 0.5 and mean_elevation > 0.3:
                return "positive", weight
            elif compactness > 0.7 and elevation_range < 0.2:  # High compactness but flat = not structural
                return "negative", weight * 0.8
            elif compactness < 0.4:  # Low compactness = chaos
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.5  # Medium compactness = uncertain
                
        elif module_name == "ElevationHistogram":
            phi0_sig = metadata.get('phi0_signature', score)
            if phi0_sig > 0.6:  # Strong φ⁰ signature
                return "positive", weight * 1.2  # Boost core φ⁰ evidence
            elif phi0_sig < 0.3:  # Weak φ⁰ signature
                return "negative", weight
            else:
                return "positive", weight
        
        # Default: treat high scores as positive, low as negative (more discriminating)
        if score > 0.7:
            return "positive", weight
        elif score < 0.3:
            return "negative", weight * 0.9
        else:
            return "positive", weight * 0.5  # Less confident for medium scores
    
    def _calculate_confidence_trend(self) -> str:
        """
        Calculate confidence trend over the streaming process
        
        Returns:
            String describing confidence trend
        """
        if len(self.refinement_history) < 2:
            return "initial"
        
        recent_confidences = [r.confidence for r in self.refinement_history[-3:]]
        
        if len(recent_confidences) >= 2:
            if recent_confidences[-1] > recent_confidences[-2] + 0.05:
                return "increasing"
            elif recent_confidences[-1] < recent_confidences[-2] - 0.05:
                return "decreasing"
        
        return "stable"
    
    def get_streaming_summary(self) -> Dict[str, Any]:
        """Get summary of streaming aggregation process"""
        return {
            "total_evidence_count": len(self.evidence_signals),
            "phi0_score": self.phi0_score,
            "current_confidence": self.refinement_history[-1].confidence if self.refinement_history else 0.0,
            "evidence_summary": {s.name: s.score for s in self.evidence_signals},
            "streaming_history": len(self.refinement_history)
        }


class StreamingDetectionAggregator:
    """
    Optimized streaming aggregator with dynamic polarity interpretation.
    
    Combines φ⁰ base scores with progressive feature evidence using
    context-aware polarity assignment and efficient early decision logic.
    """
    
    def __init__(self, 
                 base_score: float = 0.5,
                 early_decision_threshold: float = 0.85,
                 min_modules_for_decision: int = 2,
                 polarity_preferences: Dict[str, str] = None):
        """
        Initialize optimized streaming aggregator
        
        Args:
            base_score: Neutral starting score (G₂ approach)
            early_decision_threshold: Confidence threshold for early decisions
            min_modules_for_decision: Minimum modules needed before early decision
            polarity_preferences: Dict mapping feature names to polarity preferences ("positive", "negative", or None)
        """
        self.base_score = base_score
        self.early_decision_threshold = early_decision_threshold
        self.min_modules_for_decision = min_modules_for_decision
        self.polarity_preferences = polarity_preferences or {}
        
        # State tracking
        self.evidence_signals: List[EvidenceSignal] = []
        self.expected_modules = 0
        self.streaming_history: List[StreamingAggregationResult] = []
    
    def reset(self):
        """Reset aggregator for new detection"""
        self.evidence_signals.clear()
        self.streaming_history.clear()        
        
    def set_expected_modules(self, count: int):
        """Set the expected number of feature modules"""
        self.expected_modules = count
        
    def add_evidence(self, name: str, result: FeatureResult, weight: float = 1.0):
        """
        Add evidence signal from a feature module with dynamic polarity interpretation
        
        Args:
            name: Name of the feature module
            result: FeatureResult from the module
            weight: Weight for this evidence in aggregation
        """
        if result.valid:
            # Check if FeatureResult has explicit polarity set
            if hasattr(result, 'polarity') and result.polarity in ["positive", "negative"]:
                # Use explicit polarity from the feature module
                interpreted_polarity = result.polarity
                adjusted_weight = weight
            else:
                # For neutral/unspecified features, check polarity preferences first
                if name in self.polarity_preferences:
                    preferred_polarity = self.polarity_preferences[name]
                    if preferred_polarity in ["positive", "negative"]:
                        # Use configured polarity preference
                        interpreted_polarity = preferred_polarity
                        adjusted_weight = weight
                    else:
                        # Fall back to dynamic interpretation
                        interpreted_polarity, adjusted_weight = self._interpret_polarity(
                            name, result.score, result.metadata or {}, weight
                        )
                else:
                    # No preference set, use dynamic interpretation
                    interpreted_polarity, adjusted_weight = self._interpret_polarity(
                        name, result.score, result.metadata or {}, weight
                    )
            
            signal = EvidenceSignal(
                name=name,
                score=max(0.0, min(1.0, result.score)),
                weight=adjusted_weight,
                polarity=interpreted_polarity,
                metadata=result.metadata or {},
                valid=result.valid
            )
            self.evidence_signals.append(signal)
    
    def aggregate_streaming(self) -> StreamingAggregationResult:
        """
        Efficient streaming aggregation with dynamic polarity and early decision logic
        
        Returns:
            StreamingAggregationResult with current state and streaming info
        """
        valid_signals = [s for s in self.evidence_signals if s.valid]
        modules_completed = len(valid_signals)
        completion_percentage = (modules_completed / self.expected_modules * 100) if self.expected_modules > 0 else 0
        
        if not valid_signals:
            # No evidence available yet
            return StreamingAggregationResult(
                final_score=self.base_score,
                confidence=0.3,  # Low confidence without evidence
                phi0_contribution=0.0,
                feature_contribution=0.0,
                positive_evidence_count=0,
                negative_evidence_count=0,
                evidence_count=0,
                reason="No evidence available yet",
                metadata={"base_score": self.base_score},
                modules_completed=0,
                modules_total=self.expected_modules,
                completion_percentage=0.0,
                streaming_confidence=0.3,
                early_decision_possible=False
            )
        
        # Separate positive and negative evidence
        positive_signals = [s for s in valid_signals if s.polarity == "positive"]
        negative_signals = [s for s in valid_signals if s.polarity == "negative"]
        
        # Calculate weighted evidence scores
        pos_evidence = self._calculate_weighted_evidence(positive_signals)
        neg_evidence = self._calculate_weighted_evidence(negative_signals)
        
        # G₂ bidirectional scoring: base + evidence delta
        evidence_delta = (pos_evidence - neg_evidence) * 0.5
        final_score = max(0.0, min(1.0, self.base_score + evidence_delta))
        
        # Enhanced confidence calculation
        total_weight = sum(s.weight for s in valid_signals)
        completion_factor = min(1.0, modules_completed / max(1, self.expected_modules))
        
        # Calculate evidence strength and agreement
        total_evidence = pos_evidence + neg_evidence
        evidence_strength = total_evidence / 2.0 if total_evidence > 0 else 0.0
        evidence_agreement = 1.0 - abs(pos_evidence - neg_evidence) if total_evidence > 0 else 0.5
        
        # Base confidence combines strength, agreement, and completion
        base_confidence = evidence_strength * evidence_agreement * (0.4 + 0.6 * completion_factor)
        
        # Boost confidence for clear decisions
        if abs(final_score - 0.5) > 0.15:  # Clear positive or negative (lowered threshold)
            base_confidence *= 1.5  # Higher boost
        
        # Additional boost for strong evidence
        if evidence_strength > 0.7:
            base_confidence *= 1.2
        
        streaming_confidence = base_confidence
        
        # Early decision logic
        early_decision_possible = self._can_make_early_decision(
            final_score, streaming_confidence, modules_completed
        )
        
        # Generate comprehensive reason
        reason = self._generate_reason(positive_signals, negative_signals, final_score)
        
        # Collect metadata
        metadata = {
            "base_score": self.base_score,
            "positive_evidence": pos_evidence,
            "negative_evidence": neg_evidence,
            "evidence_delta": evidence_delta,
            "total_weight": total_weight,
            "completion_factor": completion_factor,
            "evidence_agreement": evidence_agreement,
            "positive_details": {s.name: {"score": s.score, "weight": s.weight} for s in positive_signals},
            "negative_details": {s.name: {"score": s.score, "weight": s.weight} for s in negative_signals}
        }
        
        # Create streaming result
        result = StreamingAggregationResult(
            final_score=final_score,
            confidence=base_confidence,
            phi0_contribution=0.0,  # G₂ approach doesn't separate φ⁰
            feature_contribution=1.0,
            positive_evidence_count=len(positive_signals),
            negative_evidence_count=len(negative_signals),
            positive_evidence_score=pos_evidence,
            negative_evidence_score=neg_evidence,
            evidence_count=modules_completed,
            reason=reason,
            metadata=metadata,
            modules_completed=modules_completed,
            modules_total=self.expected_modules,
            completion_percentage=completion_percentage,
            streaming_confidence=streaming_confidence,
            early_decision_possible=early_decision_possible,
            next_module_priority=self._suggest_next_priority()
        )
        
        self.streaming_history.append(result)
        return result
    
    def streaming_aggregate(self, available_modules: List[str] = None, total_modules: int = None) -> StreamingAggregationResult:
        """
        Perform streaming aggregation with available evidence so far
        
        Args:
            available_modules: List of module names that have completed
            total_modules: Total number of expected modules
            
        Returns:
            StreamingAggregationResult with partial evidence
        """
        # Get available evidence
        if available_modules:
            available_signals = [s for s in self.evidence_signals if s.name in available_modules and s.valid]
        else:
            available_signals = [s for s in self.evidence_signals if s.valid]
        
        modules_completed = len(available_signals)
        modules_total = total_modules or len(self.evidence_signals) or modules_completed
        completion_percentage = modules_completed / max(1, modules_total)
        
        if not available_signals:
            # Only φ⁰ available
            base_confidence = self.phi0_score * 0.6  # Reduced confidence without features
            return StreamingAggregationResult(
                final_score=self.phi0_score,
                confidence=base_confidence,
                streaming_confidence=base_confidence * completion_percentage,
                phi0_contribution=1.0,
                feature_contribution=0.0,
                evidence_count=0,
                reason="Only φ⁰ score available (streaming)",
                metadata={"phi0_score": self.phi0_score},
                modules_completed=modules_completed,
                modules_total=modules_total,
                completion_percentage=completion_percentage,
                early_decision_possible=self._can_make_early_decision(self.phi0_score, [])
            )
        
        # Calculate weighted feature contribution
        total_weight = sum(s.weight for s in available_signals)
        feature_score = sum(s.score * s.weight for s in available_signals) / total_weight
        
        # Combine φ⁰ and feature scores
        final_score = (self.phi0_weight * self.phi0_score + 
                      self.feature_weight * feature_score)
        
        # Calculate confidence with streaming adjustment
        agreement = 1.0 - abs(self.phi0_score - feature_score)
        base_confidence = final_score * (0.7 + 0.3 * agreement)
        
        # Adjust confidence based on completion percentage
        streaming_confidence = base_confidence * (0.5 + 0.5 * completion_percentage)
        
        # Check for early decision capability
        early_decision = self._can_make_early_decision(final_score, available_signals)
        
        # Suggest next priority module
        next_priority = self._suggest_next_priority_module(available_modules or [])
        
        # Generate reason
        module_names = [s.name for s in available_signals[:3]]
        feature_summary = ", ".join([f"{name}={s.score:.3f}" for s, name in zip(available_signals[:3], module_names)])
        reason = f"Streaming: φ⁰={self.phi0_score:.3f}, Features[{modules_completed}/{modules_total}]: {feature_summary}"
        
        # Collect metadata
        metadata = {
            "phi0_score": self.phi0_score,
            "feature_score": feature_score,
            "agreement": agreement,
            "available_modules": available_modules or [],
            "feature_details": {s.name: {"score": s.score, "weight": s.weight} for s in available_signals}
        }
        
        return StreamingAggregationResult(
            final_score=final_score,
            confidence=base_confidence,
            streaming_confidence=streaming_confidence,
            phi0_contribution=self.phi0_weight,
            feature_contribution=self.feature_weight,
            evidence_count=modules_completed,
            reason=reason,
            metadata=metadata,
            modules_completed=modules_completed,
            modules_total=modules_total,
            completion_percentage=completion_percentage,
            early_decision_possible=early_decision,
            next_module_priority=next_priority
        )
    
    def _can_make_early_decision(self, current_score: float, available_signals: List[EvidenceSignal]) -> bool:
        """
        Determine if we can make an early detection decision
        
        Args:
            current_score: Current aggregated score
            available_signals: Available evidence signals
            
        Returns:
            True if early decision is possible
        """
        # High confidence cases
        if current_score > 0.8 or current_score < 0.2:
            return True
        
        # Strong agreement between φ⁰ and features
        if available_signals:
            total_weight = sum(s.weight for s in available_signals)
            feature_score = sum(s.score * s.weight for s in available_signals) / total_weight
            agreement = 1.0 - abs(self.phi0_score - feature_score)
            
            if agreement > 0.8 and len(available_signals) >= 2:
                return True
        
        # Strong φ⁰ signal
        if abs(self.phi0_score - 0.5) > 0.3:
            return True
        
        return False
    
    def _calculate_weighted_evidence(self, signals: List[EvidenceSignal]) -> float:
        """Calculate weighted average of evidence signals"""
        if not signals:
            return 0.0
        total_weight = sum(s.weight for s in signals)
        return sum(s.score * s.weight for s in signals) / total_weight
    
    def _can_make_early_decision(self, score: float, confidence: float, modules_completed: int) -> bool:
        """Determine if early decision is possible based on current evidence"""
        if modules_completed < self.min_modules_for_decision:
            return False
            
        # High confidence cases (reduced threshold for testing)
        if confidence >= 0.6:
            return True
            
        # Clear positive or negative cases
        if (score > 0.75 and confidence > 0.4) or (score < 0.25 and confidence > 0.4):
            return True
            
        return False
    
    def _generate_reason(self, pos_signals: List[EvidenceSignal], 
                        neg_signals: List[EvidenceSignal], final_score: float) -> str:
        """Generate human-readable reason for the current decision"""
        reasons = []
        
        if pos_signals:
            pos_names = [f"{s.name}={s.score:.2f}" for s in pos_signals[:2]]
            reasons.append(f"✓ Positive: {', '.join(pos_names)}")
            
        if neg_signals:
            neg_names = [f"{s.name}={s.score:.2f}" for s in neg_signals[:2]]
            reasons.append(f"✗ Negative: {', '.join(neg_names)}")
        
        base_reason = f"G₂ Score: {final_score:.3f}"
        if reasons:
            return f"{base_reason}, {', '.join(reasons)}"
        return base_reason
    
    def _suggest_next_priority(self) -> Optional[str]:
        """Suggest next module to execute based on current evidence state"""
        completed_names = {s.name for s in self.evidence_signals}
        
        # Priority based on discriminative power
        priority_order = [
            "ElevationHistogram",  # Core φ⁰ evidence
            "VolumeAnalysis",      # Most discriminative
            "Compactness",         # Geometric validation
            "EdgeAnalysis",        # Boundary evidence
            "EntropyAnalysis"       # Chaos detection
        ]
        
        for module_name in priority_order:
            if module_name not in completed_names:
                return module_name
        
        return None
    
    def _interpret_polarity(self, module_name: str, score: float, metadata: Dict[str, Any], weight: float) -> Tuple[str, float]:
        """
        Dynamically interpret polarity of neutral evidence based on context
        
        Args:
            module_name: Name of the feature module
            score: Raw feature score
            metadata: Feature metadata for context
            weight: Original weight
            
        Returns:
            Tuple of (interpreted_polarity, adjusted_weight)
        """
        # Check for explicit polarity preference from profile configuration
        if module_name in self.polarity_preferences:
            preferred_polarity = self.polarity_preferences[module_name]
            if preferred_polarity in ["positive", "negative"]:
                # Use configured polarity preference, but still adjust weight based on score strength
                score_strength = max(0.1, score)  # Avoid zero weight
                adjusted_weight = weight * score_strength
                return preferred_polarity, adjusted_weight
        
        # Very low scores are typically negative evidence
        if score < 0.1:
            return "negative", weight * 0.8
            
        # Context-aware polarity interpretation based on kernel.md specifications
        if module_name == "ElevationEntropy" or module_name == "EntropyAnalysis":
            entropy = metadata.get('combined_entropy', score)
            if entropy > 0.7:  # High entropy = chaos = negative evidence
                return "negative", weight
            elif entropy < 0.3:  # Low entropy = order = positive evidence  
                return "positive", weight * 0.8
            else:  # Medium entropy = inconclusive, slightly negative
                return "negative", weight * 0.5
                
        elif module_name == "Volume" or module_name == "VolumeAnalysis":
            volume = metadata.get('normalized_volume', score)
            if volume > 0.9:  # Excessive volume = likely natural feature
                return "negative", weight * 1.2
            elif volume > 0.4:  # Meaningful volume = structure (more discriminating)
                return "positive", weight
            else:  # Insufficient volume = negative
                return "negative", weight * 0.9
                
        elif module_name == "DropoffSharpness" or module_name == "EdgeAnalysis":
            sharpness = metadata.get('edge_sharpness', score)
            max_edge = metadata.get('max_edge_strength', 0)
            
            # Sharp edges only meaningful if there's actual elevation change
            if sharpness > 0.5 and max_edge > 0.3:  # Sharp edges with meaningful elevation
                return "positive", weight
            elif sharpness < 0.2:  # Very gradual edges
                return "negative", weight
            elif max_edge < 0.1:  # No meaningful edges (flat terrain)
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.4  # Medium sharpness = weak positive
                
        elif module_name == "Compactness":
            compactness = metadata.get('compactness', score)
            elevation_range = metadata.get('value_range', 0)
            mean_elevation = metadata.get('mean_elevation', 0)
            
            # High compactness only meaningful if there's significant elevation variation
            if compactness > 0.8 and elevation_range > 0.5 and mean_elevation > 0.3:
                return "positive", weight
            elif compactness > 0.7 and elevation_range < 0.2:  # High compactness but flat = not structural
                return "negative", weight * 0.8
            elif compactness < 0.4:  # Low compactness = chaos
                return "negative", weight * 0.9
            else:
                return "positive", weight * 0.5  # Medium compactness = uncertain
                
        elif module_name == "ElevationHistogram":
            phi0_sig = metadata.get('phi0_signature', score)
            if phi0_sig > 0.6:  # Strong φ⁰ signature
                return "positive", weight * 1.2  # Boost core φ⁰ evidence
            elif phi0_sig < 0.3:  # Weak φ⁰ signature
                return "negative", weight
            else:
                return "positive", weight
        
        # Default: treat high scores as positive, low as negative (more discriminating)
        if score > 0.7:
            return "positive", weight
        elif score < 0.3:
            return "negative", weight * 0.9
        else:
            return "positive", weight * 0.5  # Less confident for medium scores


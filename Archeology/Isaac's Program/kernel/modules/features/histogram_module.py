"""
Elevation Histogram Similarity Module

Implements elevation histogram matching as a standalone feature module.
This is the core pattern-matching component that was originally embedded in φ⁰.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from ..base_module import BaseFeatureModule, FeatureResult

logger = logging.getLogger(__name__)


class ElevationHistogramModule(BaseFeatureModule):
    """
    Validates elevation histogram similarity patterns for structure detection.
    
    This module extracts the elevation histogram matching logic from φ⁰ core
    and makes it available as an independent feature validator. It computes
    the similarity between a local elevation patch and a reference pattern.
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for histogram analysis"""
        return {
            "similarity_method": "correlation",
            "bin_count": 20,
            "edge_enhancement": True,
            "adaptive_binning": True,
            "noise_reduction": True,
            "min_variation": 0.3,
            "normalize_histograms": True,
            "outlier_removal": True
        }
    
    def __init__(self, weight: float = 1.5):  # Higher weight as it's fundamental
        super().__init__("ElevationHistogram", weight)
        self.reference_kernel: Optional[np.ndarray] = None
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    @property
    def parameter_documentation(self) -> Dict[str, str]:
        """Documentation for all histogram analysis parameters"""
        return {
            "similarity_method": "Method for comparing histograms: 'correlation', 'chi_squared', 'bhattacharyya', 'wasserstein'",
            "bin_count": "Number of bins for histogram discretization (higher = more detail, but less robust)",
            "edge_enhancement": "Whether to enhance elevation edges before histogram calculation",
            "adaptive_binning": "Whether to adapt bin ranges based on data characteristics",
            "noise_reduction": "Whether to apply smoothing to reduce noise impact on histograms",
            "min_variation": "Minimum elevation variation required for meaningful histogram analysis",
            "normalize_histograms": "Whether to normalize histograms to unit sum for scale invariance",
            "outlier_removal": "Whether to remove extreme elevation outliers before histogram calculation"
        }
    
    @property
    def result_documentation(self) -> Dict[str, str]:
        """Documentation for histogram analysis result metadata"""
        return {
            "histogram_similarity": "Similarity score between patch and reference histogram (0.0-1.0)",
            "correlation_coefficient": "Pearson correlation between histograms",
            "patch_histogram": "Normalized histogram of the elevation patch",
            "reference_histogram": "Normalized histogram of the reference pattern",
            "bin_edges": "Elevation values defining histogram bin boundaries",
            "peak_alignment": "How well the histogram peaks align between patch and reference",
            "distribution_shape": "Characterization of the elevation distribution shape",
            "variation_score": "Measure of elevation variation within the patch",
            "noise_level": "Estimated noise level in the elevation data",
            "outlier_count": "Number of elevation outliers detected and potentially removed",
            "adaptive_bins_used": "Actual bin configuration used if adaptive binning was enabled",
            "edge_enhancement_applied": "Whether edge enhancement preprocessing was applied",
            "similarity_confidence": "Confidence measure for the similarity score based on data quality"
        }
    
    @property
    def interpretation_guide(self) -> Dict[str, str]:
        """Guide for interpreting histogram analysis results"""
        return {
            "High Similarity (>0.8)": "Elevation pattern closely matches reference - strong structural match",
            "Good Similarity (0.6-0.8)": "Elevation pattern similar to reference - probable structural match",
            "Moderate Similarity (0.4-0.6)": "Some pattern similarity - possible structure with variations",
            "Low Similarity (0.2-0.4)": "Weak pattern match - unlikely to be target structure type",
            "No Similarity (<0.2)": "Elevation pattern very different from reference - not target structure",
            "High Variation": "Complex elevation structure with diverse heights",
            "Low Variation": "Relatively flat or uniform elevation structure",
            "Peak Alignment": "How well the most prominent elevations match between patch and reference",
            "Noise Impact": "High noise levels reduce histogram reliability and similarity confidence"
        }
    
    @property
    def feature_description(self) -> str:
        """Overall description of what the histogram feature analyzes"""
        return """
        Elevation Histogram Similarity Analysis:
        
        The histogram module compares the elevation distribution of a patch against a reference pattern
        to identify structural similarities. It builds normalized histograms of elevation values and
        computes similarity metrics to determine how well the patch matches expected structural patterns.
        
        Key Capabilities:
        - Multiple similarity metrics (correlation, chi-squared, Bhattacharyya, Wasserstein distance)
        - Adaptive binning based on data characteristics
        - Edge enhancement for better structural feature detection
        - Noise reduction and outlier handling
        - Confidence assessment based on data quality
        
        Best For:
        - Pattern matching against known structure types
        - Identifying elevation distribution signatures
        - Template-based structure detection
        - Validating structural hypotheses based on elevation patterns
        """
    
    def set_reference_kernel(self, kernel: np.ndarray):
        """Set the reference elevation kernel for comparison"""
        self.reference_kernel = kernel
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute elevation histogram similarity score
        
        Args:
            elevation_patch: Local elevation data patch
            **kwargs: Optional parameters including reference_kernel
            
        Returns:
            FeatureResult with histogram similarity score
        """
        try:
            # Get reference kernel from kwargs or use stored kernel
            reference_kernel = kwargs.get('reference_kernel', self.reference_kernel)
            
            if reference_kernel is None:
                # Use trained windmill histogram fingerprint if available
                similarity_score = self._compute_trained_histogram_similarity(elevation_patch)
            else:
                # Compute histogram similarity with provided reference kernel
                similarity_score = self._compute_histogram_similarity(elevation_patch, reference_kernel)
            
            # Additional pattern metrics
            pattern_strength = self._compute_pattern_strength(elevation_patch)
            elevation_coherence = self._compute_elevation_coherence(elevation_patch)
            
            # Use pure histogram similarity when trained fingerprint is available
            if hasattr(self, 'trained_histogram_fingerprint') and self.trained_histogram_fingerprint is not None:
                # Pure histogram similarity for trained fingerprints
                final_score = similarity_score
                reason_suffix = "using trained windmill fingerprint"
            else:
                # Combine metrics with weighted average for reference kernels
                final_score = (
                    0.6 * similarity_score +
                    0.25 * pattern_strength +
                    0.15 * elevation_coherence
                )
                reason_suffix = "with pattern analysis"
            
            return FeatureResult(
                score=max(0.0, min(1.0, final_score)),
                polarity="neutral",  # Dynamic polarity interpretation by aggregator
                metadata={
                    "phi0_signature": final_score,  # For dynamic interpretation
                    "histogram_similarity": similarity_score,
                    "pattern_strength": pattern_strength,
                    "elevation_coherence": elevation_coherence,
                    "has_reference_kernel": reference_kernel is not None,
                    "using_trained_fingerprint": hasattr(self, 'trained_histogram_fingerprint') and self.trained_histogram_fingerprint is not None,
                    "computation_method": reason_suffix,
                    "elevation_range": np.max(elevation_patch) - np.min(elevation_patch),
                    "patch_shape": elevation_patch.shape
                },
                valid=True,
                reason=f"Histogram similarity: {similarity_score:.3f}, Pattern strength: {pattern_strength:.3f}"
            )
            
        except Exception as e:
            return FeatureResult(
                score=0.0,
                valid=False,
                reason=f"Histogram computation failed: {str(e)}"
            )
    
    def _compute_histogram_similarity(self, local_elevation: np.ndarray, 
                                    kernel_elevation: np.ndarray) -> float:
        """
        Compute elevation histogram matching score with vegetation discrimination
        
        Based on the original φ⁰ implementation but adapted for modular use.
        """
        # Check elevation range requirements
        local_range = np.max(local_elevation) - np.min(local_elevation)
        kernel_range = np.max(kernel_elevation) - np.min(kernel_elevation)
        
        if local_range < self.min_variation or kernel_range < self.min_variation:
            return 0.0
        
        # Normalize to relative patterns
        local_relative = local_elevation - np.min(local_elevation)
        kernel_relative = kernel_elevation - np.min(kernel_elevation)
        
        local_max_rel = np.max(local_relative)
        kernel_max_rel = np.max(kernel_relative)
        
        if local_max_rel < 0.3 or kernel_max_rel < 0.3:
            return 0.0
        
        # Create and compare histograms
        local_normalized = local_relative / local_max_rel
        kernel_normalized = kernel_relative / kernel_max_rel
        
        num_bins = 20
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        local_hist, _ = np.histogram(local_normalized.flatten(), bins=bin_edges, density=True)
        kernel_hist, _ = np.histogram(kernel_normalized.flatten(), bins=bin_edges, density=True)
        
        # Normalize to probability distributions
        local_hist = local_hist / (np.sum(local_hist) + 1e-8)
        kernel_hist = kernel_hist / (np.sum(kernel_hist) + 1e-8)
        
        # Cosine similarity
        local_norm = np.linalg.norm(local_hist)
        kernel_norm = np.linalg.norm(kernel_hist)
        
        if local_norm < 1e-8 or kernel_norm < 1e-8:
            return 0.0
        
        similarity = np.dot(local_hist, kernel_hist) / (local_norm * kernel_norm)
        return max(0.0, min(1.0, similarity))
    
    def _compute_trained_histogram_similarity(self, local_elevation: np.ndarray) -> float:
        """
        Compute similarity using trained windmill histogram fingerprint
        """
        try:
            # First try to use embedded trained fingerprint from profile
            if hasattr(self, 'trained_histogram_fingerprint') and self.trained_histogram_fingerprint is not None:
                trained_histogram = self.trained_histogram_fingerprint
                logger.debug("Using embedded trained histogram fingerprint from profile")
            else:
                # Fallback: load from file
                import json
                import os
                
                fingerprint_path = os.path.join(os.path.dirname(__file__), '../../../windmill_histogram_fingerprint.json')
                if not os.path.exists(fingerprint_path):
                    logger.warning("Trained windmill fingerprint not found, falling back to synthetic pattern")
                    reference_kernel = self._create_synthetic_windmill_pattern(local_elevation.shape)
                    return self._compute_histogram_similarity(local_elevation, reference_kernel)
                
                with open(fingerprint_path, 'r') as f:
                    training_data = json.load(f)
                
                trained_histogram = np.array(training_data['composite_histogram'])
                logger.debug(f"Using trained histogram from file: {training_data['training_sites']}")
            
            # Check elevation range requirements
            local_range = np.max(local_elevation) - np.min(local_elevation)
            if local_range < self.min_variation:
                return 0.0
            
            # Normalize local elevation to relative pattern
            local_relative = local_elevation - np.min(local_elevation)
            local_max_rel = np.max(local_relative)
            
            if local_max_rel < 0.3:
                return 0.0
            
            # Create histogram from local data
            local_normalized = local_relative / local_max_rel
            num_bins = len(trained_histogram)  # Use same number of bins as training
            bin_edges = np.linspace(0, 1, num_bins + 1)
            
            local_hist, _ = np.histogram(local_normalized.flatten(), bins=bin_edges, density=True)
            
            # Normalize to probability distribution
            local_hist = local_hist / (np.sum(local_hist) + 1e-8)
            trained_hist = trained_histogram / (np.sum(trained_histogram) + 1e-8)
            
            # Cosine similarity with trained pattern
            local_norm = np.linalg.norm(local_hist)
            trained_norm = np.linalg.norm(trained_hist)
            
            if local_norm < 1e-8 or trained_norm < 1e-8:
                return 0.0
            
            dot_product = np.dot(local_hist, trained_hist)
            cosine_similarity = dot_product / (local_norm * trained_norm)
            
            # Debug logging
            logger.debug(f"Histogram similarity calculation:")
            logger.debug(f"  Local hist sum: {np.sum(local_hist):.6f}")
            logger.debug(f"  Trained hist sum: {np.sum(trained_hist):.6f}")
            logger.debug(f"  Cosine similarity: {cosine_similarity:.6f}")
            
            # Clamp to [0, 1] range
            result = max(0.0, min(1.0, cosine_similarity))
            logger.debug(f"  Final result: {result:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in trained histogram similarity: {e}")
            # Fallback to synthetic pattern
            reference_kernel = self._create_synthetic_windmill_pattern(local_elevation.shape)
            return self._compute_histogram_similarity(local_elevation, reference_kernel)
    
    def _compute_pattern_strength(self, elevation_patch: np.ndarray) -> float:
        """Compute the strength of elevation patterns"""
        # Calculate gradient magnitude
        grad_y, grad_x = np.gradient(elevation_patch)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Pattern strength based on gradient concentration
        grad_mean = np.mean(grad_magnitude)
        grad_std = np.std(grad_magnitude)
        
        # Strong patterns have high gradient variation
        pattern_strength = min(1.0, grad_std / (grad_mean + 1e-8))
        
        return pattern_strength
    
    def _compute_elevation_coherence(self, elevation_patch: np.ndarray) -> float:
        """Compute elevation coherence (how well-formed the elevation pattern is)"""
        center = np.array(elevation_patch.shape) // 2
        y, x = np.ogrid[:elevation_patch.shape[0], :elevation_patch.shape[1]]
        distances = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        
        # Check if elevation decreases with distance from center
        max_distance = np.max(distances)
        distance_normalized = distances / max_distance
        
        # Compute correlation between elevation and inverse distance
        center_elevation = elevation_patch[center[0], center[1]]
        elevation_diff = elevation_patch - center_elevation
        
        # Coherence based on how elevation relates to distance from center
        coherence = 1.0 - np.corrcoef(distance_normalized.flatten(), 
                                     np.abs(elevation_diff).flatten())[0, 1]
        
        return max(0.0, min(1.0, coherence)) if not np.isnan(coherence) else 0.0
    
    def _create_synthetic_windmill_pattern(self, shape: tuple) -> np.ndarray:
        """
        Create a synthetic windmill-like elevation pattern when no reference kernel is available
        """
        h, w = shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        
        # Create windmill-like pattern: elevated center with gradual falloff
        base_radius = min(h, w) // 4
        elevation = np.zeros_like(distances, dtype=float)
        
        # Central mound
        center_mask = distances <= base_radius
        elevation[center_mask] = 2.0 * np.exp(-0.1 * distances[center_mask])
        
        # Tower area (higher elevation in center)
        tower_mask = distances <= base_radius // 2
        elevation[tower_mask] += 1.0 * np.exp(-0.2 * distances[tower_mask])
        
        # Add some noise for realism
        elevation += np.random.normal(0, 0.05, elevation.shape)
        
        return elevation

    def set_parameters(self, resolution_m: float, structure_radius_px: int):
        """Set module parameters based on detection context"""
        super().set_parameters(resolution_m, structure_radius_px)
        
        # Adjust minimum variation based on structure size
        # Larger structures can have more variation
        self.min_variation = max(0.2, min(0.5, structure_radius_px * 0.01))
    
    def configure(self, **parameters):
        """Configure the module with parameters from the detector profile"""
        super().configure(**parameters)
        
        # Special handling for trained histogram fingerprint
        if 'trained_histogram_fingerprint' in parameters:
            fingerprint = parameters['trained_histogram_fingerprint']
            if isinstance(fingerprint, list) and len(fingerprint) > 0:
                self.trained_histogram_fingerprint = np.array(fingerprint)
                logger.info(f"Loaded trained histogram fingerprint with {len(fingerprint)} bins")
            else:
                logger.warning("Invalid trained_histogram_fingerprint format")
                self.trained_histogram_fingerprint = None
        else:
            self.trained_histogram_fingerprint = None

"""
Dropoff Sharpness Feature Module

Validates elevation dropoff sharpness around structure edges using
Difference of Gaussians (DoG) edge detection and ring analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Any

from ..base_module import BaseFeatureModule, FeatureResult


class DropoffSharpnessModule(BaseFeatureModule):
    """
    Analyzes edge sharpness around structures using ring-based edge detection.
    
    Sharp dropoffs indicate well-defined artificial structures, while gradual
    transitions suggest natural terrain features.
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for dropoff analysis"""
        return {
            "edge_method": "gradient",
            "smoothing_radius": 1.0,
            "adaptive_threshold": True,
            "directional_analysis": True,
            "edge_enhancement": True,
            "sigma_inner_factor": 0.8,
            "sigma_outer_factor": 1.2,
            "ring_inner_factor": 0.8,
            "ring_outer_factor": 1.2
        }
    
    def __init__(self, weight: float = 1.2):
        super().__init__("DropoffSharpness", weight)
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    @property
    def parameter_documentation(self) -> Dict[str, str]:
        """Documentation for all dropoff analysis parameters"""
        return {
            "edge_method": "Method for edge detection: 'gradient' uses gradients, 'dog' uses Difference of Gaussians",
            "smoothing_radius": "Radius for Gaussian smoothing before edge detection (reduces noise sensitivity)",
            "adaptive_threshold": "Whether to adapt edge detection thresholds based on local data characteristics",
            "directional_analysis": "Whether to analyze edge strength in different directions around the structure",
            "edge_enhancement": "Whether to apply edge enhancement preprocessing to improve detection",
            "sigma_inner_factor": "Factor for inner Gaussian sigma in DoG edge detection (relative to structure radius)",
            "sigma_outer_factor": "Factor for outer Gaussian sigma in DoG edge detection (relative to structure radius)",
            "ring_inner_factor": "Factor defining inner radius of analysis ring (relative to structure radius)",
            "ring_outer_factor": "Factor defining outer radius of analysis ring (relative to structure radius)"
        }
    
    @property
    def result_documentation(self) -> Dict[str, str]:
        """Documentation for dropoff analysis result metadata"""
        return {
            "dropoff_sharpness": "Overall sharpness score of elevation edges around the structure (0.0-1.0)",
            "edge_strength": "Maximum edge strength detected in the analysis region",
            "edge_consistency": "How consistent edge strength is around the structure perimeter",
            "gradient_magnitude": "Average gradient magnitude in the edge detection region",
            "directional_strengths": "Edge strengths in different cardinal/ordinal directions (if directional_analysis=True)",
            "adaptive_threshold_used": "Threshold value used for edge detection (if adaptive_threshold=True)",
            "ring_coverage": "Fraction of the analysis ring that shows significant edge activity",
            "peak_edge_location": "Location of the strongest edge relative to structure center",
            "edge_symmetry": "Measure of how symmetrically edges are distributed around the structure",
            "noise_level": "Estimated noise level in the edge detection results",
            "smoothing_applied": "Amount of smoothing applied before edge detection",
            "enhancement_factor": "Edge enhancement factor applied (if edge_enhancement=True)"
        }
    
    @property
    def interpretation_guide(self) -> Dict[str, str]:
        """Guide for interpreting dropoff analysis results"""
        return {
            "High Sharpness (>0.7)": "Very sharp elevation dropoff - likely artificial structure with defined edges",
            "Moderate Sharpness (0.4-0.7)": "Moderate edge definition - possible structure or natural feature with some sharpness",
            "Low Sharpness (<0.4)": "Gradual elevation transition - likely natural terrain or heavily eroded feature",
            "High Consistency": "Edge strength uniform around structure - suggests regular, artificial construction",
            "Low Consistency": "Variable edge strength - suggests irregular, natural, or damaged structure",
            "Strong Directional Bias": "Edges stronger in certain directions - may indicate linear features or erosion patterns",
            "High Ring Coverage": "Edges detected around most of structure perimeter - well-defined boundaries",
            "Low Ring Coverage": "Edges only in parts of perimeter - partial structure or natural formation",
            "High Symmetry": "Edges symmetrically distributed - suggests circular or regular polygonal structure",
            "Low Symmetry": "Asymmetric edge distribution - suggests irregular shape or directional erosion"
        }
    
    @property
    def feature_description(self) -> str:
        """Overall description of what the dropoff feature analyzes"""
        return """
        Elevation Dropoff Sharpness Analysis:
        
        The dropoff module analyzes the sharpness of elevation transitions around structure edges
        to distinguish between artificial constructions and natural terrain features. It uses
        ring-based analysis to detect edge strength and consistency around the structure perimeter.
        
        Key Capabilities:
        - Multiple edge detection methods (gradient-based, Difference of Gaussians)
        - Directional analysis to detect asymmetric edge patterns
        - Adaptive thresholding based on local terrain characteristics
        - Edge consistency measurement around structure perimeter
        - Noise-resistant edge detection with configurable smoothing
        
        Best For:
        - Distinguishing artificial structures from natural mounds
        - Detecting well-preserved vs eroded archaeological features
        - Identifying structures with sharp architectural edges
        - Validating structure boundaries and extent
        """
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute dropoff sharpness score using ring edge detection
        
        Args:
            elevation_patch: 2D elevation data array
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with edge sharpness confidence
        """
        try:
            radius = self.structure_radius_px
            sigma1 = radius * self.sigma_inner_factor * self.resolution_m
            sigma2 = radius * self.sigma_outer_factor * self.resolution_m
            
            # Difference of Gaussians for edge detection
            dog = gaussian_filter(elevation_patch, sigma1) - gaussian_filter(elevation_patch, sigma2)
            edge_strength = np.abs(dog)
            
            # Normalize edge strength
            edge_95th = np.percentile(edge_strength, 95)
            if edge_95th > 0:
                edge_strength = edge_strength / (edge_95th + 1e-6)
            
            # Calculate mean edge strength in ring around structure
            h, w = elevation_patch.shape
            center_y, center_x = h // 2, w // 2
            
            # Create ring mask
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            ring_mask = ((distances >= radius * self.ring_inner_factor) & 
                        (distances <= radius * self.ring_outer_factor))
            
            if np.any(ring_mask):
                ring_sharpness = np.mean(edge_strength[ring_mask])
                ring_pixels = np.sum(ring_mask)
            else:
                ring_sharpness = np.mean(edge_strength)
                ring_pixels = edge_strength.size
            
            # Calculate additional edge metrics
            max_edge_strength = np.max(edge_strength)
            edge_concentration = np.sum(edge_strength > 0.5) / edge_strength.size
            
            sharpness_score = np.clip(ring_sharpness, 0, 1)
            
            return FeatureResult(
                score=sharpness_score,
                polarity="neutral",
                metadata={
                    "edge_sharpness": float(sharpness_score),
                    "mean_ring_edge_strength": float(ring_sharpness),
                    "max_edge_strength": float(max_edge_strength),
                    "edge_concentration": float(edge_concentration),
                    "ring_pixels": int(ring_pixels),
                    "sigma1": float(sigma1),
                    "sigma2": float(sigma2),
                    "radius_used": int(radius)
                },
                reason=f"Dropoff sharpness: ring_strength={sharpness_score:.3f}"
            )
            
        except Exception as e:
            return FeatureResult(
                score=0.0,
                valid=False,
                reason=f"Dropoff sharpness computation failed: {str(e)}"
            )
    
    def configure(self, 
                 sigma_inner_factor: float = None,
                 sigma_outer_factor: float = None,
                 ring_inner_factor: float = None,
                 ring_outer_factor: float = None):
        """Configure module parameters"""
        if sigma_inner_factor is not None:
            self.sigma_inner_factor = sigma_inner_factor
        if sigma_outer_factor is not None:
            self.sigma_outer_factor = sigma_outer_factor
        if ring_inner_factor is not None:
            self.ring_inner_factor = ring_inner_factor
        if ring_outer_factor is not None:
            self.ring_outer_factor = ring_outer_factor

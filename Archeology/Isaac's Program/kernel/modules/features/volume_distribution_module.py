"""
Volume Distribution Feature Module

Analyzes how volume is distributed vertically to distinguish:
- Windmill mounds: Volume concentrated near base (low vertical spread)
- Buildings: Volume distributed throughout height (high vertical spread)
"""

import numpy as np
import logging
from typing import Dict, Any

from ..base_module import BaseFeatureModule, FeatureResult

logger = logging.getLogger(__name__)


class VolumeDistributionModule(BaseFeatureModule):
    """
    Analyzes vertical volume distribution to distinguish windmill mounds from buildings.
    
    Windmills: Volume concentrated near base level (mound structure)
    Buildings: Volume distributed throughout height (solid structures)
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for volume distribution analysis"""
        return {
            "height_bins": 10,
            "base_fraction_threshold": 0.7,  # Fraction of volume that should be in lower bins for windmills
            "border_width_factor": 0.25,
            "min_height_range": 2.0,  # Minimum height range to consider
        }
    
    def __init__(self, weight: float = 1.0):
        super().__init__("VolumeDistribution", weight)
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute volume distribution score
        
        Args:
            elevation_patch: Local elevation data patch
            
        Returns:
            FeatureResult with volume distribution analysis
        """
        try:
            h, w = elevation_patch.shape
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            
            # Create circular mask for structure
            y, x = np.ogrid[:h, :w]
            structure_mask = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
            
            # Calculate base elevation from edges
            border_width = max(2, int(radius * self.border_width_factor))
            edge_mask = np.zeros_like(elevation_patch, dtype=bool)
            edge_mask[:border_width, :] = True
            edge_mask[-border_width:, :] = True
            edge_mask[:, :border_width] = True
            edge_mask[:, -border_width:] = True
            
            if np.any(edge_mask):
                base_elevation = np.median(elevation_patch[edge_mask])
            else:
                base_elevation = np.median(elevation_patch)
            
            # Get structure heights above base
            if np.any(structure_mask):
                structure_heights = elevation_patch[structure_mask]
                heights_above_base = structure_heights - base_elevation
                heights_above_base = np.maximum(0, heights_above_base)
                
                height_range = np.max(heights_above_base) - np.min(heights_above_base)
                
                if height_range < self.min_height_range:
                    return FeatureResult(
                        score=0.5,
                        reason="Insufficient height range for volume distribution analysis"
                    )
                
                # Create height bins
                max_height = np.max(heights_above_base)
                bin_edges = np.linspace(0, max_height, self.height_bins + 1)
                
                # Calculate volume in each height bin
                volume_per_bin = []
                for i in range(self.height_bins):
                    bin_mask = (heights_above_base >= bin_edges[i]) & (heights_above_base < bin_edges[i + 1])
                    if i == self.height_bins - 1:  # Include max value in last bin
                        bin_mask = (heights_above_base >= bin_edges[i])
                    
                    volume_in_bin = np.sum(bin_mask) * np.mean(heights_above_base[bin_mask]) if np.any(bin_mask) else 0
                    volume_per_bin.append(volume_in_bin)
                
                volume_per_bin = np.array(volume_per_bin)
                total_volume = np.sum(volume_per_bin)
                
                if total_volume > 0:
                    volume_distribution = volume_per_bin / total_volume
                    
                    # Calculate windmill-likeness metrics
                    base_bins = int(self.height_bins * 0.4)  # Bottom 40% of height range
                    base_volume_fraction = np.sum(volume_distribution[:base_bins])
                    
                    # Volume concentration in base (good for windmills)
                    base_concentration_score = min(1.0, base_volume_fraction / self.base_fraction_threshold)
                    
                    # Inverse of top-heavy distribution (bad for windmills)
                    top_bins = int(self.height_bins * 0.6)  # Top 60% of height range
                    top_volume_fraction = np.sum(volume_distribution[top_bins:])
                    top_heavy_penalty = top_volume_fraction
                    
                    # Combined score: favor base concentration, penalize top-heavy
                    distribution_score = base_concentration_score * (1.0 - top_heavy_penalty * 0.5)
                    distribution_score = max(0.0, min(1.0, distribution_score))
                    
                    return FeatureResult(
                        score=distribution_score,
                        polarity="positive",
                        metadata={
                            "volume_distribution": volume_distribution.tolist(),
                            "base_volume_fraction": float(base_volume_fraction),
                            "top_volume_fraction": float(top_volume_fraction),
                            "base_concentration_score": float(base_concentration_score),
                            "top_heavy_penalty": float(top_heavy_penalty),
                            "height_range": float(height_range),
                            "max_height": float(max_height),
                            "base_elevation": float(base_elevation),
                            "total_volume": float(total_volume)
                        },
                        reason=f"Volume distribution: {base_volume_fraction:.1%} in base, {top_volume_fraction:.1%} in top"
                    )
                else:
                    return FeatureResult(
                        score=0.0,
                        reason="No significant volume above base elevation"
                    )
            else:
                return FeatureResult(
                    score=0.0,
                    reason="No structure detected in elevation patch"
                )
                
        except Exception as e:
            return FeatureResult(
                score=0.0,
                valid=False,
                reason=f"Volume distribution computation failed: {str(e)}"
            )

"""
Volume Feature Module

Validates structure volume and prominence by analyzing elevation
distribution and mass concentration with fully adaptive normalization.
"""

import numpy as np
import logging
from typing import Dict, Any

from ..base_module import BaseFeatureModule, FeatureResult

logger = logging.getLogger(__name__)


class VolumeModule(BaseFeatureModule):
    """
    Analyzes structure volume and height prominence with adaptive scaling.
    
    Uses data-driven adaptive normalization without hardcoded structure types.
    Learns optimal scaling from local elevation characteristics.
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for volume analysis"""
        return {
            "volume_method": "adaptive",
            "base_volume_normalization": 50.0,
            "base_prominence_normalization": 5.0,
            "border_width_factor": 0.25,
            "adaptive_scaling": True,
            "size_scaling_factor": 1.0,
            "area_scaling_reference": 100.0,
            "context_weight": 0.3,
            "concentration_bonus": 1.1,
            "relative_prominence_weight": 0.4,
            "min_volume_threshold": 5.0,
            "max_volume_saturation": 1000.0,
            "min_prominence_threshold": 0.5,
            "max_prominence_saturation": 20.0,
            "auto_range_adaptation": True,
            "percentile_normalization": True,
            "local_statistics_radius": 2.0
        }
    
    def __init__(self, weight: float = 1.3):
        super().__init__("Volume", weight)
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    @property
    def parameter_documentation(self) -> Dict[str, str]:
        """Documentation for all volume analysis parameters"""
        return {
            "volume_method": "Method for volume calculation: 'adaptive' uses data-driven scaling, 'fixed' uses preset values",
            "base_volume_normalization": "Base volume value used for normalization (cubic units)",
            "base_prominence_normalization": "Base prominence value used for normalization (elevation units)",
            "border_width_factor": "Fraction of patch size used as border for baseline calculation (0.0-0.5)",
            "adaptive_scaling": "Whether to adapt normalization values based on local data characteristics",
            "size_scaling_factor": "Multiplier for size-based volume scaling adjustments",
            "area_scaling_reference": "Reference area (pixels) for size-based scaling calculations",
            "context_weight": "Weight given to contextual information in adaptive scaling (0.0-1.0)",
            "concentration_bonus": "Bonus multiplier for concentrated volume distributions (≥1.0)",
            "relative_prominence_weight": "Weight for relative prominence vs absolute volume (0.0-1.0)",
            "min_volume_threshold": "Minimum volume threshold below which structures are considered insignificant",
            "max_volume_saturation": "Maximum volume above which the score saturates to prevent outliers",
            "min_prominence_threshold": "Minimum prominence threshold for meaningful elevation differences",
            "max_prominence_saturation": "Maximum prominence above which the score saturates",
            "auto_range_adaptation": "Whether to automatically adapt thresholds based on data range",
            "percentile_normalization": "Whether to use percentile-based normalization for robustness",
            "local_statistics_radius": "Radius (in structure units) for calculating local statistics"
        }
    
    @property
    def result_documentation(self) -> Dict[str, str]:
        """Documentation for volume analysis result metadata"""
        return {
            "volume": "Calculated volume of the structure (normalized)",
            "prominence": "Height prominence above local baseline (normalized)",
            "concentration": "Measure of how concentrated the volume is (0.0-1.0)",
            "baseline_elevation": "Calculated baseline elevation for prominence measurement",
            "peak_elevation": "Maximum elevation within the structure",
            "volume_raw": "Raw volume calculation before normalization",
            "prominence_raw": "Raw prominence calculation before normalization",
            "normalization_factor_volume": "Factor used to normalize volume values",
            "normalization_factor_prominence": "Factor used to normalize prominence values",
            "adaptive_scaling_applied": "Whether adaptive scaling was used in calculations",
            "size_factor": "Size-based scaling factor applied to the result",
            "concentration_bonus_applied": "Bonus factor applied for concentrated distributions",
            "border_statistics": "Statistical properties of the border region used for baseline",
            "patch_area": "Total area of the analysis patch in pixels",
            "effective_structure_area": "Estimated area of the structure itself",
            "volume_density": "Volume per unit area (volume/effective_area)",
            "prominence_consistency": "Measure of how consistent the prominence is across the structure"
        }
    
    @property
    def interpretation_guide(self) -> Dict[str, str]:
        """Guide for interpreting volume analysis results"""
        return {
            "High Volume (>0.7)": "Strong elevation mass, likely significant structure",
            "Medium Volume (0.3-0.7)": "Moderate elevation mass, possible structure or natural feature",
            "Low Volume (<0.3)": "Minimal elevation mass, likely flat terrain or noise",
            "High Prominence (>0.8)": "Structure rises significantly above surroundings",
            "Medium Prominence (0.4-0.8)": "Moderate elevation difference from surroundings",
            "Low Prominence (<0.4)": "Little elevation difference, may be natural variation",
            "High Concentration (>0.7)": "Volume is concentrated in a small area (windmill-like)",
            "Low Concentration (<0.3)": "Volume is spread out (ridge-like or natural)",
            "Adaptive Scaling": "Normalization adapted to local data characteristics for better discrimination",
            "Size Scaling": "Results adjusted based on structure size relative to expected dimensions"
        }
    
    @property
    def feature_description(self) -> str:
        """Overall description of what the volume feature analyzes"""
        return """
        Volume Feature Analysis:
        
        The volume module quantifies the three-dimensional mass and prominence of elevation structures.
        It calculates both absolute volume (total elevation mass) and relative prominence (height above
        local baseline), using adaptive normalization to handle diverse terrain types and structure sizes.
        
        Key Capabilities:
        - Adaptive normalization based on local data characteristics
        - Concentration analysis to distinguish compact vs dispersed features
        - Size-aware scaling for structures of different dimensions
        - Robust baseline calculation using border statistics
        - Percentile-based normalization for outlier resistance
        
        Best For:
        - Detecting raised structures (mounds, platforms, buildings)
        - Distinguishing significant elevation features from noise
        - Quantifying structure mass and prominence
        - Size-based feature discrimination
        """
    
    def _analyze_structure_characteristics(self, elevation_patch: np.ndarray, structure_mask: np.ndarray) -> Dict[str, float]:
        """Analyze structure characteristics to determine adaptive scaling factors"""
        if not np.any(structure_mask):
            return {"volume_scale": 1.0, "prominence_scale": 1.0, "complexity": 0.0}
            
        structure_heights = elevation_patch[structure_mask]
        
        # Calculate morphological characteristics
        height_range = np.ptp(structure_heights)
        height_std = np.std(structure_heights)
        height_mean = np.mean(structure_heights)
        height_cv = height_std / (height_mean + 1e-6)  # coefficient of variation
        
        # Shape analysis
        structure_area = np.sum(structure_mask)
        perimeter = self._calculate_perimeter(structure_mask)
        compactness = (4 * np.pi * structure_area) / (perimeter**2 + 1e-6)
        
        # Elevation distribution analysis
        skewness = self._calculate_skewness(structure_heights)
        kurtosis = self._calculate_kurtosis(structure_heights)
        
        # Adaptive scaling based on characteristics
        # More complex structures get different scaling
        complexity_score = height_cv * 0.4 + (1 - compactness) * 0.3 + abs(skewness) * 0.2 + abs(kurtosis - 3) * 0.1
        
        # Volume scaling: more variable structures need different normalization
        volume_scale = 1.0 + complexity_score * 0.5
        
        # Prominence scaling: flatter structures emphasize relative prominence more
        prominence_scale = 1.0 - height_cv * 0.3 if height_cv < 0.5 else 1.0 + height_cv * 0.2
        
        return {
            "volume_scale": np.clip(volume_scale, 0.3, 2.0),
            "prominence_scale": np.clip(prominence_scale, 0.5, 1.5),
            "complexity": complexity_score,
            "height_cv": height_cv,
            "compactness": compactness,
            "height_range": height_range,
            "skewness": skewness
        }
    
    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calculate perimeter of binary mask"""
        try:
            from scipy import ndimage
            # Use gradient to find edges
            edges = ndimage.binary_dilation(mask) ^ mask
            return np.sum(edges)
        except ImportError:
            # Fallback: simple edge detection
            h, w = mask.shape
            perimeter = 0
            for i in range(h):
                for j in range(w):
                    if mask[i, j]:
                        # Check if this pixel is on the edge
                        neighbors = []
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                neighbors.append(mask[ni, nj])
                            else:
                                neighbors.append(False)
                        if not all(neighbors):
                            perimeter += 1
            return perimeter
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 3.0  # Normal distribution kurtosis
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3.0
        return np.mean(((data - mean) / std) ** 4)
    
    def _adaptive_normalization(self, volume: float, prominence: float, 
                              structure_area: float, patch_area: float) -> tuple:
        """Calculate adaptive normalization based on structure and patch characteristics"""
        
        if not self.adaptive_scaling:
            return self.base_volume_normalization, self.base_prominence_normalization
        
        # Scale based on structure size relative to patch
        size_ratio = structure_area / (patch_area + 1e-6)
        area_scale = min(2.0, max(0.5, 1.0 / (size_ratio + 0.1)))
        
        # Scale based on absolute area (larger structures expected to have more volume)
        area_scale_factor = np.sqrt(structure_area / self.area_scaling_reference) * self.size_scaling_factor
        
        # Adaptive volume normalization
        volume_norm = self.base_volume_normalization * area_scale_factor * area_scale
        volume_norm = np.clip(volume_norm, self.min_volume_threshold, self.max_volume_saturation)
        
        # Adaptive prominence normalization (less dependent on area)
        prominence_norm = self.base_prominence_normalization * area_scale
        prominence_norm = np.clip(prominence_norm, self.min_prominence_threshold, self.max_prominence_saturation)
        
        return volume_norm, prominence_norm
    
    def _calculate_local_percentile(self, elevation_patch: np.ndarray, value: float) -> float:
        """Calculate what percentile the given value represents in the local patch"""
        if not self.percentile_normalization:
            return 50.0
            
        try:
            flat_elevations = elevation_patch.flatten()
            percentile = (np.sum(flat_elevations <= value) / len(flat_elevations)) * 100
            return percentile
        except:
            return 50.0  # Default to median
    
    def _calculate_relative_prominence(self, elevation_patch: np.ndarray, 
                                     structure_mask: np.ndarray, height_prominence: float) -> float:
        """Calculate prominence relative to surrounding variation"""
        try:
            h, w = elevation_patch.shape
            center_y, center_x = h // 2, w // 2
            radius = self.structure_radius_px
            
            # Surrounding area analysis with adaptive radius
            y, x = np.ogrid[:h, :w]
            surround_inner_radius = radius + 2
            surround_outer_radius = min(h, w) // 2 - 2
            surround_outer_radius = max(surround_outer_radius, int(radius * self.local_statistics_radius))
            
            surround_mask = (((y - center_y)**2 + (x - center_x)**2 <= surround_outer_radius**2) &
                           ((y - center_y)**2 + (x - center_x)**2 > surround_inner_radius**2))
            
            if np.any(surround_mask):
                surround_heights = elevation_patch[surround_mask]
                surround_std = np.std(surround_heights)
                relative_prominence = height_prominence / (surround_std + 0.1)
            else:
                relative_prominence = height_prominence
                
            return relative_prominence
        except:
            return height_prominence  # Fallback to absolute prominence
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute volume-based prominence score with adaptive normalization
        
        Args:
            elevation_patch: 2D elevation data array
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with volume-based confidence
        """
        try:
            h, w = elevation_patch.shape
            center_y, center_x = h // 2, w // 2
            radius = self.structure_radius_px
            patch_area = h * w * (self.resolution_m ** 2)
            
            # Create circular mask for structure
            y, x = np.ogrid[:h, :w]
            structure_mask = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
            
            # Calculate base elevation from edge areas
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
            
            # Calculate volume and prominence metrics
            if np.any(structure_mask):
                structure_heights = elevation_patch[structure_mask]
                structure_area = np.sum(structure_mask) * (self.resolution_m ** 2)
                
                # Volume above base
                volume_above_base = np.sum(np.maximum(0, structure_heights - base_elevation)) * (self.resolution_m ** 2)
                
                # Height prominence
                structure_max = np.max(structure_heights)
                structure_mean = np.mean(structure_heights)
                height_prominence = structure_max - base_elevation
                
                # Analyze structure characteristics for adaptive scaling
                structure_characteristics = self._analyze_structure_characteristics(elevation_patch, structure_mask)
                
                # Adaptive normalization
                volume_norm, prominence_norm = self._adaptive_normalization(
                    volume_above_base, height_prominence, structure_area, patch_area
                )
                
                # Apply structure-derived scaling
                volume_norm *= structure_characteristics["volume_scale"]
                prominence_norm *= structure_characteristics["prominence_scale"]
                
                # Relative assessment - compare to local context
                local_volume_percentile = self._calculate_local_percentile(elevation_patch, volume_above_base)
                local_prominence_percentile = self._calculate_local_percentile(elevation_patch, height_prominence)
                
                # Normalize scores with adaptive scaling
                volume_score = min(1.0, volume_above_base / volume_norm)
                prominence_score = min(1.0, height_prominence / prominence_norm)
                
                # Incorporate relative context
                volume_score = (1 - self.context_weight) * volume_score + self.context_weight * (local_volume_percentile / 100.0)
                prominence_score = (1 - self.context_weight) * prominence_score + self.context_weight * (local_prominence_percentile / 100.0)
                
                # Combined score with weighting
                combined_score = (0.6 * volume_score + 0.4 * prominence_score)
                
                # Calculate concentration metrics
                height_density = volume_above_base / (structure_area + 1e-6)
                
                # Bonus for concentrated elevation (vs. spread out)
                if height_density > 1.0:
                    combined_score *= self.concentration_bonus
                
                # Additional relative assessment
                relative_prominence = self._calculate_relative_prominence(
                    elevation_patch, structure_mask, height_prominence
                )
                
                # Incorporate relative prominence
                relative_prominence_score = min(1.0, relative_prominence / 3.0)  # Scale relative prominence
                combined_score = ((1 - self.relative_prominence_weight) * combined_score + 
                                self.relative_prominence_weight * relative_prominence_score)
                
                # Ensure final score is always bounded [0, 1]
                combined_score = np.clip(combined_score, 0.0, 1.0)
                
            else:
                combined_score = 0.0
                volume_above_base = 0.0
                height_prominence = 0.0
                structure_max = base_elevation
                structure_mean = base_elevation
                relative_prominence = 0.0
                height_density = 0.0
                structure_area = 0.0
                volume_score = 0.0
                prominence_score = 0.0
                local_volume_percentile = 0.0
                local_prominence_percentile = 0.0
                structure_characteristics = {"volume_scale": 1.0, "prominence_scale": 1.0, "complexity": 0.0}
                volume_norm = self.base_volume_normalization
                prominence_norm = self.base_prominence_normalization
            
            return FeatureResult(
                score=combined_score,
                polarity="neutral",
                metadata={
                    "normalized_volume": float(combined_score),
                    "volume_above_base": float(volume_above_base),
                    "height_prominence": float(height_prominence),
                    "relative_prominence": float(relative_prominence),
                    "structure_max_height": float(structure_max),
                    "structure_mean_height": float(structure_mean),
                    "base_elevation": float(base_elevation),
                    "structure_area": float(structure_area),
                    "height_density": float(height_density),
                    "structure_pixels": int(np.sum(structure_mask)),
                    "volume_score": float(volume_score),
                    "prominence_score": float(prominence_score),
                    "local_volume_percentile": float(local_volume_percentile),
                    "local_prominence_percentile": float(local_prominence_percentile),
                    "adaptive_volume_norm": float(volume_norm),
                    "adaptive_prominence_norm": float(prominence_norm),
                    "structure_complexity": float(structure_characteristics.get("complexity", 0.0)),
                    "volume_scale_factor": float(structure_characteristics.get("volume_scale", 1.0)),
                    "prominence_scale_factor": float(structure_characteristics.get("prominence_scale", 1.0))
                },
                reason=f"Adaptive volume analysis: vol={volume_above_base:.1f}, prom={height_prominence:.2f}m, complexity={structure_characteristics.get('complexity', 0.0):.2f}, local_ctx={local_volume_percentile:.0f}%"
            )
            
        except Exception as e:
            return FeatureResult(
                score=0.0,
                valid=False,
                reason=f"Volume computation failed: {str(e)}"
            )

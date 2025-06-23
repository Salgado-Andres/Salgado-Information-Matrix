"""
Compactness Feature Module

Validates structure compactness and circular geometry by analyzing
radial symmetry, aspect ratio, and shape regularity with explainable metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..base_module import BaseFeatureModule, FeatureResult, FeatureDocumentation, ParameterInfo, MetricInfo


class CompactnessModule(BaseFeatureModule):
    """
    Analyzes geometric compactness using explainable metrics:
    - Aspect ratio (circular vs elongated)
    - Circular symmetry (radial consistency)
    - Central dominance (single peak vs multiple)
    - Radial monotonicity (decreases from center)
    - Size appropriateness (reasonable structure size)
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for compactness analysis"""
        return {
            "n_angles": 36,
            "min_samples": 8,
            "symmetry_factor": 0.8,
            "noise_tolerance": 0.1,
            "min_radius": 3,
            "max_radius": 50
        }
    
    @classmethod
    def get_documentation(cls) -> FeatureDocumentation:
        """Return comprehensive documentation for the compactness module"""
        return FeatureDocumentation(
            module_name="Compactness Analysis",
            purpose="Distinguishes circular windmill mounds from linear ridges and irregular natural formations using geometric analysis",
            how_it_works="Analyzes 5 key geometric properties: aspect ratio (circularity), radial symmetry, central dominance (single peak), radial monotonicity (decreases from center), and size appropriateness. Each metric is weighted and combined to produce an explainable compactness score.",
            
            parameters=[
                ParameterInfo(
                    name="n_angles",
                    description="Number of angular samples for radial symmetry analysis",
                    data_type="int",
                    default_value=36,
                    valid_range="16-72",
                    example_values=[24, 36, 48],
                    impact="Higher values give more precise symmetry analysis but slower computation"
                ),
                ParameterInfo(
                    name="min_samples",
                    description="Minimum valid elevation points required for analysis",
                    data_type="int", 
                    default_value=8,
                    valid_range="5-20",
                    example_values=[8, 12, 16],
                    impact="Lower values allow analysis of smaller structures but may be less reliable"
                ),
                ParameterInfo(
                    name="min_radius",
                    description="Minimum structure radius in pixels for appropriate sizing",
                    data_type="int",
                    default_value=3,
                    valid_range="2-10",
                    example_values=[3, 5, 8],
                    impact="Structures smaller than this are penalized as too small"
                ),
                ParameterInfo(
                    name="max_radius", 
                    description="Maximum structure radius in pixels for appropriate sizing",
                    data_type="int",
                    default_value=50,
                    valid_range="20-100",
                    example_values=[30, 50, 80],
                    impact="Structures larger than this are penalized as too large"
                )
            ],
            
            output_metrics=[
                MetricInfo(
                    name="aspect_ratio",
                    description="Ratio of minor axis to major axis (circularity measure)",
                    range="0.0-1.0",
                    interpretation="1.0 = perfect circle, 0.0 = perfect line",
                    good_values="Windmills: >0.7, Linear ridges: <0.4",
                    example="Windmill: 0.85, Ridge: 0.15"
                ),
                MetricInfo(
                    name="circular_symmetry",
                    description="Consistency of elevation values at equal distances from center",
                    range="0.0-1.0", 
                    interpretation="1.0 = perfect radial symmetry, 0.0 = highly irregular",
                    good_values="Windmills: >0.6, Natural mounds: <0.5",
                    example="Windmill: 0.75, Natural: 0.35"
                ),
                MetricInfo(
                    name="central_dominance",
                    description="How much the center elevation exceeds the mean elevation",
                    range="0.0-10.0+",
                    interpretation="Higher values indicate prominent central peak",
                    good_values="Windmills: >2.0, Flat areas: <1.0",
                    example="Windmill: 3.5, Flat: 0.8"
                ),
                MetricInfo(
                    name="radial_monotonicity",
                    description="How consistently elevation decreases from center to edge",
                    range="0.0-1.0",
                    interpretation="1.0 = perfect radial decrease, 0.5 = random",
                    good_values="Windmills: >0.7, Irregular: <0.5",
                    example="Windmill: 0.82, Irregular: 0.35"
                ),
                MetricInfo(
                    name="effective_radius",
                    description="Estimated radius of the elevated structure in pixels",
                    range="1.0-100.0+",
                    interpretation="Radius of the structure based on elevated area",
                    good_values="Windmills: 5-25 pixels (2.5-12.5m at 0.5m resolution)",
                    example="Typical windmill: 8-15 pixels"
                ),
                MetricInfo(
                    name="peak_count",
                    description="Number of significant elevation peaks detected",
                    range="0-10+",
                    interpretation="Number of local maxima above threshold",
                    good_values="Windmills: 1 peak, Complex structures: 2+",
                    example="Simple windmill: 1, Settlement: 3"
                )
            ],
            
            interpretation_guide="The compactness score combines weighted geometric metrics to distinguish windmill mounds from other features. Key discriminators: aspect ratio (rejects linear ridges), central dominance (requires clear peak), and radial symmetry (rejects irregular natural formations). Scores >0.7 typically indicate windmill-like structures, while scores <0.4 suggest linear or irregular features.",
            
            typical_scores={
                "circular_windmill": "0.7-0.9",
                "linear_ridge": "0.1-0.4", 
                "natural_mound": "0.4-0.7",
                "flat_terrain": "0.0-0.3",
                "settlement": "0.5-0.8"
            }
        )
    
    def __init__(self, weight: float = 1.1):
        super().__init__("Compactness", weight)
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    @property
    def parameter_documentation(self) -> Dict[str, str]:
        """Documentation for all compactness analysis parameters"""
        return {
            "n_angles": "Number of angular samples for radial symmetry analysis (16-72, default: 36)",
            "min_samples": "Minimum valid elevation points required for reliable analysis (5-20, default: 8)",
            "symmetry_factor": "Weight factor for symmetry calculations in scoring (0.5-2.0, default: 0.8)",
            "noise_tolerance": "Tolerance level for noise in elevation data (0.0-0.5, default: 0.1)",
            "min_radius": "Minimum expected structure radius in pixels for size validation (1-10, default: 3)",
            "max_radius": "Maximum expected structure radius in pixels for size validation (20-100, default: 50)"
        }
    
    @property
    def result_documentation(self) -> Dict[str, str]:
        """Documentation for compactness analysis result metadata"""
        return {
            "aspect_ratio": "Ratio of minor to major axis (1.0=perfect circle, 0.0=line) - key discriminator",
            "circular_symmetry": "Radial consistency measure (0.0-1.0, >0.6 indicates good circular pattern)",
            "central_dominance": "Ratio of center height to mean height (>2.0 indicates strong central peak)",
            "radial_monotonicity": "How well elevation decreases from center (0.0-1.0, >0.7 indicates windmill-like)",
            "effective_radius": "Estimated structure radius in pixels based on elevated area",
            "peak_count": "Number of significant elevation peaks detected (1 is ideal for windmills)",
            "major_axis_length": "Length of the major principal axis in pixels",
            "minor_axis_length": "Length of the minor principal axis in pixels",
            "shape_classification": "Categorical shape assessment: 'circular' (aspect_ratio > 0.7) or 'elongated'",
            "size_score": "Appropriateness of structure size (0.0-1.0, >0.8 indicates good size)",
            "size_classification": "Categorical size assessment: 'appropriate' or 'unusual'",
            "compactness": "Overall compactness score (0.0-1.0, combines all geometric metrics)",
            "explanations": "List of human-readable explanations for the score",
            "radial_profile_length": "Number of valid points in radial symmetry analysis",
            "mean_elevation": "Average elevation value within the analysis patch",
            "elevation_std": "Standard deviation of elevation values"
        }
    
    @property
    def interpretation_guide(self) -> Dict[str, str]:
        """Guide for interpreting compactness analysis results"""
        return {
            "High Compactness (>0.7)": "Strong circular geometry with central peak - likely windmill mound",
            "Medium Compactness (0.4-0.7)": "Some circular features but not ideal - possible eroded or partial structure",
            "Low Compactness (<0.4)": "Poor circular geometry - likely linear ridge, natural mound, or flat terrain",
            "High Aspect Ratio (>0.8)": "Nearly circular shape - excellent windmill candidate",
            "Low Aspect Ratio (<0.4)": "Linear or elongated shape - ridge-like feature, not windmill",
            "High Circular Symmetry (>0.7)": "Good radial consistency around structure",
            "Low Circular Symmetry (<0.4)": "Irregular or asymmetric elevation pattern",
            "Strong Central Dominance (>3.0)": "Prominent central peak typical of constructed mounds",
            "Weak Central Dominance (<1.5)": "No clear central peak, likely natural or flat terrain",
            "Good Radial Monotonicity (>0.7)": "Elevation decreases smoothly from center outward",
            "Poor Radial Monotonicity (<0.4)": "Irregular elevation pattern, not mound-like",
            "Single Peak": "One dominant elevation peak (ideal for windmills)",
            "Multiple Peaks": "Several elevation peaks (suggests complex or natural formation)"
        }
    
    @property
    def feature_description(self) -> str:
        """Overall description of what the compactness feature analyzes"""
        return """
        Compactness Geometric Analysis:
        
        The compactness module distinguishes circular windmill mounds from linear ridges and irregular 
        natural formations using five key geometric metrics with explainable scoring.
        
        Key Capabilities:
        - Aspect Ratio Analysis: Distinguishes circular vs elongated shapes using PCA
        - Circular Symmetry: Measures radial consistency around structure perimeter  
        - Central Dominance: Detects single prominent peaks vs multiple or flat patterns
        - Radial Monotonicity: Validates mound-like elevation decrease from center
        - Size Appropriateness: Ensures structure size matches expected windmill dimensions
        
        Explainable Scoring:
        Each metric contributes specific weights to the final score with clear explanations.
        The system heavily penalizes linear features (aspect ratio < 0.7) while rewarding
        circular patterns with central peaks and appropriate size.
        
        Best For:
        - Distinguishing windmill mounds from linear ridges (primary use case)
        - Identifying well-preserved circular structures  
        - Filtering out natural formations and vegetation
        - Providing interpretable geometric feature analysis
        """
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute compactness score based on explainable metrics
        
        Args:
            elevation_patch: 2D elevation data array
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with compactness score and explainable metadata
        """
        try:
            rows, cols = elevation_patch.shape
            center_row, center_col = rows // 2, cols // 2
            
            # Basic elevation statistics
            values = elevation_patch.flatten()
            values = values[~np.isnan(values)]
            
            if len(values) < self.min_samples:
                return FeatureResult(
                    score=0.0,
                    polarity="neutral",
                    metadata={"error": "insufficient_data", "reason": "Not enough valid elevation points"},
                    reason="Insufficient data for compactness analysis"
                )
            
            mean_val = np.mean(values)
            std_dev = np.std(values)
            
            # 1. ASPECT RATIO ANALYSIS (most explainable discriminator)
            aspect_ratio, major_axis, minor_axis = self._calculate_aspect_ratio(elevation_patch, mean_val, std_dev)
            
            # 2. CIRCULAR SYMMETRY (radial consistency)
            circular_symmetry, radial_profile = self._calculate_circular_symmetry(elevation_patch, center_row, center_col)
            
            # 3. CENTRAL DOMINANCE (single peak vs multiple peaks)
            central_dominance, peak_count = self._calculate_central_dominance(elevation_patch, mean_val, std_dev)
            
            # 4. RADIAL MONOTONICITY (decreases from center)
            radial_monotonicity = self._calculate_radial_monotonicity(radial_profile)
            
            # 5. SIZE APPROPRIATENESS (reasonable structure size)
            effective_radius, size_score = self._calculate_size_metrics(elevation_patch, mean_val, std_dev)
            
            # COMPACTNESS SCORE CALCULATION with explainable weights
            compactness_score = 0.0
            explanations = []
            
            # Weight 1: Aspect ratio (0.3) - Linear features penalized
            if aspect_ratio > 0.7:  # Circular
                aspect_contribution = 0.3 * aspect_ratio
                explanations.append(f"Circular shape (aspect ratio: {aspect_ratio:.3f})")
            else:  # Linear
                aspect_contribution = 0.1 * aspect_ratio  # Heavy penalty
                explanations.append(f"Linear/elongated shape (aspect ratio: {aspect_ratio:.3f})")
            compactness_score += aspect_contribution
            
            # Weight 2: Circular symmetry (0.25)
            if circular_symmetry > 0.6:
                symmetry_contribution = 0.25 * circular_symmetry
                explanations.append(f"Good radial symmetry ({circular_symmetry:.3f})")
            else:
                symmetry_contribution = 0.1 * circular_symmetry
                explanations.append(f"Poor radial symmetry ({circular_symmetry:.3f})")
            compactness_score += symmetry_contribution
            
            # Weight 3: Central dominance (0.2) - Single peak preferred
            if central_dominance > 2.0 and peak_count == 1:
                dominance_contribution = 0.2
                explanations.append(f"Single central peak (dominance: {central_dominance:.1f})")
            elif peak_count == 1:
                dominance_contribution = 0.15
                explanations.append(f"Central peak present (dominance: {central_dominance:.1f})")
            else:
                dominance_contribution = 0.05
                explanations.append(f"Multiple peaks detected ({peak_count} peaks)")
            compactness_score += dominance_contribution
            
            # Weight 4: Radial monotonicity (0.15)
            if radial_monotonicity > 0.7:
                monotonic_contribution = 0.15
                explanations.append(f"Decreases from center ({radial_monotonicity:.3f})")
            else:
                monotonic_contribution = 0.05
                explanations.append(f"Irregular radial pattern ({radial_monotonicity:.3f})")
            compactness_score += monotonic_contribution
            
            # Weight 5: Size appropriateness (0.1)
            compactness_score += 0.1 * size_score
            if size_score > 0.8:
                explanations.append(f"Appropriate size (radius: {effective_radius:.1f})")
            else:
                explanations.append(f"Unusual size (radius: {effective_radius:.1f})")
            
            # Final compactness score
            final_score = min(1.0, max(0.0, compactness_score))
            
            return FeatureResult(
                score=final_score,
                polarity="neutral",
                metadata={
                    # Explainable primary metrics
                    "aspect_ratio": float(aspect_ratio),
                    "circular_symmetry": float(circular_symmetry), 
                    "central_dominance": float(central_dominance),
                    "radial_monotonicity": float(radial_monotonicity),
                    "effective_radius": float(effective_radius),
                    "peak_count": int(peak_count),
                    
                    # Shape characteristics  
                    "major_axis_length": float(major_axis),
                    "minor_axis_length": float(minor_axis),
                    "shape_classification": "circular" if aspect_ratio > 0.7 else "elongated",
                    
                    # Size characteristics
                    "size_score": float(size_score),
                    "size_classification": "appropriate" if size_score > 0.8 else "unusual",
                    
                    # Overall assessment
                    "compactness": float(final_score),
                    "explanations": explanations,
                    
                    # Technical details
                    "radial_profile_length": len(radial_profile),
                    "mean_elevation": float(mean_val),
                    "elevation_std": float(std_dev)
                },
                reason=f"Compactness: {final_score:.3f} (" + ", ".join(explanations[:2]) + ")"
            )
            
        except Exception as e:
            return FeatureResult(
                score=0.0,
                polarity="neutral", 
                metadata={"error": str(e)},
                reason=f"Compactness computation failed: {str(e)}"
            )
    
    def _calculate_aspect_ratio(self, elevation_patch: np.ndarray, mean_val: float, std_dev: float) -> Tuple[float, float, float]:
        """
        Calculate aspect ratio (circular vs elongated) using elevation contours
        
        Returns:
            aspect_ratio: ratio of minor to major axis (1.0 = perfect circle, 0.0 = line)
            major_axis: length of major axis
            minor_axis: length of minor axis
        """
        try:
            # Create binary mask for elevated regions
            threshold = mean_val + 0.5 * std_dev
            mask = elevation_patch > threshold
            
            if np.sum(mask) < 5:  # Too few points
                return 0.1, 1.0, 0.1  # Linear assumption
            
            # Find coordinates of elevated points
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) < 3:
                return 0.1, 1.0, 0.1
            
            # Calculate covariance matrix for PCA
            coords = np.column_stack([x_coords, y_coords])
            coords_centered = coords - np.mean(coords, axis=0)
            cov_matrix = np.cov(coords_centered.T)
            
            # Eigenvalues give the variance along principal axes
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            major_axis = 2 * np.sqrt(eigenvalues[0])  # 2 * std dev
            minor_axis = 2 * np.sqrt(eigenvalues[1]) if len(eigenvalues) > 1 else major_axis
            
            # Aspect ratio (how circular vs elongated)
            aspect_ratio = minor_axis / (major_axis + 1e-6)
            
            return aspect_ratio, major_axis, minor_axis
            
        except Exception:
            return 0.1, 1.0, 0.1  # Default to linear
    
    def _calculate_circular_symmetry(self, elevation_patch: np.ndarray, center_row: int, center_col: int) -> Tuple[float, np.ndarray]:
        """
        Calculate circular symmetry by analyzing radial consistency
        
        Returns:
            circular_symmetry: score from 0-1 (1 = perfect radial symmetry)
            radial_profile: elevation values at different angles
        """
        try:
            rows, cols = elevation_patch.shape
            max_radius = min(center_row, center_col, rows - center_row, cols - center_col) - 1
            
            # Sample at multiple radii for robustness
            radii = np.linspace(2, max_radius, min(5, max_radius-1)) if max_radius > 2 else [2]
            
            all_profiles = []
            
            for radius in radii:
                angles = np.linspace(0, 2*np.pi, self.n_angles, endpoint=False)
                profile = []
                
                for angle in angles:
                    y = int(center_row + radius * np.sin(angle))
                    x = int(center_col + radius * np.cos(angle))
                    
                    if 0 <= y < rows and 0 <= x < cols:
                        profile.append(elevation_patch[y, x])
                    else:
                        profile.append(np.nan)
                
                # Remove NaN values
                profile = np.array(profile)
                profile = profile[~np.isnan(profile)]
                
                if len(profile) >= self.min_samples:
                    all_profiles.extend(profile)
            
            if len(all_profiles) < self.min_samples:
                return 0.0, np.array([])
            
            profile_array = np.array(all_profiles)
            
            # Calculate symmetry as inverse of coefficient of variation
            mean_val = np.mean(profile_array)
            std_val = np.std(profile_array)
            cv = std_val / (abs(mean_val) + 1e-6)
            
            # Convert to symmetry score (lower CV = higher symmetry)
            symmetry = 1.0 / (1.0 + 2.0 * cv)
            
            return symmetry, profile_array
            
        except Exception:
            return 0.0, np.array([])
    
    def _calculate_central_dominance(self, elevation_patch: np.ndarray, mean_val: float, std_dev: float) -> Tuple[float, int]:
        """
        Calculate central dominance (single peak vs multiple peaks)
        
        Returns:
            central_dominance: ratio of center height to mean height
            peak_count: number of significant peaks detected
        """
        try:
            rows, cols = elevation_patch.shape
            center_row, center_col = rows // 2, cols // 2
            
            # Central elevation (average of 3x3 center region)
            center_region = elevation_patch[max(0, center_row-1):center_row+2, 
                                         max(0, center_col-1):center_col+2]
            center_elevation = np.nanmean(center_region)
            
            # Central dominance ratio
            dominance = center_elevation / (mean_val + 1e-6)
            
            # Count peaks using local maxima
            from scipy import ndimage
            
            # Smooth slightly to avoid noise peaks
            smoothed = ndimage.gaussian_filter(elevation_patch, sigma=0.8)
            
            # Find local maxima
            threshold = mean_val + 0.3 * std_dev
            maxima_mask = ndimage.maximum_filter(smoothed, size=3) == smoothed
            significant_peaks = maxima_mask & (smoothed > threshold)
            
            peak_count = np.sum(significant_peaks)
            
            return dominance, peak_count
            
        except Exception:
            return 1.0, 1  # Default values
    
    def _calculate_radial_monotonicity(self, radial_profile: np.ndarray) -> float:
        """
        Calculate radial monotonicity (how much it decreases from center)
        
        Args:
            radial_profile: elevation values at different radii
            
        Returns:
            monotonicity: score from 0-1 (1 = perfectly decreases from center)
        """
        try:
            if len(radial_profile) < 3:
                return 0.5  # Neutral if insufficient data
            
            # For true monotonicity, we need center and edge samples
            # This is a simplified version using the profile variance
            
            # Calculate how much the profile follows a decreasing trend
            # Higher values at start = better monotonicity
            n = len(radial_profile)
            if n < 4:
                return 0.5
            
            # Split into inner and outer parts
            inner_part = radial_profile[:n//2]
            outer_part = radial_profile[n//2:]
            
            inner_mean = np.mean(inner_part)
            outer_mean = np.mean(outer_part)
            
            # Monotonicity = how much inner exceeds outer
            if inner_mean + outer_mean == 0:
                return 0.5
            
            monotonicity = (inner_mean - outer_mean) / (inner_mean + outer_mean + 1e-6)
            monotonicity = max(0.0, min(1.0, monotonicity + 0.5))  # Normalize to 0-1
            
            return monotonicity
            
        except Exception:
            return 0.5
    
    def _calculate_size_metrics(self, elevation_patch: np.ndarray, mean_val: float, std_dev: float) -> Tuple[float, float]:
        """
        Calculate size appropriateness for windmill structures
        
        Returns:
            effective_radius: estimated structure radius in pixels
            size_score: score from 0-1 (1 = appropriate size)
        """
        try:
            rows, cols = elevation_patch.shape
            
            # Find elevated region
            threshold = mean_val + 0.3 * std_dev
            mask = elevation_patch > threshold
            
            if np.sum(mask) < 3:
                return 1.0, 0.1  # Very small
            
            # Calculate effective radius from area
            area = np.sum(mask)
            effective_radius = np.sqrt(area / np.pi)
            
            # Size appropriateness (windmills should be medium-sized)
            # Appropriate radius range: 3-25 pixels
            if self.min_radius <= effective_radius <= self.max_radius:
                size_score = 1.0
            elif effective_radius < self.min_radius:
                size_score = effective_radius / self.min_radius
            else:  # too large
                size_score = self.max_radius / effective_radius
            
            size_score = max(0.1, min(1.0, size_score))
            
            return effective_radius, size_score
            
        except Exception:
            return 1.0, 0.5
    
    def configure(self, 
                 n_angles: int = None,
                 min_samples: int = None,
                 symmetry_factor: float = None,
                 min_radius: int = None,
                 max_radius: int = None):
        """Configure module parameters"""
        if n_angles is not None:
            self.n_angles = n_angles
        if min_samples is not None:
            self.min_samples = min_samples
        if symmetry_factor is not None:
            self.symmetry_factor = symmetry_factor
        if min_radius is not None:
            self.min_radius = min_radius
        if max_radius is not None:
            self.max_radius = max_radius

"""
Planarity Feature Module

Validates local planarity and surface regularity using least squares
plane fitting and residual analysis.
"""

import numpy as np
from typing import Dict, Any

from ..base_module import BaseFeatureModule, FeatureResult


class PlanarityModule(BaseFeatureModule):
    """
    Analyzes surface planarity within the structure region.
    
    Structures typically have more regular surfaces than natural terrain,
    which can be detected through plane fitting and residual analysis.
    """
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Return default parameters for planarity analysis"""
        return {
            "plane_method": "least_squares",
            "outlier_threshold": 2.0,
            "adaptive_fitting": True,
            "robust_estimation": True,
            "local_plane_analysis": True,
            "min_points": 10,
            "planarity_factor": 1.0,
            "residual_analysis": True,
            "surface_roughness": True
        }
    
    def __init__(self, weight: float = 0.9):
        super().__init__("Planarity", weight)
        
        # Initialize with default parameters
        defaults = self.get_default_parameters()
        for param, value in defaults.items():
            setattr(self, param, value)
    
    @property
    def parameter_documentation(self) -> Dict[str, str]:
        """Documentation for all planarity analysis parameters"""
        return {
            "plane_method": "Method for plane fitting: 'least_squares' for standard fitting, 'ransac' for robust fitting",
            "outlier_threshold": "Threshold (in standard deviations) for identifying elevation outliers",
            "adaptive_fitting": "Whether to adapt fitting method based on data characteristics",
            "robust_estimation": "Whether to use robust estimation methods that resist outlier influence",
            "local_plane_analysis": "Whether to analyze planarity in local sub-regions as well as globally",
            "min_points": "Minimum number of valid elevation points required for reliable plane fitting",
            "planarity_factor": "Scaling factor for planarity score interpretation (higher = stricter planarity requirement)",
            "residual_analysis": "Whether to perform detailed analysis of plane fitting residuals",
            "surface_roughness": "Whether to calculate additional surface roughness metrics"
        }
    
    @property
    def result_documentation(self) -> Dict[str, str]:
        """Documentation for planarity analysis result metadata"""
        return {
            "planarity_score": "Overall planarity score (higher = more planar/flat surface)",
            "plane_fit_quality": "Quality of the best-fit plane (R-squared value)",
            "residual_std": "Standard deviation of residuals from the fitted plane",
            "residual_mean": "Mean of residuals from the fitted plane (should be near zero)",
            "plane_normal": "Normal vector of the fitted plane [x, y, z components]",
            "plane_slope": "Slope angle of the fitted plane in degrees",
            "plane_intercept": "Z-intercept of the fitted plane at patch center",
            "outlier_count": "Number of elevation points identified as outliers",
            "outlier_fraction": "Fraction of points that are outliers (0.0-1.0)",
            "surface_roughness_rms": "Root mean square surface roughness",
            "local_planarity_scores": "Planarity scores for sub-regions (if local_plane_analysis=True)",
            "fitting_method_used": "Actual plane fitting method used (may differ from input if adaptive)",
            "goodness_of_fit": "Overall goodness of fit measure combining multiple quality metrics",
            "elevation_range": "Range of elevation values within the analysis region",
            "planar_trend": "Primary trend direction of the fitted plane"
        }
    
    @property
    def interpretation_guide(self) -> Dict[str, str]:
        """Guide for interpreting planarity analysis results"""
        return {
            "High Planarity (>0.8)": "Very flat surface - likely artificial platform or well-preserved structure top",
            "Good Planarity (0.6-0.8)": "Reasonably flat surface - possible structure platform or natural terrace",
            "Moderate Planarity (0.4-0.6)": "Somewhat irregular surface - eroded structure or natural slope",
            "Low Planarity (<0.4)": "Irregular surface - likely natural terrain, vegetation, or heavily damaged structure",
            "Low Residual Std": "Elevation points closely follow fitted plane - high surface regularity",
            "High Residual Std": "Elevation points deviate significantly from plane - irregular surface",
            "Few Outliers": "Most elevation points fit the planar model - consistent surface",
            "Many Outliers": "Many points don't fit planar model - complex or damaged surface",
            "Low Slope": "Nearly horizontal surface typical of platforms or level construction",
            "High Slope": "Steep surface typical of natural slopes or structure sides",
            "Good Fit Quality": "Plane model explains most elevation variation - truly planar surface"
        }
    
    @property
    def feature_description(self) -> str:
        """Overall description of what the planarity feature analyzes"""
        return """
        Surface Planarity Analysis:
        
        The planarity module fits mathematical planes to elevation surfaces to quantify how flat
        and regular they are. This helps distinguish artificial platforms and construction surfaces
        from irregular natural terrain and vegetation-covered areas.
        
        Key Capabilities:
        - Least squares and robust (RANSAC) plane fitting methods
        - Outlier detection and resistant estimation
        - Local planarity analysis in sub-regions
        - Surface roughness quantification
        - Slope and orientation analysis of fitted planes
        
        Best For:
        - Detecting artificial platforms and level surfaces
        - Identifying construction floors and paved areas
        - Distinguishing preserved structures from eroded features
        - Analyzing surface regularity and construction quality
        """
    
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute planarity score using least squares fitting
        
        Args:
            elevation_patch: 2D elevation data array
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with planarity confidence
        """
        try:
            h, w = elevation_patch.shape
            center_y, center_x = h // 2, w // 2
            radius = self.structure_radius_px
            
            # Extract local patch around center
            y_start = max(0, center_y - radius)
            y_end = min(h, center_y + radius + 1)
            x_start = max(0, center_x - radius)
            x_end = min(w, center_x + radius + 1)
            
            local_patch = elevation_patch[y_start:y_end, x_start:x_end]
            yy, xx = np.mgrid[:local_patch.shape[0], :local_patch.shape[1]]
            
            # Create circular mask
            local_center_y = local_patch.shape[0] // 2
            local_center_x = local_patch.shape[1] // 2
            mask = (yy - local_center_y)**2 + (xx - local_center_x)**2 <= radius**2
            
            if np.sum(mask) < self.min_points:
                return FeatureResult(
                    score=0.0,
                    valid=False,
                    reason=f"Insufficient points for plane fitting: {np.sum(mask)} < {self.min_points}"
                )
            
            # Fit plane using least squares
            points = np.column_stack([xx[mask], yy[mask], np.ones(np.sum(mask))])
            z_values = local_patch[mask]
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(points, z_values, rcond=None)
                
                # Calculate fitted plane
                z_fit = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
                fit_residuals = np.abs(local_patch - z_fit)[mask]
                
                # Planarity metrics
                rmse = np.sqrt(np.mean(fit_residuals**2))
                residual_std = np.std(fit_residuals)
                max_residual = np.max(fit_residuals)
                
                # Planarity score (lower residuals = higher planarity)
                planarity_score = self.planarity_factor / (self.planarity_factor + residual_std)
                
                # Additional surface regularity metrics
                surface_variation = np.std(z_values)
                slope_magnitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                
                # Check for systematic patterns in residuals
                residual_range = np.max(fit_residuals) - np.min(fit_residuals)
                relative_rmse = rmse / (surface_variation + 1e-6)
                
                # Penalty for highly sloped surfaces (might be natural slopes)
                if slope_magnitude > 0.5:  # Steep slope
                    planarity_score *= 0.8
                
                return FeatureResult(
                    score=planarity_score,
                    polarity="neutral",
                    metadata={
                        "planarity": float(planarity_score),
                        "rmse": float(rmse),
                        "residual_std": float(residual_std),
                        "max_residual": float(max_residual),
                        "residual_range": float(residual_range),
                        "relative_rmse": float(relative_rmse),
                        "surface_variation": float(surface_variation),
                        "slope_magnitude": float(slope_magnitude),
                        "plane_coeffs": [float(c) for c in coeffs],
                        "mask_pixels": int(np.sum(mask)),
                        "patch_size": local_patch.shape,
                        "radius_used": int(radius)
                    },
                    reason=f"Planarity: score={planarity_score:.3f}, rmse={rmse:.3f}"
                )
                
            except np.linalg.LinAlgError:
                return FeatureResult(
                    score=0.0,
                    valid=False,
                    reason="Singular matrix in plane fitting (degenerate surface)"
                )
            
        except Exception as e:
            return FeatureResult(
                score=0.0,
                valid=False,
                reason=f"Planarity computation failed: {str(e)}"
            )
    
    def configure(self, 
                 min_points: int = None,
                 planarity_factor: float = None):
        """Configure module parameters"""
        if min_points is not None:
            self.min_points = min_points
        if planarity_factor is not None:
            self.planarity_factor = planarity_factor

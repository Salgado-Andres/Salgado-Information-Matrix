"""
Detector Profile System for G₂ Kernel Configuration

This module provides configurable detector profiles that can be optimized
for different archaeological structures, terrain types, and detection scenarios.
Profiles define feature selection, weights, thresholds, and geometric parameters.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PatchShape(Enum):
    """Supported patch shapes for detection"""
    SQUARE = "square"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    IRREGULAR = "irregular"  # For custom mask-based shapes


class StructureType(Enum):
    """Archaeological structure types with different characteristics"""
    WINDMILL = "windmill"
    SETTLEMENT = "settlement"
    EARTHWORK = "earthwork"
    PLATFORM = "platform"
    GEOGLYPH = "geoglyph"
    ROAD = "road"
    CANAL = "canal"
    GENERIC = "generic"


@dataclass
class GeometricParameters:
    """Geometric constraints and expectations for detection"""
    resolution_m: float = 0.5  # Resolution in meters per pixel
    structure_radius_m: float = 8.0  # Expected structure radius in meters
    min_structure_size_m: float = 3.0  # Minimum detectable structure size
    max_structure_size_m: float = 50.0  # Maximum expected structure size
    patch_shape: PatchShape = PatchShape.SQUARE
    patch_size_m: Tuple[float, float] = (20.0, 20.0)  # (width, height) in meters
    aspect_ratio_tolerance: float = 0.3  # Tolerance for non-square patches
    
    def get_patch_size_px(self) -> Tuple[int, int]:
        """Convert patch size from meters to pixels"""
        width_px = int(self.patch_size_m[0] / self.resolution_m)
        height_px = int(self.patch_size_m[1] / self.resolution_m)
        return (width_px, height_px)
    
    def get_structure_radius_px(self) -> int:
        """Convert structure radius from meters to pixels"""
        return int(self.structure_radius_m / self.resolution_m)


@dataclass
class FeatureConfiguration:
    """Configuration for a specific feature module"""
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    polarity_preference: Optional[str] = None  # "positive", "negative", or None for dynamic
    confidence_threshold: float = 0.0  # Minimum confidence to include this feature
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.weight < 0:
            raise ValueError("Feature weight must be non-negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")


def create_default_feature_configurations() -> Dict[str, FeatureConfiguration]:
    """Create default feature configurations by querying modules for their parameters"""
    from .modules.features.histogram_module import ElevationHistogramModule
    from .modules.features.volume_module import VolumeModule
    from .modules.features.compactness_module import CompactnessModule
    from .modules.features.dropoff_module import DropoffSharpnessModule
    from .modules.features.entropy_module import ElevationEntropyModule
    from .modules.features.planarity_module import PlanarityModule
    
    # Map feature names to module classes
    feature_modules = {
        "histogram": (ElevationHistogramModule, 1.5),
        "volume": (VolumeModule, 1.3),
        "dropoff": (DropoffSharpnessModule, 1.2),
        "compactness": (CompactnessModule, 1.1),
        "entropy": (ElevationEntropyModule, 1.0),
        "planarity": (PlanarityModule, 0.9)
    }
    
    configurations = {}
    for feature_name, (module_class, default_weight) in feature_modules.items():
        try:
            # Get default parameters from the module itself
            default_params = module_class.get_default_parameters()
            configurations[feature_name] = FeatureConfiguration(
                enabled=True,
                weight=default_weight,
                parameters=default_params
            )
        except Exception as e:
            logger.warning(f"Could not get default parameters for {feature_name}: {e}")
            # Fallback to minimal configuration
            configurations[feature_name] = FeatureConfiguration(
                enabled=True,
                weight=default_weight,
                parameters={}
            )
    
    return configurations


@dataclass
class DetectionThresholds:
    """Thresholds for detection decision-making"""
    detection_threshold: float = 0.5  # Final score threshold for positive detection
    confidence_threshold: float = 0.6  # Minimum confidence for reliable detection
    early_decision_threshold: float = 0.85  # Threshold for early termination
    min_modules_for_decision: int = 2  # Minimum modules before early decision
    max_modules_for_efficiency: int = 6  # Maximum modules to run for efficiency
    uncertainty_tolerance: float = 0.2  # Tolerance for conflicting evidence


@dataclass
class DetectorProfile:
    """
    Complete detector profile defining all parameters for G₂ detection
    
    This profile can be saved, loaded, and optimized for specific use cases.
    It encapsulates geometry, features, thresholds, and decision logic.
    """
    name: str
    description: str = ""
    structure_type: StructureType = StructureType.GENERIC
    version: str = "1.0"
    created_by: str = "G₂ System"
    
    # Core configuration components
    geometry: GeometricParameters = field(default_factory=GeometricParameters)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    
    # Feature module configurations
    features: Dict[str, FeatureConfiguration] = field(default_factory=create_default_feature_configurations)
    
    # Advanced configuration
    aggregation_method: str = "streaming"  # "streaming" or "batch"
    parallel_execution: bool = True
    max_workers: int = 5
    enable_refinement: bool = True
    max_refinement_attempts: int = 2
    
    # Metadata for optimization and tracking
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_used: Optional[str] = None
    use_count: int = 0
    
    def get_enabled_features(self) -> Dict[str, FeatureConfiguration]:
        """Get only the enabled feature configurations"""
        return {name: config for name, config in self.features.items() if config.enabled}
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get weights for enabled features"""
        return {name: config.weight for name, config in self.features.items() if config.enabled}
    
    def get_total_feature_weight(self) -> float:
        """Calculate total weight of enabled features"""
        return sum(config.weight for config in self.features.values() if config.enabled)
    
    def normalize_weights(self) -> None:
        """Normalize feature weights to sum to total enabled modules"""
        enabled_features = self.get_enabled_features()
        if not enabled_features:
            return
        
        total_weight = sum(config.weight for config in enabled_features.values())
        target_sum = len(enabled_features)
        
        for config in enabled_features.values():
            config.weight = (config.weight / total_weight) * target_sum
    
    def validate(self) -> List[str]:
        """Validate profile configuration and return list of issues"""
        issues = []
        
        # Validate geometry
        if self.geometry.resolution_m <= 0:
            issues.append("Resolution must be positive")
        
        if self.geometry.structure_radius_m <= 0:
            issues.append("Structure radius must be positive")
        
        if self.geometry.min_structure_size_m >= self.geometry.max_structure_size_m:
            issues.append("Minimum structure size must be less than maximum")
        
        # Validate patch size
        if any(size <= 0 for size in self.geometry.patch_size_m):
            issues.append("Patch dimensions must be positive")
        
        # Validate thresholds
        for threshold_name, threshold_value in asdict(self.thresholds).items():
            if isinstance(threshold_value, float) and not 0 <= threshold_value <= 1:
                if "threshold" in threshold_name:
                    issues.append(f"{threshold_name} must be between 0 and 1")
        
        # Validate features
        enabled_count = len(self.get_enabled_features())
        if enabled_count == 0:
            issues.append("At least one feature must be enabled")
        
        # Check for reasonable weights
        for name, config in self.features.items():
            if config.enabled and config.weight <= 0:
                issues.append(f"Feature '{name}' has non-positive weight")
        
        return issues
    
    def optimize_for_structure_type(self) -> None:
        """Optimize profile parameters for the specified structure type"""
        optimizations = {
            StructureType.WINDMILL: {
                "geometry.structure_radius_m": 8.0,
                "features.histogram.weight": 1.5,
                "features.compactness.weight": 1.3,
                "features.dropoff.weight": 1.2,
                "features.volume.parameters.base_volume_normalization": 30.0,
                "features.volume.parameters.base_prominence_normalization": 3.0,
                "thresholds.detection_threshold": 0.55
            },
            StructureType.SETTLEMENT: {
                "geometry.structure_radius_m": 15.0,
                "features.volume.weight": 1.4,
                "features.entropy.weight": 1.2,
                "features.planarity.weight": 1.1,
                "features.volume.parameters.base_volume_normalization": 80.0,
                "features.volume.parameters.base_prominence_normalization": 4.0,
                "features.volume.parameters.local_context_weight": 0.4,
                "thresholds.detection_threshold": 0.45
            },
            StructureType.EARTHWORK: {
                "geometry.structure_radius_m": 12.0,
                "features.volume.weight": 1.5,
                "features.dropoff.weight": 1.3,
                "features.planarity.weight": 0.7,
                "features.volume.parameters.base_volume_normalization": 120.0,
                "features.volume.parameters.base_prominence_normalization": 8.0,
                "features.volume.parameters.local_context_weight": 0.2,
                "thresholds.detection_threshold": 0.5
            },
            StructureType.GEOGLYPH: {
                "geometry.structure_radius_m": 25.0,
                "features.compactness.weight": 1.4,
                "features.entropy.weight": 0.8,
                "features.planarity.weight": 1.2,
                "features.volume.parameters.base_volume_normalization": 200.0,
                "features.volume.parameters.base_prominence_normalization": 2.0,
                "features.volume.parameters.local_context_weight": 0.5,
                "thresholds.detection_threshold": 0.4
            }
        }
        
        if self.structure_type in optimizations:
            opts = optimizations[self.structure_type]
            for path, value in opts.items():
                self._set_nested_attribute(path, value)
            
            logger.info(f"Optimized profile for {self.structure_type.value}")
    
    def _set_nested_attribute(self, path: str, value: Any) -> None:
        """Set a nested attribute using dot notation (e.g., 'geometry.resolution_m')"""
        parts = path.split('.')
        obj = self
        
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        
        final_attr = parts[-1]
        if isinstance(obj, dict):
            obj[final_attr] = value
        else:
            setattr(obj, final_attr, value)
    
    def create_volume_optimized_profile(self, structure_type: StructureType) -> 'DetectorProfile':
        """Create a new profile optimized for volume detection with adaptive parameters"""
        volume_optimizations = {
            StructureType.WINDMILL: {
                "name": "Volume-Optimized Windmill",
                "description": "Windmill detection with adaptive volume analysis",
                "volume_params": {
                    "base_volume_normalization": 30.0,
                    "base_prominence_normalization": 3.0,
                    "adaptive_scaling": True,
                    "size_scaling_factor": 1.2,
                    "context_weight": 0.25,
                    "concentration_bonus": 1.15,
                    "min_volume_threshold": 3.0,
                    "max_volume_saturation": 500.0
                }
            },
            StructureType.SETTLEMENT: {
                "name": "Volume-Optimized Settlement",
                "description": "Settlement detection with multi-scale adaptive volume analysis",
                "volume_params": {
                    "base_volume_normalization": 80.0,
                    "base_prominence_normalization": 4.0,
                    "adaptive_scaling": True,
                    "size_scaling_factor": 1.0,
                    "context_weight": 0.4,
                    "concentration_bonus": 1.05,
                    "relative_prominence_weight": 0.5,
                    "min_volume_threshold": 8.0,
                    "max_volume_saturation": 1500.0
                }
            },
            StructureType.EARTHWORK: {
                "name": "Volume-Optimized Earthwork",
                "description": "Earthwork detection with prominence-focused adaptive volume analysis",
                "volume_params": {
                    "base_volume_normalization": 120.0,
                    "base_prominence_normalization": 8.0,
                    "adaptive_scaling": True,
                    "size_scaling_factor": 0.8,
                    "context_weight": 0.2,
                    "concentration_bonus": 1.2,
                    "relative_prominence_weight": 0.6,
                    "min_prominence_threshold": 1.0,
                    "max_prominence_saturation": 30.0
                }
            }
        }
        
        if structure_type not in volume_optimizations:
            # Use generic optimization
            new_profile = DetectorProfile(
                name=f"Volume-Optimized {structure_type.value.title()}",
                description=f"Generic volume-optimized detection for {structure_type.value}",
                structure_type=structure_type
            )
        else:
            opt = volume_optimizations[structure_type]
            new_profile = DetectorProfile(
                name=opt["name"],
                description=opt["description"],
                structure_type=structure_type
            )
            
            # Update volume parameters
            for param, value in opt["volume_params"].items():
                new_profile.features["volume"].parameters[param] = value
        
        # Optimize for structure type
        new_profile.optimize_for_structure_type()
        
        return new_profile
    
    @classmethod
    def create_adaptive_volume_profile(cls, structure_type: StructureType = StructureType.GENERIC) -> 'DetectorProfile':
        """Create a profile specifically optimized for adaptive volume detection"""
        volume_overrides = {
            "volume": {
                "weight": 2.0,  # Increase volume weight
                "adaptive_scaling": True,
                "size_scaling_factor": 1.1,
                "area_scaling_reference": 150.0,
                "context_weight": 0.35,
                "concentration_bonus": 1.2,
                "relative_prominence_weight": 0.45,
                "auto_range_adaptation": True,
                "percentile_normalization": True,
                "local_statistics_radius": 2.5,
                "min_volume_threshold": 3.0,
                "max_volume_saturation": 800.0
            },
            "histogram": {"weight": 1.2},
            "dropoff": {"weight": 1.0},
            "compactness": {"weight": 0.8},
            "entropy": {"weight": 0.6},
            "planarity": {"weight": 0.5}
        }
        
        profile = cls.create_specialized_profile(
            name="Adaptive Volume Detection",
            description="Profile optimized for adaptive volume-based structure detection",
            structure_type=structure_type,
            feature_overrides=volume_overrides
        )
        
        # Lower detection threshold since volume is primary indicator
        profile.thresholds.detection_threshold = 0.45
        profile.thresholds.confidence_threshold = 0.55
        
        return profile
    
    def create_additional_adaptive_template_profiles(self) -> None:
        """Create additional template profiles showcasing adaptive volume capabilities"""
        adaptive_templates = [
            DetectorProfile(
                name="Ultra-Adaptive Generic",
                description="Fully adaptive profile that learns from any structure type",
                structure_type=StructureType.GENERIC,
                features={
                    "volume": FeatureConfiguration(enabled=True, weight=1.8,
                                                 parameters={
                                                     "adaptive_scaling": True,
                                                     "size_scaling_factor": 1.0,
                                                     "area_scaling_reference": 100.0,
                                                     "context_weight": 0.4,
                                                     "concentration_bonus": 1.15,
                                                     "relative_prominence_weight": 0.5,
                                                     "auto_range_adaptation": True,
                                                     "percentile_normalization": True,
                                                     "local_statistics_radius": 2.0,
                                                     "min_volume_threshold": 1.0,
                                                     "max_volume_saturation": 2000.0,
                                                     "min_prominence_threshold": 0.1,
                                                     "max_prominence_saturation": 50.0
                                                 }),
                    "histogram": FeatureConfiguration(enabled=True, weight=1.5),
                    "dropoff": FeatureConfiguration(enabled=True, weight=1.2),
                    "compactness": FeatureConfiguration(enabled=True, weight=1.0),
                    "entropy": FeatureConfiguration(enabled=True, weight=0.8),
                    "planarity": FeatureConfiguration(enabled=True, weight=0.6)
                }
            ),
            DetectorProfile(
                name="Micro-Structure Sensitive",
                description="Optimized for small, subtle structures with high sensitivity",
                structure_type=StructureType.GENERIC,
                features={
                    "volume": FeatureConfiguration(enabled=True, weight=2.2,
                                                 parameters={
                                                     "adaptive_scaling": True,
                                                     "size_scaling_factor": 1.5,
                                                     "area_scaling_reference": 25.0,  # Small reference area
                                                     "context_weight": 0.5,  # High context sensitivity
                                                     "concentration_bonus": 1.3,
                                                     "relative_prominence_weight": 0.6,
                                                     "min_volume_threshold": 0.5,  # Very low threshold
                                                     "max_volume_saturation": 200.0,
                                                     "min_prominence_threshold": 0.05,
                                                     "max_prominence_saturation": 5.0
                                                 })
                },
                thresholds=DetectionThresholds(
                    detection_threshold=0.35,  # Lower threshold for sensitivity
                    confidence_threshold=0.45
                )
            ),
            DetectorProfile(
                name="Macro-Structure Robust",
                description="Optimized for large, prominent structures with noise robustness",
                structure_type=StructureType.GENERIC,
                features={
                    "volume": FeatureConfiguration(enabled=True, weight=1.6,
                                                 parameters={
                                                     "adaptive_scaling": True,
                                                     "size_scaling_factor": 0.8,
                                                     "area_scaling_reference": 500.0,  # Large reference area
                                                     "context_weight": 0.2,  # Lower context sensitivity
                                                     "concentration_bonus": 1.1,
                                                     "relative_prominence_weight": 0.3,
                                                     "min_volume_threshold": 20.0,  # Higher threshold
                                                     "max_volume_saturation": 5000.0,
                                                     "min_prominence_threshold": 2.0,
                                                     "max_prominence_saturation": 100.0
                                                 })
                },
                thresholds=DetectionThresholds(
                    detection_threshold=0.6,  # Higher threshold for robustness
                    confidence_threshold=0.7
                )
            )
        ]
        
        for profile in adaptive_templates:
            profile.optimize_for_structure_type()
            self.save_template(profile)
            self.save_profile(profile)
        
        logger.info(f"Created {len(adaptive_templates)} adaptive template profiles")
    
    @classmethod
    def create_specialized_profile(cls, 
                                 name: str,
                                 description: str = "",
                                 structure_type: StructureType = StructureType.GENERIC,
                                 feature_overrides: Dict[str, Dict[str, Any]] = None) -> 'DetectorProfile':
        """
        Create a specialized profile with feature parameter overrides
        
        Args:
            name: Profile name
            description: Profile description 
            structure_type: Type of archaeological structure
            feature_overrides: Dict of {feature_name: {param_name: value}} to override defaults
            
        Returns:
            DetectorProfile with customized parameters
        """
        # Start with default configuration
        profile = cls(
            name=name,
            description=description,
            structure_type=structure_type
        )
        
        # Apply feature parameter overrides
        if feature_overrides:
            for feature_name, param_overrides in feature_overrides.items():
                if feature_name in profile.features:
                    # Update existing parameters
                    profile.features[feature_name].parameters.update(param_overrides)
                else:
                    logger.warning(f"Feature '{feature_name}' not found in profile")
        
        logger.info(f"Created specialized profile '{name}' with {len(feature_overrides or {})} feature overrides")
        return profile
    
    @classmethod
    def create_high_precision_profile(cls) -> 'DetectorProfile':
        """Create a profile optimized for high precision detection"""
        overrides = {
            "volume": {
                "context_weight": 0.5,  # High context sensitivity
                "min_volume_threshold": 0.5,  # Very low threshold for sensitivity
                "percentile_normalization": True
            },
            "histogram": {
                "bin_count": 30,  # More bins for precision
                "edge_enhancement": True,
                "noise_reduction": True
            },
            "compactness": {
                "n_angles": 72,  # More angles for precision
                "fourier_analysis": True
            }
        }
        
        profile = cls.create_specialized_profile(
            name="High Precision Detection",
            description="Profile optimized for maximum detection precision",
            feature_overrides=overrides
        )
        
        # Stricter thresholds for precision
        profile.thresholds.detection_threshold = 0.7
        profile.thresholds.confidence_threshold = 0.8
        
        return profile
    
    @classmethod
    def create_fast_survey_profile(cls) -> 'DetectorProfile':
        """Create a profile optimized for fast large-area surveys"""
        # Now we only need to specify what we want to change!
        overrides = {
            "volume": {
                "context_weight": 0.2,  # Less context for speed
                "percentile_normalization": False,  # Faster computation
                "local_statistics_radius": 1.5
            },
            "histogram": {
                "bin_count": 15,  # Fewer bins for speed
                "edge_enhancement": False,
                "adaptive_binning": False
            },
            "entropy": {"weight": 0.0},  # Disable for speed
            "planarity": {"weight": 0.0}  # Disable for speed
        }
        
        profile = cls.create_specialized_profile(
            name="Fast Survey Mode",
            description="Profile optimized for rapid large-area detection",
            feature_overrides=overrides
        )
        
        # Fast decision making
        profile.thresholds.early_decision_threshold = 0.75
        profile.thresholds.min_modules_for_decision = 2
        
        return profile


class DetectorProfileManager:
    """Manager for saving, loading, and organizing detector profiles"""
    
    def __init__(self, profiles_dir: str = "profiles", templates_dir: str = "templates"):
        self.profiles_dir = Path(profiles_dir)
        self.templates_dir = Path(templates_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        self._profiles_cache: Dict[str, DetectorProfile] = {}
    
    def save_profile(self, profile: DetectorProfile, filename: Optional[str] = None) -> str:
        """Save a detector profile to disk"""
        if filename is None:
            filename = f"{profile.name.lower().replace(' ', '_')}.json"
        
        filepath = self.profiles_dir / filename
        
        # Convert profile to dict for JSON serialization
        profile_dict = asdict(profile)
        
        # Convert enums to strings
        profile_dict['structure_type'] = profile.structure_type.value
        profile_dict['geometry']['patch_shape'] = profile.geometry.patch_shape.value
        
        with open(filepath, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)
        
        logger.info(f"Saved profile '{profile.name}' to {filepath}")
        return str(filepath)
    
    def load_profile(self, filename: str) -> DetectorProfile:
        """Load a detector profile from disk"""
        filepath = self.profiles_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Profile file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            profile_dict = json.load(f)
        
        # Convert string enums back to enum objects
        profile_dict['structure_type'] = StructureType(profile_dict['structure_type'])
        profile_dict['geometry']['patch_shape'] = PatchShape(profile_dict['geometry']['patch_shape'])
        
        # Reconstruct nested objects
        geometry = GeometricParameters(**profile_dict['geometry'])
        thresholds = DetectionThresholds(**profile_dict['thresholds'])
        
        # Reconstruct feature configurations
        features = {}
        for name, config_dict in profile_dict['features'].items():
            features[name] = FeatureConfiguration(**config_dict)
        
        profile_dict['geometry'] = geometry
        profile_dict['thresholds'] = thresholds
        profile_dict['features'] = features
        
        profile = DetectorProfile(**profile_dict)
        
        logger.info(f"Loaded profile '{profile.name}' from {filepath}")
        return profile
    
    def list_profiles(self) -> List[str]:
        """List all available profile files"""
        return [f.name for f in self.profiles_dir.glob("*.json")]
    
    def create_preset_profiles(self) -> None:
        """Create a set of preset profiles for common use cases"""
        presets = [
            DetectorProfile(
                name="Amazon Windmill",
                description="Optimized for detecting windmill structures in Amazonian terrain",
                structure_type=StructureType.WINDMILL,
                geometry=GeometricParameters(
                    resolution_m=0.5,
                    structure_radius_m=8.0,
                    patch_size_m=(20.0, 20.0)
                )
            ),
            DetectorProfile(
                name="High Resolution Settlement",
                description="High-resolution detection for small settlements",
                structure_type=StructureType.SETTLEMENT,
                geometry=GeometricParameters(
                    resolution_m=0.25,
                    structure_radius_m=15.0,
                    patch_size_m=(40.0, 40.0)
                )
            ),
            DetectorProfile(
                name="Large Scale Earthwork",
                description="Detection of large earthwork structures",
                structure_type=StructureType.EARTHWORK,
                geometry=GeometricParameters(
                    resolution_m=1.0,
                    structure_radius_m=25.0,
                    patch_size_m=(60.0, 60.0)
                )
            ),
            DetectorProfile(
                name="Fast Survey Mode",
                description="Quick detection with minimal features for large area surveys",
                structure_type=StructureType.GENERIC,
                features={
                    "histogram": FeatureConfiguration(enabled=True, weight=2.0),
                    "volume": FeatureConfiguration(enabled=True, weight=1.5,
                                                 parameters={
                                                     "adaptive_scaling": True,
                                                     "context_weight": 0.2,  # Less context for speed
                                                     "auto_range_adaptation": True,
                                                     "percentile_normalization": False,  # Faster computation
                                                     "local_statistics_radius": 1.5,  # Smaller radius for speed
                                                     "concentration_bonus": 1.1
                                                 }),
                    "dropoff": FeatureConfiguration(enabled=False),
                    "compactness": FeatureConfiguration(enabled=False),
                    "entropy": FeatureConfiguration(enabled=False),
                    "planarity": FeatureConfiguration(enabled=False)
                },
                thresholds=DetectionThresholds(
                    early_decision_threshold=0.75,
                    min_modules_for_decision=1
                )
            )
        ]
        
        for profile in presets:
            profile.optimize_for_structure_type()
            # Save templates to templates directory
            self.save_template(profile)
            # Also save to profiles for immediate use
            self.save_profile(profile)
        
        # Create additional adaptive templates
        self.create_adaptive_template_profiles()
        
        logger.info(f"Created {len(presets)} standard template profiles + adaptive templates")
    
    def save_template(self, profile: DetectorProfile, filename: Optional[str] = None) -> str:
        """Save a detector profile as a template"""
        if filename is None:
            filename = f"{profile.name.lower().replace(' ', '_')}.json"
        
        filepath = self.templates_dir / filename
        
        # Convert profile to dict for JSON serialization
        profile_dict = asdict(profile)
        
        # Convert enums to strings
        profile_dict['structure_type'] = profile.structure_type.value
        profile_dict['geometry']['patch_shape'] = profile.geometry.patch_shape.value
        
        with open(filepath, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)
        
        logger.info(f"Saved template '{profile.name}' to {filepath}")
        return str(filepath)
    
    def load_template(self, filename: str) -> DetectorProfile:
        """Load a detector profile template"""
        filepath = self.templates_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Template file not found: {filepath}")
        
        return self._load_profile_from_path(filepath)
    
    def list_templates(self) -> List[str]:
        """List all available template files"""
        return [f.name for f in self.templates_dir.glob("*.json")]
    
    def copy_template_to_profile(self, template_name: str, new_profile_name: str = None) -> DetectorProfile:
        """Copy a template to the profiles directory for customization"""
        # Load the template
        template = self.load_template(template_name)
        
        # Optionally rename
        if new_profile_name:
            template.name = new_profile_name
        
        # Save to profiles directory
        self.save_profile(template)
        
        logger.info(f"Copied template '{template_name}' to profiles as '{template.name}'")
        return template
    
    def _load_profile_from_path(self, filepath: Path) -> DetectorProfile:
        """Helper method to load profile from any path"""
        with open(filepath, 'r') as f:
            profile_dict = json.load(f)
        
        # Convert string enums back to enum objects
        profile_dict['structure_type'] = StructureType(profile_dict['structure_type'])
        profile_dict['geometry']['patch_shape'] = PatchShape(profile_dict['geometry']['patch_shape'])
        
        # Reconstruct nested objects
        geometry = GeometricParameters(**profile_dict['geometry'])
        thresholds = DetectionThresholds(**profile_dict['thresholds'])
        
        # Check if this is a new-style template with feature_overrides
        if 'feature_overrides' in profile_dict:
            # Create profile using the new override system
            feature_overrides = profile_dict.get('feature_overrides', {})
            feature_weights = profile_dict.get('feature_weights', {})
            
            # Create base profile
            profile = DetectorProfile(
                name=profile_dict.get('name', 'Loaded Profile'),
                description=profile_dict.get('description', ''),
                structure_type=profile_dict['structure_type']
            )
            
            # Apply geometry and thresholds
            profile.geometry = geometry
            profile.thresholds = thresholds
            
            # Apply feature parameter overrides
            for feature_name, param_overrides in feature_overrides.items():
                if feature_name in profile.features:
                    # Extract polarity_preference separately since it's not a parameter
                    if 'polarity_preference' in param_overrides:
                        profile.features[feature_name].polarity_preference = param_overrides['polarity_preference']
                        # Remove from param_overrides so it doesn't go into parameters
                        param_overrides = {k: v for k, v in param_overrides.items() if k != 'polarity_preference'}
                    profile.features[feature_name].parameters.update(param_overrides)
                    
            # Apply custom weights
            for feature_name, weight in feature_weights.items():
                if feature_name in profile.features:
                    profile.features[feature_name].weight = weight
            
            # Apply polarity preferences from template
            polarity_preferences = profile_dict.get('polarity_preferences', {})
            for feature_name, polarity_pref in polarity_preferences.items():
                if feature_name in profile.features:
                    profile.features[feature_name].polarity_preference = polarity_pref
                    
            # Set other profile attributes
            profile.aggregation_method = profile_dict.get('aggregation_method', 'streaming')
            profile.parallel_execution = profile_dict.get('parallel_execution', True)
            profile.max_workers = profile_dict.get('max_workers', 4)
            profile.enable_refinement = profile_dict.get('enable_refinement', True)
            profile.max_refinement_attempts = profile_dict.get('max_refinement_attempts', 2)
            
        else:
            # Legacy format - reconstruct feature configurations
            features = {}
            for name, config_dict in profile_dict['features'].items():
                features[name] = FeatureConfiguration(**config_dict)
            
            profile_dict['geometry'] = geometry
            profile_dict['thresholds'] = thresholds
            profile_dict['features'] = features
            
            profile = DetectorProfile(**profile_dict)
        
        logger.info(f"Loaded profile '{profile.name}' from {filepath}")
        return profile

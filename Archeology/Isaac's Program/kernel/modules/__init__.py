"""
Module exports for the feature modules package with unified architecture
"""

from .base_module import BaseFeatureModule, FeatureResult
from .registry import FeatureModuleRegistry, feature_registry

# Import individual feature modules (clean modular architecture)
from .features.histogram_module import ElevationHistogramModule
from .features.volume_module import VolumeModule as VolumeAnalysisModule
from .features.compactness_module import CompactnessModule
from .features.dropoff_module import DropoffSharpnessModule as EdgeAnalysisModule  
from .features.entropy_module import ElevationEntropyModule as EntropyAnalysisModule
from .features.planarity_module import PlanarityModule

__all__ = [
    "BaseFeatureModule",
    "FeatureResult", 
    "FeatureModuleRegistry",
    "feature_registry",
    # Individual feature modules
    "ElevationHistogramModule",
    "VolumeAnalysisModule", 
    "CompactnessModule",
    "EdgeAnalysisModule",
    "EntropyAnalysisModule",
    "PlanarityModule"
]

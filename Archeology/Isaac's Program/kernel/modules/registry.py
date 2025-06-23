"""
Feature Module Registry for Dynamic Loading

This module provides a registry system for dynamically loading and managing
feature detection modules in the G₂ kernel subsystem.
"""

import importlib
import inspect
from typing import Dict, List, Type, Any
from pathlib import Path
import logging

from .base_module import BaseFeatureModule

logger = logging.getLogger(__name__)


class FeatureModuleRegistry:
    """Registry for managing feature modules with dynamic loading capabilities"""
    
    def __init__(self):
        self._modules: Dict[str, Type[BaseFeatureModule]] = {}
        self._instances: Dict[str, BaseFeatureModule] = {}
    
    def register(self, name: str, module_class: Type[BaseFeatureModule], weight: float = 1.0):
        """
        Register a feature module class
        
        Args:
            name: Unique name for the module
            module_class: Class inheriting from BaseFeatureModule
            weight: Default weight for this module
        """
        if not issubclass(module_class, BaseFeatureModule):
            raise ValueError(f"Module {module_class} must inherit from BaseFeatureModule")
        
        self._modules[name] = module_class
        logger.info(f"Registered feature module: {name} -> {module_class.__name__}")
    
    def unregister(self, name: str):
        """Unregister a feature module"""
        if name in self._modules:
            del self._modules[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Unregistered feature module: {name}")
    
    def get_module(self, name: str, weight: float = None) -> BaseFeatureModule:
        """
        Get an instance of a feature module
        
        Args:
            name: Name of the module
            weight: Override weight for this instance
            
        Returns:
            Instance of the feature module
        """
        if name not in self._modules:
            raise KeyError(f"Feature module '{name}' not registered")
        
        # Create new instance if not cached or weight is different
        if name not in self._instances or (weight is not None):
            module_class = self._modules[name]
            instance = module_class(weight=weight or 1.0)
            if weight is None:  # Only cache if using default weight
                self._instances[name] = instance
            return instance
        
        return self._instances[name]
    
    def get_all_modules(self, weights: Dict[str, float] = None) -> Dict[str, BaseFeatureModule]:
        """
        Get instances of all registered modules
        
        Args:
            weights: Optional weight overrides for modules
            
        Returns:
            Dictionary of module instances
        """
        weights = weights or {}
        modules = {}
        
        for name in self._modules:
            weight = weights.get(name)
            modules[name] = self.get_module(name, weight)
        
        return modules
    
    def list_modules(self) -> List[str]:
        """List all registered module names"""
        return list(self._modules.keys())
    
    def auto_discover(self, features_package: str = "kernel.modules.features"):
        """
        Automatically discover and register feature modules from a package
        
        Args:
            features_package: Package path to scan for feature modules
        """
        try:
            features_path = Path(__file__).parent / "features"
            
            for py_file in features_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                module_name = py_file.stem
                full_module_path = f"{features_package}.{module_name}"
                
                try:
                    # Import the module
                    module = importlib.import_module(full_module_path)
                    
                    # Find classes that inherit from BaseFeatureModule
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseFeatureModule) and 
                            obj != BaseFeatureModule and
                            obj.__module__ == full_module_path):
                            
                            # Use module name as registry key (removing 'Module' suffix if present)
                            registry_name = module_name.replace("_module", "").replace("_", "")
                            if registry_name.endswith("module"):
                                registry_name = registry_name[:-6]
                            
                            self.register(registry_name, obj)
                            
                except Exception as e:
                    logger.warning(f"Failed to import feature module {full_module_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")


# Global registry instance
feature_registry = FeatureModuleRegistry()

# Register individual feature modules
def register_individual_modules():
    """Register the individual feature modules with clean modular architecture"""
    try:
        from .features.histogram_module import ElevationHistogramModule
        from .features.volume_module import VolumeModule
        from .features.compactness_module import CompactnessModule
        from .features.dropoff_module import DropoffSharpnessModule
        from .features.entropy_module import ElevationEntropyModule
        from .features.planarity_module import PlanarityModule
        from .features.volume_distribution_module import VolumeDistributionModule
        
        # Register individual modules with their default weights
        feature_registry.register("ElevationHistogram", ElevationHistogramModule)
        feature_registry.register("Volume", VolumeModule)
        feature_registry.register("Compactness", CompactnessModule)
        feature_registry.register("DropoffSharpness", DropoffSharpnessModule)
        feature_registry.register("ElevationEntropy", ElevationEntropyModule)
        feature_registry.register("Planarity", PlanarityModule)
        feature_registry.register("VolumeDistribution", VolumeDistributionModule)
        
        logger.info("Registered individual feature modules")
        
    except ImportError as e:
        logger.warning(f"Could not register individual modules: {e}")

# Auto-register individual modules
register_individual_modules()

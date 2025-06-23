"""
Base Feature Module Interface for G₂ Detection System

All feature modules inherit from this base class and implement the compute method
to provide independent, parallelizable feature validation with comprehensive documentation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class ParameterInfo:
    """Documentation for a module parameter"""
    name: str
    description: str
    data_type: str
    default_value: Any
    valid_range: Optional[str] = None
    example_values: List[Any] = field(default_factory=list)
    impact: str = ""  # How this parameter affects results


@dataclass
class MetricInfo:
    """Documentation for a result metric"""
    name: str
    description: str
    range: str
    interpretation: str
    good_values: str = ""  # What values indicate good/bad results
    example: str = ""


@dataclass
class FeatureDocumentation:
    """Complete documentation for a feature module"""
    module_name: str
    purpose: str
    how_it_works: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    output_metrics: List[MetricInfo] = field(default_factory=list)
    interpretation_guide: str = ""
    typical_scores: Dict[str, str] = field(default_factory=dict)  # e.g., {"windmill": "0.7-0.9", "linear": "0.1-0.3"}


@dataclass
class FeatureResult:
    """Result from a feature module computation with polarity support"""
    score: float  # Confidence score in [0, 1] range
    polarity: str = "positive"  # "positive" or "negative" for bidirectional evidence
    metadata: Dict[str, Any] = None
    reason: str = ""
    valid: bool = True


class BaseFeatureModule(ABC):
    """Base class for all feature modules with comprehensive documentation support"""
    
    def __init__(self, name: str = None, weight: float = 1.0):
        self.name = name or self.__class__.__name__
        self.weight = weight
        self.resolution_m = 0.5
        self.structure_radius_px = 16
    
    @abstractmethod
    def compute(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Compute feature score for the given elevation patch
        
        Args:
            elevation_patch: 2D numpy array of elevation data
            **kwargs: Additional parameters specific to the feature
            
        Returns:
            FeatureResult with score, metadata, and validation info
        """
        pass
    
    @property
    def documentation(self) -> FeatureDocumentation:
        """Get comprehensive documentation for this feature module"""
        return self.get_documentation()
    
    @property
    def parameter_docs(self) -> List[ParameterInfo]:
        """Get documentation for all parameters"""
        return self.get_documentation().parameters
    
    @property
    def metric_docs(self) -> List[MetricInfo]:
        """Get documentation for all output metrics"""
        return self.get_documentation().output_metrics
    
    @property
    def usage_guide(self) -> str:
        """Get usage and interpretation guide"""
        doc = self.get_documentation()
        guide = f"📋 {doc.module_name}\n"
        guide += f"Purpose: {doc.purpose}\n\n"
        guide += f"How it works: {doc.how_it_works}\n\n"
        
        if doc.parameters:
            guide += "Parameters:\n"
            for param in doc.parameters:
                guide += f"  • {param.name} ({param.data_type}): {param.description}\n"
                if param.valid_range:
                    guide += f"    Range: {param.valid_range}\n"
        
        guide += f"\nInterpretation: {doc.interpretation_guide}\n"
        
        if doc.typical_scores:
            guide += "\nTypical scores:\n"
            for structure, score_range in doc.typical_scores.items():
                guide += f"  • {structure}: {score_range}\n"
        
        return guide
    
    def explain_result(self, result: FeatureResult) -> str:
        """Get human-readable explanation of a specific result"""
        explanation = f"📊 {self.name} Analysis\n"
        explanation += f"Score: {result.score:.3f}\n"
        explanation += f"Reason: {result.reason}\n"
        
        if result.metadata:
            explanation += "\nDetailed Metrics:\n"
            for key, value in result.metadata.items():
                if isinstance(value, (int, float)):
                    explanation += f"  • {key}: {value:.3f}\n"
                elif isinstance(value, list) and key == "explanations":
                    explanation += f"  • Explanations:\n"
                    for exp in value[:3]:  # Top 3 explanations
                        explanation += f"    - {exp}\n"
                else:
                    explanation += f"  • {key}: {value}\n"
        
        return explanation
    
    def extract_features(self, elevation_patch: np.ndarray, **kwargs) -> FeatureResult:
        """
        Alias for compute method to maintain compatibility
        """
        return self.compute(elevation_patch, **kwargs)
    
    def set_parameters(self, resolution_m: float, structure_radius_px: int):
        """Set common parameters for the feature module"""
        self.resolution_m = resolution_m
        self.structure_radius_px = structure_radius_px
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Return default parameters for this feature module.
        Subclasses should override this to define their parameters.
        
        Returns:
            Dictionary of parameter_name: default_value
        """
        return {}
    
    def configure(self, **kwargs):
        """
        Configure module parameters dynamically.
        Default implementation sets any parameter that exists as an attribute.
        
        Args:
            **kwargs: Parameter name-value pairs to configure
        """
        configured_count = 0
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                configured_count += 1
            else:
                # Only warn if it's not a common parameter that might not apply to all modules
                if param not in ['weight', 'enabled']:
                    print(f"Warning: {self.name} module doesn't have parameter '{param}'")
        
        if configured_count > 0:
            print(f"{self.name} module configured with {configured_count} parameters")

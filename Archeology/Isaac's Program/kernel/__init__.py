"""
Kernel Subsystem for φ⁰ (Phi-Zero) Structure Detection

This module provides a modularized, parallelized architecture for geometric structure detection
based on the φ⁰ core algorithm with G₂-level recursive reasoning capabilities.

Key Components:
- Feature modules: Independent validators running in parallel
- Aggregator: Recursive detection decision engine combining evidence
- Core detector: Main detection orchestrator
"""

from .core_detector import G2StructureDetector, G2DetectionResult
from .aggregator import RecursiveDetectionAggregator, AggregationResult
from .modules import *

__version__ = "1.0.0"
__all__ = ["G2StructureDetector", "G2DetectionResult", "RecursiveDetectionAggregator", "AggregationResult"]

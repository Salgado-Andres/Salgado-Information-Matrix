from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class ElevationPatch:
    """Container for elevation data and metadata"""
    elevation_data: np.ndarray
    lat: float = None
    lon: float = None
    source: str = "unknown"
    resolution_m: float = 0.5
    coordinates: Tuple[float, float] = None
    patch_size_m: float = None
    metadata: Dict = None

@dataclass
class DetectionCandidate:
    """Container for detection results"""
    center_y: int
    center_x: int
    psi0_score: float
    coherence: float = 0.0
    confidence: float = 0.0

@dataclass
class DetectionResult:
    """Enhanced detection result with geometric validation"""
    detected: bool
    confidence: float
    reason: str
    max_score: float
    center_score: float
    geometric_score: float
    details: Dict

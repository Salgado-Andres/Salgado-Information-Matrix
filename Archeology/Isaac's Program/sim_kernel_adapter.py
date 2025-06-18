"""Lightweight adapter to use salgado_sim_kernel_v3 with test scripts."""

import numpy as np
from .sim_data_structures import DetectionResult
from kernel.salgado_sim_kernel_v3 import (
    extract_octonionic_features,
    detect_structures_phi0,
)

class PhiZeroStructureDetector:
    """Adapter providing a subset of the original detector interface."""

    def __init__(self, resolution_m: float = 0.5, kernel_size: int = 21, structure_type: str = "windmill"):
        self.resolution_m = resolution_m
        self.kernel_size = kernel_size
        self.structure_type = structure_type

    def extract_octonionic_features(self, elevation_data: np.ndarray) -> np.ndarray:
        return extract_octonionic_features(elevation_data, self.resolution_m)

    def detect_with_geometric_validation(self, feature_data: np.ndarray, elevation_data: np.ndarray = None):
        result = detect_structures_phi0(feature_data, elevation_data)
        candidates = result.get("candidates", [])
        if not candidates:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                reason="no candidates",
                max_score=0.0,
                center_score=0.0,
                geometric_score=0.0,
                details=result,
            )
        top = candidates[0]
        return DetectionResult(
            detected=True,
            confidence=top.get("score", 0.0),
            reason="detected",
            max_score=top.get("score", 0.0),
            center_score=top.get("phi0_score", 0.0),
            geometric_score=top.get("coherence_score", 0.0),
            details=result,
        )

    def learn_pattern_kernel(self, training_patches, use_apex_center: bool = True):
        if not training_patches:
            return np.zeros((self.kernel_size, self.kernel_size))
        kernels = [p.elevation_data for p in training_patches]
        return np.mean(np.stack(kernels), axis=0)

    def get_adaptive_thresholds(self):
        return {
            "min_phi0_threshold": 0.3,
            "geometric_threshold": 0.4,
        }

    def update_adaptive_thresholds_from_validation(self, *args, **kwargs):
        return self.get_adaptive_thresholds()

    def visualize_adaptive_threshold_performance(self, *args, **kwargs):
        pass

#!/usr/bin/env python3
"""
Phi-Zero (φ⁰) Circular Structure Detection Core - Clean Implementation

A streamlined, well-organized implementation of the v8 circular structure detection algorithm.
Combines elevation histogram matching with geometric pattern analysis for robust
detection of circular elevated structures in elevation data.

Key Features:
- 8-dimensional octonionic feature extraction
- Raw elevation histogram matching
- Geometric pattern validation
- Performance analytics and visualization
- Structure-agnostic design (windmills, towers, mounds, etc.)

Author: Structure Detection Team
Version: v8-clean
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import (
    uniform_filter, gaussian_filter, maximum_filter,
    distance_transform_edt, sobel
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

from .sim_data_structures import (
    ElevationPatch,
    DetectionCandidate,
    DetectionResult,
)


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================

class PhiZeroStructureDetector:
    """
    Clean implementation of the Phi-Zero circular structure detection algorithm.
    
    This detector uses an 8-dimensional octonionic feature space combined with
    elevation histogram matching and geometric pattern validation to detect
    circular elevated structures in elevation data.
    
    Supports various structure types:
    - Windmill foundations
    - Ancient mounds and settlements  
    - Communication towers
    - Water towers
    - Archaeological circular features
    """
    
    def __init__(self, resolution_m: float = 0.5, kernel_size: int = 21, structure_type: str = "windmill"):
        """
        Initialize the Phi-Zero structure detector.
        
        Args:
            resolution_m: Spatial resolution in meters per pixel
            kernel_size: Size of the detection kernel (should be odd)
            structure_type: Type of structure to detect ("windmill", "tower", "mound", "settlement", "generic")
        """
        self.resolution_m = resolution_m
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.structure_type = structure_type
        self.n_features = 8
        
        # Structure-specific parameters
        if structure_type == "windmill":
            self.detection_threshold = 0.50
            self.structure_radius_m = 8.0  # Typical windmill foundation radius
            self.structure_scales = [10, 15, 25]
        elif structure_type == "tower":
            self.detection_threshold = 0.45
            self.structure_radius_m = 5.0  # Communication/water towers
            self.structure_scales = [5, 8, 12]
        elif structure_type == "mound":
            self.detection_threshold = 0.30
            self.structure_radius_m = 15.0  # Archaeological mounds
            self.structure_scales = [15, 25, 40]
        elif structure_type == "settlement":
            self.detection_threshold = 0.25
            self.structure_radius_m = 20.0  # Settlement circles
            self.structure_scales = [20, 30, 50]
        else:  # generic
            self.detection_threshold = 0.35
            self.structure_radius_m = 10.0
            self.structure_scales = [5, 10, 20]
        
        self.structure_radius_px = int(self.structure_radius_m / resolution_m)
        
        # Internal state
        self.psi0_kernel = None
        self.elevation_kernel = None
        
        # Training-derived statistics for adaptive thresholds
        self.training_stats = {
            'height_prominence': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
            'volume_above_base': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
            'elevation_range': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
            'structure_max_height': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
            'relative_prominence': {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None},
            'sample_count': 0
        }
        
        logger.info(f"PhiZero detector initialized for {structure_type} structures at {resolution_m}m resolution")
    
    
    # =========================================================================
    # CORE FEATURE EXTRACTION
    # =========================================================================
    
    def extract_octonionic_features(self, elevation_data: np.ndarray) -> np.ndarray:
        """
        Extract 8-dimensional octonionic features from elevation data with enhanced size discrimination.
        
        Features:
        0. Radial Height Prominence (enhanced with volume discrimination)
        1. Circular Symmetry
        2. Radial Gradient Consistency
        3. Ring Edge Sharpness
        4. Hough Response
        5. Local Planarity
        6. Isolation Score (enhanced with height prominence)
        7. Geometric Coherence
        
        Args:
            elevation_data: 2D elevation array
            
        Returns:
            3D array of shape (h, w, 8) containing features
        """
        elevation = np.nan_to_num(elevation_data.astype(np.float64), nan=0.0)
        h, w = elevation.shape
        features = np.zeros((h, w, 8))
        
        # Compute gradients once
        grad_x = np.gradient(elevation, axis=1) / self.resolution_m
        grad_y = np.gradient(elevation, axis=0) / self.resolution_m
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract base features
        base_radial_prominence = self._compute_radial_prominence(elevation)
        base_isolation = self._compute_isolation_score(elevation)
        
        # Enhanced size discrimination metrics
        volume_metric = self._compute_structure_volume_metric(elevation)
        height_prominence = self._compute_height_prominence_metric(elevation)
        
        # Combine features with size discrimination
        features[..., 0] = base_radial_prominence * (0.7 + 0.3 * volume_metric)  # Volume-weighted prominence
        features[..., 1] = self._compute_circular_symmetry(elevation)
        features[..., 2] = self._compute_radial_gradient_consistency(grad_x, grad_y)
        features[..., 3] = self._compute_ring_edges(elevation)
        features[..., 4] = self._compute_hough_response(grad_magnitude)
        features[..., 5] = self._compute_local_planarity(elevation)
        features[..., 6] = base_isolation * (0.6 + 0.4 * height_prominence)  # Height-weighted isolation
        features[..., 7] = self._compute_geometric_coherence(elevation, grad_magnitude)
        
        return features
    
    def _compute_radial_prominence(self, elevation: np.ndarray) -> np.ndarray:
        """Compute f0: Radial Height Prominence"""
        radius = self.structure_radius_px
        local_max_filter = maximum_filter(elevation, size=2*radius+1)
        local_mean = uniform_filter(elevation, size=2*radius+1)
        prominence = elevation - local_mean
        relative_prominence = prominence / (local_max_filter - local_mean + 1e-6)
        return np.clip(relative_prominence, 0, 1)
    
    def _compute_circular_symmetry(self, elevation: np.ndarray) -> np.ndarray:
        """Compute f1: Circular Symmetry around each point"""
        h, w = elevation.shape
        radius = self.structure_radius_px
        symmetry = np.zeros((h, w))
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pad_size = radius + 1
        padded = np.pad(elevation, pad_size, mode='reflect')
        
        for y in range(h):
            for x in range(w):
                y_pad, x_pad = y + pad_size, x + pad_size
                values = []
                
                for angle in angles:
                    dy = int(radius * np.sin(angle))
                    dx = int(radius * np.cos(angle))
                    if (0 <= y_pad + dy < padded.shape[0] and 0 <= x_pad + dx < padded.shape[1]):
                        values.append(padded[y_pad + dy, x_pad + dx])
                
                if len(values) >= 6:
                    values = np.array(values)
                    std_dev = np.std(values)
                    mean_val = np.mean(values)
                    relative_std = std_dev / (abs(mean_val) + 1e-6)
                    symmetry[y, x] = 1.0 / (1.0 + 2.0 * relative_std)
                else:
                    symmetry[y, x] = 0.0
        
        return symmetry
    
    def _compute_radial_gradient_consistency(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Compute f2: Radial Gradient Consistency"""
        h, w = grad_x.shape
        radius = self.structure_radius_px
        consistency = np.zeros((h, w))
        
        for cy in range(radius, h-radius):
            for cx in range(radius, w-radius):
                y_min, y_max = cy-radius, cy+radius+1
                x_min, x_max = cx-radius, cx+radius+1
                local_gx = grad_x[y_min:y_max, x_min:x_max]
                local_gy = grad_y[y_min:y_max, x_min:x_max]
                
                local_y, local_x = np.ogrid[:2*radius+1, :2*radius+1]
                dy = local_y - radius
                dx = local_x - radius
                dist = np.sqrt(dy**2 + dx**2) + 1e-6
                
                expected_gx = dx / dist
                expected_gy = dy / dist
                dot_product = local_gx * expected_gx + local_gy * expected_gy
                weight = np.exp(-dist / radius) * np.sqrt(local_gx**2 + local_gy**2)
                mask = dist <= radius
                
                if np.sum(mask) > 0:
                    consistency[cy, cx] = np.sum(dot_product[mask] * weight[mask]) / (np.sum(weight[mask]) + 1e-6)
        
        return np.clip(consistency, -1, 1)
    
    def _compute_ring_edges(self, elevation: np.ndarray) -> np.ndarray:
        """Compute f3: Ring Edge Sharpness using DoG"""
        radius = self.structure_radius_px
        sigma1 = radius * 0.8 * self.resolution_m
        sigma2 = radius * 1.2 * self.resolution_m
        dog = gaussian_filter(elevation, sigma1) - gaussian_filter(elevation, sigma2)
        edge_strength = np.abs(dog)
        
        if np.percentile(edge_strength, 95) > 0:
            edge_strength = edge_strength / (np.percentile(edge_strength, 95) + 1e-6)
        
        return np.clip(edge_strength, 0, 1)
    
    def _compute_hough_response(self, gradient_magnitude: np.ndarray) -> np.ndarray:
        """Compute f4: Hough Circle Transform Response"""
        h, w = gradient_magnitude.shape
        hough_response = np.zeros((h, w))
        target_radius = self.structure_radius_px
        
        try:
            from skimage.transform import hough_circle
            edges = gradient_magnitude > np.percentile(gradient_magnitude, 75)
            edges = edges.astype(np.uint8) * 255
            radii = np.arange(max(5, target_radius-2), target_radius+3, 1)
            hough_res = hough_circle(edges, radii)
            
            for radius_idx, radius in enumerate(radii):
                accumulator = hough_res[radius_idx]
                weight = np.exp(-0.5 * ((radius - target_radius) / 2)**2)
                hough_response += accumulator * weight
            
            if np.max(hough_response) > 0:
                hough_response = hough_response / np.max(hough_response)
                
        except Exception as e:
            logger.debug(f"Hough transform failed: {e}, using zero response")
        
        return hough_response
    
    def _compute_local_planarity(self, elevation: np.ndarray) -> np.ndarray:
        """Compute f5: Local Planarity via least squares fitting"""
        h, w = elevation.shape
        radius = self.structure_radius_px
        planarity = np.zeros((h, w))
        
        for y in range(radius, h-radius):
            for x in range(radius, w-radius):
                local_patch = elevation[y-radius:y+radius+1, x-radius:x+radius+1]
                yy, xx = np.mgrid[:local_patch.shape[0], :local_patch.shape[1]]
                center_y, center_x = radius, radius
                mask = (yy - center_y)**2 + (xx - center_x)**2 <= radius**2
                
                if np.sum(mask) > 3:
                    points = np.column_stack([xx[mask], yy[mask], np.ones(np.sum(mask))])
                    z_values = local_patch[mask]
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(points, z_values, rcond=None)
                        z_fit = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
                        residuals = np.abs(local_patch - z_fit)[mask]
                        planarity[y, x] = 1.0 / (1.0 + np.std(residuals))
                    except:
                        planarity[y, x] = 0.0
        
        return planarity
    
    def _compute_isolation_score(self, elevation: np.ndarray) -> np.ndarray:
        """Compute f6: Isolation Score"""
        radius = self.structure_radius_px
        local_max = maximum_filter(elevation, size=2*radius+1)
        extended_max = maximum_filter(elevation, size=4*radius+1)
        isolation = (local_max == extended_max).astype(float)
        prominence = local_max - uniform_filter(elevation, size=2*radius+1)
        prominence_std = np.std(prominence) + 1e-6
        isolation = isolation * (1 - np.exp(-prominence / prominence_std))
        return isolation
    
    def _compute_geometric_coherence(self, elevation: np.ndarray, gradient_magnitude: np.ndarray) -> np.ndarray:
        """Compute f7: Geometric Coherence"""
        radius = self.structure_radius_px
        edges = gradient_magnitude > np.percentile(gradient_magnitude, 80)
        dist_from_edge = distance_transform_edt(~edges)
        
        h, w = elevation.shape
        coherence = np.zeros_like(elevation)
        pad = radius
        
        if h > 2*pad and w > 2*pad:
            center_y, center_x = h//2, w//2
            y_start, y_end = max(pad, center_y-10), min(h-pad, center_y+11)
            x_start, x_end = max(pad, center_x-10), min(w-pad, center_x+11)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    local_dist = dist_from_edge[y-pad:y+pad+1, x-pad:x+pad+1]
                    center_dist = local_dist[pad, pad]
                    mean_edge_dist = (np.mean(local_dist[0, :]) + np.mean(local_dist[-1, :]) + 
                                    np.mean(local_dist[:, 0]) + np.mean(local_dist[:, -1])) / 4
                    if mean_edge_dist > 0:
                        coherence[y, x] = center_dist / (mean_edge_dist + 1)
        
        coherence = gaussian_filter(coherence, sigma=2)
        if np.max(coherence) > 0:
            coherence = coherence / np.max(coherence)
        
        return coherence
    
    
    # =========================================================================
    # TRAINING STATISTICS COLLECTION
    # =========================================================================
    
    def _analyze_patch_statistics(self, elevation: np.ndarray) -> Dict:
        """
        Analyze statistical properties of a training patch for adaptive threshold learning.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            Dictionary with statistical measurements
        """
        h, w = elevation.shape
        center_y, center_x = h // 2, w // 2
        radius = self.structure_radius_px
        
        # Structure region
        y, x = np.ogrid[:h, :w]
        structure_mask = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
        
        # Surrounding region (ring around structure)
        surround_inner = radius + 2
        surround_outer = min(h, w) // 2 - 2
        surround_mask = (((y - center_y)**2 + (x - center_x)**2) > surround_inner**2) & \
                       (((y - center_y)**2 + (x - center_x)**2) <= surround_outer**2)
        
        if not (np.any(structure_mask) and np.any(surround_mask)):
            return None
        
        # Height statistics
        structure_heights = elevation[structure_mask]
        surround_heights = elevation[surround_mask]
        
        structure_max = np.max(structure_heights)
        structure_mean = np.mean(structure_heights)
        surround_mean = np.mean(surround_heights)
        surround_std = np.std(surround_heights)
        
        # Calculate base elevation (use edge areas)
        edge_mask = np.zeros_like(elevation, dtype=bool)
        border_width = max(2, radius // 4)
        edge_mask[:border_width, :] = True
        edge_mask[-border_width:, :] = True
        edge_mask[:, :border_width] = True
        edge_mask[:, -border_width:] = True
        
        if np.any(edge_mask):
            base_elevation = np.median(elevation[edge_mask])
        else:
            base_elevation = np.median(elevation)
        
        # Calculate volume above base
        volume_above_base = np.sum(np.maximum(0, structure_heights - base_elevation)) * (self.resolution_m ** 2)
        
        # Measurements
        height_prominence = structure_max - surround_mean
        relative_prominence = height_prominence / (surround_std + 0.1)
        elevation_range = np.max(elevation) - np.min(elevation)
        
        return {
            'height_prominence': height_prominence,
            'volume_above_base': volume_above_base,
            'elevation_range': elevation_range,
            'structure_max_height': structure_max,
            'relative_prominence': relative_prominence,
            'base_elevation': base_elevation,
            'structure_mean': structure_mean,
            'surround_mean': surround_mean,
            'surround_std': surround_std
        }
    
    def _store_training_statistics(self, height_prominences, volumes, elevation_ranges, max_heights, relative_prominences):
        """
        Store collected training statistics for adaptive threshold setting.
        
        Args:
            height_prominences: List of height prominence values from training patches
            volumes: List of volume measurements
            elevation_ranges: List of elevation range measurements
            max_heights: List of maximum structure heights
            relative_prominences: List of relative prominence values
        """
        def compute_stats(values):
            if not values:
                return {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None}
            values_array = np.array(values)
            return {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array))
            }
        
        self.training_stats = {
            'height_prominence': compute_stats(height_prominences),
            'volume_above_base': compute_stats(volumes),
            'elevation_range': compute_stats(elevation_ranges),
            'structure_max_height': compute_stats(max_heights),
            'relative_prominence': compute_stats(relative_prominences),
            'sample_count': len(height_prominences)
        }
        
        logger.info(f"📊 Training statistics collected from {len(height_prominences)} valid patches")
    
    def get_adaptive_thresholds(self) -> Dict:
        """
        Get adaptive thresholds based on training statistics.
        
        Returns:
            Dictionary with adaptive threshold values
        """
        if self.training_stats['sample_count'] == 0:
            # Fallback to structure-type defaults - much more lenient based on real data
            if self.structure_type == "windmill":
                return {
                    'min_height_prominence': 1.0,  # Reduced from 2.0m
                    'min_volume': 20.0,            # Reduced from 50.0m³
                    'min_elevation_range': 0.5,    # Reduced from 1.0m
                    'geometric_threshold': 0.25,   # Reduced from 0.50
                    'min_phi0_threshold': 0.30     # Reduced from 0.35
                }
            elif self.structure_type == "tower":
                return {
                    'min_height_prominence': 1.5,
                    'min_volume': 20.0,
                    'min_elevation_range': 0.8,
                    'geometric_threshold': 0.45,
                    'min_phi0_threshold': 0.32
                }
            else:
                return {
                    'min_height_prominence': 1.0,
                    'min_volume': 25.0,
                    'min_elevation_range': 0.6,
                    'geometric_threshold': 0.42,
                    'min_phi0_threshold': 0.30
                }
        
        # Use training-derived thresholds (conservative approach)
        height_stats = self.training_stats['height_prominence']
        volume_stats = self.training_stats['volume_above_base']
        range_stats = self.training_stats['elevation_range']
        
        # ENHANCED: Data-driven threshold computation based on statistical distributions
        # Use percentile-based approach for robustness against outliers
        
        # Height prominence: Use 25th percentile as minimum (more conservative than median - std)
        min_height = max(0.5, height_stats['median'] - 0.5 * height_stats['std'])
        if height_stats['median'] is not None:
            # Use quantile approach: set threshold at 25th percentile of training data
            height_range = height_stats['max'] - height_stats['min']
            min_height = max(0.5, height_stats['min'] + 0.25 * height_range)
        
        # Volume: Use similar quantile approach
        min_volume = max(10.0, volume_stats['median'] - volume_stats['std'])
        if volume_stats['median'] is not None:
            volume_range = volume_stats['max'] - volume_stats['min']
            min_volume = max(10.0, volume_stats['min'] + 0.25 * volume_range)
        
        # Elevation range: Critical for size discrimination
        min_range = max(0.3, range_stats['median'] - range_stats['std'])
        if range_stats['median'] is not None:
            range_spread = range_stats['max'] - range_stats['min']
            min_range = max(0.3, range_stats['min'] + 0.25 * range_spread)
        
        # ADAPTIVE GEOMETRIC AND PHI0 THRESHOLDS
        # Base thresholds on training data quality and consistency
        
        # Calculate coefficient of variation (CV) for training consistency
        height_cv = height_stats['std'] / height_stats['mean'] if height_stats['mean'] > 0 else 1.0
        volume_cv = volume_stats['std'] / volume_stats['mean'] if volume_stats['mean'] > 0 else 1.0
        
        # Lower CV indicates more consistent training data -> can use stricter thresholds
        consistency_factor = 1.0 - min(0.3, (height_cv + volume_cv) / 2.0)  # Max 30% adjustment
        
        # Adaptive geometric threshold based on training consistency
        base_geometric = {
            "windmill": 0.40,  # Increased to eliminate false positives (FP has 0.290, TPs have 0.423+)
            "tower": 0.45,     # Increased from 0.40
            "mound": 0.35      # Increased from 0.30
        }.get(self.structure_type, 0.40)
        
        # Make geometric threshold more selective to eliminate false positives
        # False positive has geo=0.290, true positives have geo=0.423+
        adaptive_geometric = 0.40 - (0.05 * (1.0 - consistency_factor))  # Start at 40%, can go as low as 35%
        
        # Adaptive phi0 threshold - slightly higher to eliminate false positives  
        base_phi0 = {
            "windmill": 0.32,  # Increased from 0.30 to eliminate false positives
            "tower": 0.30,     # Increased from 0.28
            "mound": 0.24      # Increased from 0.22
        }.get(self.structure_type, 0.27)
        
        # Keep phi0 threshold more stable
        adaptive_phi0 = base_phi0 + (0.02 * consistency_factor)  # Up to 2% stricter
        
        return {
            'min_height_prominence': min_height,
            'min_volume': min_volume,
            'min_elevation_range': min_range,
            'geometric_threshold': adaptive_geometric,
            'min_phi0_threshold': adaptive_phi0,
            'training_derived': True,
            'training_quality': {
                'height_cv': height_cv,
                'volume_cv': volume_cv,
                'consistency_factor': consistency_factor,
                'sample_count': self.training_stats['sample_count']
            }
        }
    
    
    # =========================================================================
    # KERNEL LEARNING AND PATTERN MATCHING
    # =========================================================================
    
    def _find_optimal_kernel_center(self, elevation: np.ndarray) -> Tuple[int, int]:
        """
        Find the optimal center location for kernel extraction.
        
        This method locates the elevation apex (maximum point) within the patch,
        which typically corresponds to the center of circular elevated structures.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            Tuple of (center_y, center_x) coordinates of the optimal center
        """
        # Find the location of maximum elevation (apex)
        max_pos = np.unravel_index(np.argmax(elevation), elevation.shape)
        center_y, center_x = max_pos[0], max_pos[1]
        
        # Validate that the apex is not at the edge (which could be noise)
        h, w = elevation.shape
        min_margin = max(3, self.kernel_size // 4)  # Minimum distance from edge
        
        # If apex is too close to edge, find a more central high point
        if (center_y < min_margin or center_y >= h - min_margin or 
            center_x < min_margin or center_x >= w - min_margin):
            
            # Create a mask excluding edge regions
            mask = np.zeros_like(elevation, dtype=bool)
            mask[min_margin:h-min_margin, min_margin:w-min_margin] = True
            
            # Find the highest point within the safe region
            masked_elevation = elevation.copy()
            masked_elevation[~mask] = np.min(elevation) - 1  # Set edges to very low value
            
            max_pos = np.unravel_index(np.argmax(masked_elevation), masked_elevation.shape)
            center_y, center_x = max_pos[0], max_pos[1]
        
        return center_y, center_x
    
    def learn_pattern_kernel(self, training_patches: List[ElevationPatch], 
                           use_apex_center: bool = True) -> np.ndarray:
        """
        Learn the Phi-Zero kernel from training patches.
        
        This method builds both elevation and feature kernels for pattern matching.
        The elevation kernel preserves raw elevation patterns for histogram matching,
        while the feature kernel provides geometric validation.
        
        ENHANCED: Now collects training statistics for adaptive thresholds.
        
        Args:
            training_patches: List of training elevation patches
            use_apex_center: If True, center kernel on elevation peak; if False, use geometric center
            
        Returns:
            Normalized feature kernel
        """
        logger.info(f"Learning φ⁰ kernel from {len(training_patches)} training patches")
        logger.info(f"Kernel extraction mode: {'apex-centered' if use_apex_center else 'geometric-centered'}")
        
        if not training_patches:
            logger.warning("No training patches provided, using default kernel")
            return self._create_default_kernel()
        
        all_elevations = []
        all_features = []
        
        # ENHANCED: Collect training statistics for adaptive thresholds
        training_height_prominences = []
        training_volumes = []
        training_elevation_ranges = []
        training_max_heights = []
        training_relative_prominences = []
        
        for i, patch in enumerate(training_patches):
            if hasattr(patch, 'elevation_data') and patch.elevation_data is not None:
                elevation = patch.elevation_data
                features = self.extract_octonionic_features(elevation)
                
                # ENHANCED: Analyze training patch statistics
                patch_stats = self._analyze_patch_statistics(elevation)
                if patch_stats:
                    training_height_prominences.append(patch_stats['height_prominence'])
                    training_volumes.append(patch_stats['volume_above_base'])
                    training_elevation_ranges.append(patch_stats['elevation_range'])
                    training_max_heights.append(patch_stats['structure_max_height'])
                    training_relative_prominences.append(patch_stats['relative_prominence'])
                
                h, w = elevation.shape
                if h >= self.kernel_size and w >= self.kernel_size:
                    
                    # Determine kernel center location
                    if use_apex_center:
                        center_y, center_x = self._find_optimal_kernel_center(elevation)
                        logger.debug(f"Patch {i}: apex at ({center_y}, {center_x}), geometric center at ({h//2}, {w//2})")
                    else:
                        center_y, center_x = h // 2, w // 2
                    
                    # Check if kernel fits within patch bounds
                    half_kernel = self.kernel_size // 2
                    if (center_y >= half_kernel and center_y < h - half_kernel and
                        center_x >= half_kernel and center_x < w - half_kernel):
                        
                        # Extract kernel-sized patches centered on chosen point
                        start_y = center_y - half_kernel
                        start_x = center_x - half_kernel
                        
                        elevation_kernel = elevation[start_y:start_y+self.kernel_size, start_x:start_x+self.kernel_size]
                        feature_kernel = features[start_y:start_y+self.kernel_size, start_x:start_x+self.kernel_size, :]
                        
                        # Only use patches with meaningful variation
                        if np.std(elevation_kernel) > 0.01:
                            all_elevations.append(elevation_kernel)
                            all_features.append(feature_kernel)
                        else:
                            logger.warning(f"Patch {i}: Low variance in kernel, skipping")
                    else:
                        logger.warning(f"Patch {i}: Apex too close to edge for kernel size {self.kernel_size}, using geometric center")
                        # Fallback to geometric center
                        start_y = (h - self.kernel_size) // 2
                        start_x = (w - self.kernel_size) // 2
                        
                        elevation_kernel = elevation[start_y:start_y+self.kernel_size, start_x:start_x+self.kernel_size]
                        feature_kernel = features[start_y:start_y+self.kernel_size, start_x:start_x+self.kernel_size, :]
                        
                        if np.std(elevation_kernel) > 0.01:
                            all_elevations.append(elevation_kernel)
                            all_features.append(feature_kernel)
        
        # ENHANCED: Store training statistics for adaptive thresholds
        if training_height_prominences:
            self._store_training_statistics(
                training_height_prominences,
                training_volumes,
                training_elevation_ranges,
                training_max_heights,
                training_relative_prominences
            )
        
        if not all_features:
            logger.warning("No valid training features, using default kernel")
            return self._create_default_kernel()
        
        # Build kernels
        elevation_kernel = np.mean(all_elevations, axis=0)  # Raw elevation
        feature_kernel = np.mean(all_features, axis=0)      # Features
        
        # Store raw elevation kernel for histogram matching (NO G2 symmetrization)
        self.elevation_kernel = elevation_kernel
        
        # Normalize feature kernel
        feature_kernel_normalized = self._normalize_kernel(feature_kernel)
        self.psi0_kernel = feature_kernel_normalized
        
        logger.info(f"✅ φ⁰ kernel constructed with shape {feature_kernel_normalized.shape}")
        logger.info(f"📊 Elevation kernel range: {np.min(self.elevation_kernel):.2f} to {np.max(self.elevation_kernel):.2f}m")
        
        # ENHANCED: Log training-derived thresholds
        if self.training_stats['sample_count'] > 0:
            logger.info(f"📊 Training-derived statistics (n={self.training_stats['sample_count']}):")
            logger.info(f"   Height prominence: {self.training_stats['height_prominence']['median']:.2f}m (median)")
            logger.info(f"   Volume above base: {self.training_stats['volume_above_base']['median']:.1f}m³ (median)")
            logger.info(f"   Elevation range: {self.training_stats['elevation_range']['median']:.2f}m (median)")
            logger.info(f"   Max structure height: {self.training_stats['structure_max_height']['median']:.2f}m (median)")
        
        return feature_kernel_normalized
    
    def _create_default_kernel(self) -> np.ndarray:
        """Create a default radial kernel when no training data is available"""
        kernel = np.zeros((self.kernel_size, self.kernel_size, self.n_features))
        center = self.kernel_size // 2
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                if distance < self.kernel_size // 2:
                    value = np.exp(-distance / 3.0)
                    kernel[i, j, :] = value
        
        return self._normalize_kernel(kernel)
    
    def _apply_g2_symmetrization(self, pattern: np.ndarray) -> np.ndarray:
        """Apply G2 symmetrization (4-fold rotational symmetry)"""
        symmetrized = pattern.copy()
        for angle in [90, 180, 270]:
            rotated = np.rot90(pattern, k=angle//90, axes=(0, 1))
            symmetrized += rotated
        return symmetrized / 4.0
    
    def _normalize_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """Normalize kernel channels to zero mean, unit variance"""
        normalized = kernel.copy()
        for f in range(self.n_features):
            channel = kernel[..., f]
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 1e-10:
                normalized[..., f] = (channel - mean) / std
            else:
                normalized[..., f] = channel - mean
        return normalized
    
    
    # =========================================================================
    # PATTERN DETECTION AND MATCHING
    # =========================================================================
    
    def detect_patterns(self, feature_data: np.ndarray, elevation_data: np.ndarray = None, 
                       enable_center_bias: bool = True) -> np.ndarray:
        """
        Apply Phi-Zero pattern detection to feature data.
        
        Args:
            feature_data: 3D array of octonionic features
            elevation_data: Optional raw elevation data for histogram matching
            enable_center_bias: Whether to apply center bias weighting
            
        Returns:
            2D coherence map with detection scores
        """
        if self.psi0_kernel is None:
            raise ValueError("No φ⁰ kernel available. Train kernel first using learn_pattern_kernel()")
        
        h, w = feature_data.shape[:2]
        coherence_map = np.zeros((h, w))
        half_kernel = self.kernel_size // 2
        
        # Apply pattern matching at each location
        for y in range(half_kernel, h - half_kernel):
            for x in range(half_kernel, w - half_kernel):
                local_patch = feature_data[y-half_kernel:y+half_kernel+1, x-half_kernel:x+half_kernel+1, :]
                
                if local_patch.shape != (self.kernel_size, self.kernel_size, self.n_features):
                    continue
                
                # Extract corresponding elevation patch if available
                elevation_patch = None
                if elevation_data is not None:
                    elevation_patch = elevation_data[y-half_kernel:y+half_kernel+1, x-half_kernel:x+half_kernel+1]
                
                # Calculate coherence score
                coherence = self._calculate_coherence_score(local_patch, elevation_patch)
                coherence_map[y, x] = coherence
        
        # Apply center bias if requested
        if enable_center_bias:
            coherence_map = self._apply_center_bias(coherence_map)
        
        return coherence_map
    
    def _calculate_coherence_score(self, local_patch: np.ndarray, elevation_patch: np.ndarray = None) -> float:
        """
        Calculate coherence score combining elevation histogram matching and geometric validation.
        
        Strategy:
        1. Primary: Elevation histogram matching (80% weight)
        2. Secondary: Geometric feature correlation (20% weight)
        """
        try:
            elevation_score = 0.0
            geometric_score = 0.0
            
            # === ELEVATION HISTOGRAM MATCHING ===
            if elevation_patch is not None and self.elevation_kernel is not None:
                elevation_score = self._compute_elevation_histogram_score(elevation_patch, self.elevation_kernel)
            
            # === GEOMETRIC FEATURE VALIDATION ===
            if local_patch is not None and self.psi0_kernel is not None:
                geometric_score = self._compute_geometric_correlation_score(local_patch, self.psi0_kernel)
            
            # === WEIGHTED COMBINATION ===
            if elevation_score > 0 and geometric_score > 0:
                # Both available - heavily weight elevation
                combined_score = 0.80 * elevation_score + 0.20 * geometric_score
            elif elevation_score > 0:
                # Elevation only - use with small penalty
                combined_score = elevation_score * 0.85
            elif geometric_score > 0:
                # Geometric only - heavily penalize
                combined_score = geometric_score * 0.10
            else:
                combined_score = 0.0
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.warning(f"Coherence calculation error: {e}")
            return 0.0
    
    def _compute_elevation_histogram_score(self, local_elevation: np.ndarray, kernel_elevation: np.ndarray) -> float:
        """Compute elevation histogram matching score"""
        # ENHANCED: Stricter elevation range check for meaningful variation
        local_range = np.max(local_elevation) - np.min(local_elevation)
        kernel_range = np.max(kernel_elevation) - np.min(kernel_elevation)
        
        # More stringent minimum variation requirements by structure type
        if self.structure_type == "windmill":
            min_variation = 1.5  # Need at least 1.5m variation for windmills
        elif self.structure_type == "tower":
            min_variation = 1.2  # Tower foundations
        elif self.structure_type == "mound":
            min_variation = 0.8  # Archaeological features can be more subtle
        else:
            min_variation = 1.0  # Generic structures
        
        if local_range < min_variation or kernel_range < min_variation:
            return 0.0
        
        # Normalize to relative patterns (remove base elevation)
        local_relative = local_elevation - np.min(local_elevation)
        kernel_relative = kernel_elevation - np.min(kernel_elevation)
        
        local_max_rel = np.max(local_relative)
        kernel_max_rel = np.max(kernel_relative)
        
        if local_max_rel < 0.3 or kernel_max_rel < 0.3:  # Need at least 30cm relative height (increased from 10cm)
            return 0.0
        
        # Scale to [0,1] for comparison
        local_normalized = local_relative / local_max_rel
        kernel_normalized = kernel_relative / kernel_max_rel
        
        # Create histograms
        num_bins = 16
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        local_hist, _ = np.histogram(local_normalized.flatten(), bins=bin_edges, density=True)
        kernel_hist, _ = np.histogram(kernel_normalized.flatten(), bins=bin_edges, density=True)
        
        # Normalize to probability distributions
        local_hist = local_hist / (np.sum(local_hist) + 1e-8)
        kernel_hist = kernel_hist / (np.sum(kernel_hist) + 1e-8)
        
        # Cosine similarity
        local_norm = np.linalg.norm(local_hist)
        kernel_norm = np.linalg.norm(kernel_hist)
        
        if local_norm > 1e-8 and kernel_norm > 1e-8:
            similarity = np.dot(local_hist, kernel_hist) / (local_norm * kernel_norm)
            return max(0.0, min(1.0, similarity))
        
        return 0.0
    
    def _compute_geometric_correlation_score(self, local_patch: np.ndarray, kernel: np.ndarray) -> float:
        """Compute geometric feature correlation score"""
        if local_patch.shape != kernel.shape or local_patch.ndim != 3:
            return 0.0
        
        # Use subset of geometric features (skip channel 0 which might be elevation-based)
        if local_patch.shape[2] > 2:
            local_features = local_patch[:, :, 1:3].flatten()
            kernel_features = kernel[:, :, 1:3].flatten()
            
            if len(local_features) > 0 and len(kernel_features) > 0:
                correlation_matrix = np.corrcoef(local_features, kernel_features)
                if correlation_matrix.shape == (2, 2):
                    correlation = correlation_matrix[0, 1]
                    return max(0.0, correlation)  # Convert [-1,1] to [0,1]
        
        return 0.0
    
    def _apply_center_bias(self, coherence_map: np.ndarray) -> np.ndarray:
        """Apply Gaussian center bias to coherence map"""
        h, w = coherence_map.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distance_squared = ((y - center_y) / h)**2 + ((x - center_x) / w)**2
        center_weights = np.exp(-distance_squared / (2 * 0.3**2))
        return 0.5 * coherence_map + 0.5 * (coherence_map * center_weights)
    
    
    # =========================================================================
    # ADAPTIVE SIZE DISCRIMINATION METHODS
    # =========================================================================
    
    def apply_adaptive_size_filter(self, candidates: List[DetectionCandidate], 
                                 elevation_data: np.ndarray) -> List[DetectionCandidate]:
        """
        Apply adaptive size discrimination filter to detection candidates.
        
        Uses statistical distributions from kernel training to filter out candidates
        that don't meet the learned size characteristics of the target structures.
        
        Args:
            candidates: List of detection candidates
            elevation_data: Full elevation array for extracting candidate patches
            
        Returns:
            Filtered list of candidates that pass adaptive size discrimination
        """
        if not candidates:
            return candidates
            
        # Get adaptive thresholds from training statistics
        adaptive_thresholds = self.get_adaptive_thresholds()
        
        filtered_candidates = []
        filter_stats = {
            'total_input': len(candidates),
            'passed_height': 0,
            'passed_volume': 0, 
            'passed_range': 0,
            'passed_all_filters': 0
        }
        
        logger.info(f"🔍 Applying adaptive size filter to {len(candidates)} candidates")
        logger.info(f"   Thresholds - Height: {adaptive_thresholds['min_height_prominence']:.2f}m, "
                   f"Volume: {adaptive_thresholds['min_volume']:.1f}m³, "
                   f"Range: {adaptive_thresholds['min_elevation_range']:.2f}m")
        
        for candidate in candidates:
            # Extract elevation patch around candidate
            patch_size = self.kernel_size * 2  # Use larger patch for better statistics
            half_patch = patch_size // 2
            
            y, x = candidate.location
            y_start = max(0, y - half_patch)
            y_end = min(elevation_data.shape[0], y + half_patch)
            x_start = max(0, x - half_patch) 
            x_end = min(elevation_data.shape[1], x + half_patch)
            
            if (y_end - y_start) < patch_size//2 or (x_end - x_start) < patch_size//2:
                continue  # Skip if patch too small
                
            patch_elevation = elevation_data[y_start:y_end, x_start:x_end]
            
            # Compute size discrimination metrics
            patch_stats = self._analyze_patch_statistics(patch_elevation)
            if not patch_stats:
                continue
                
            # Apply adaptive filters
            height_passed = patch_stats['height_prominence'] >= adaptive_thresholds['min_height_prominence']
            volume_passed = patch_stats['volume_above_base'] >= adaptive_thresholds['min_volume'] 
            range_passed = patch_stats['elevation_range'] >= adaptive_thresholds['min_elevation_range']
            
            # Update statistics
            if height_passed:
                filter_stats['passed_height'] += 1
            if volume_passed:
                filter_stats['passed_volume'] += 1
            if range_passed:
                filter_stats['passed_range'] += 1
                
            # Candidate must pass all critical size filters
            if height_passed and volume_passed and range_passed:
                filter_stats['passed_all_filters'] += 1
                
                # Enhance candidate with size metrics for later ranking
                candidate.size_metrics = {
                    'height_prominence': patch_stats['height_prominence'],
                    'volume_above_base': patch_stats['volume_above_base'],
                    'elevation_range': patch_stats['elevation_range'],
                    'relative_prominence': patch_stats['relative_prominence'],
                    'size_confidence': min(1.0, (
                        (patch_stats['height_prominence'] / adaptive_thresholds['min_height_prominence']) +
                        (patch_stats['volume_above_base'] / adaptive_thresholds['min_volume']) + 
                        (patch_stats['elevation_range'] / adaptive_thresholds['min_elevation_range'])
                    ) / 3.0)
                }
                
                filtered_candidates.append(candidate)
        
        # Log filtering results
        logger.info(f"📊 Size filtering results:")
        logger.info(f"   Input candidates: {filter_stats['total_input']}")
        logger.info(f"   Passed height filter: {filter_stats['passed_height']} ({filter_stats['passed_height']/filter_stats['total_input']*100:.1f}%)")
        logger.info(f"   Passed volume filter: {filter_stats['passed_volume']} ({filter_stats['passed_volume']/filter_stats['total_input']*100:.1f}%)")
        logger.info(f"   Passed range filter: {filter_stats['passed_range']} ({filter_stats['passed_range']/filter_stats['total_input']*100:.1f}%)")
        logger.info(f"   Passed all filters: {filter_stats['passed_all_filters']} ({filter_stats['passed_all_filters']/filter_stats['total_input']*100:.1f}%)")
        
        return filtered_candidates
    
    def rank_candidates_by_adaptive_criteria(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        """
        Rank detection candidates using adaptive size and training-derived criteria.
        
        Args:
            candidates: List of filtered candidates with size metrics
            
        Returns:
            Candidates sorted by comprehensive adaptive ranking score
        """
        if not candidates:
            return candidates
            
        adaptive_thresholds = self.get_adaptive_thresholds()
        training_quality = adaptive_thresholds.get('training_quality', {})
        
        for candidate in candidates:
            if not hasattr(candidate, 'size_metrics'):
                candidate.adaptive_score = candidate.confidence  # Fallback
                continue
            
            size_metrics = candidate.size_metrics
            
            # Base score from detection confidence
            base_score = candidate.confidence
            
            # Size consistency bonus (how well it matches training statistics)
            size_score = size_metrics['size_confidence']
            
            # Training quality adjustment
            consistency_factor = training_quality.get('consistency_factor', 0.5)
            quality_weight = 0.8 + 0.2 * consistency_factor  # Higher weight for consistent training
            
            # Combine scores with adaptive weighting
            candidate.adaptive_score = (
                base_score * 0.6 +           # Detection confidence (60%)
                size_score * 0.3 * quality_weight +  # Size consistency (30%, quality-weighted) 
                candidate.confidence * 0.1    # Detection confidence bonus (10%)
            )
            
            # Additional bonus for exceptional size characteristics
            if (size_metrics['height_prominence'] > adaptive_thresholds['min_height_prominence'] * 1.5 and
                size_metrics['volume_above_base'] > adaptive_thresholds['min_volume'] * 1.2):
                candidate.adaptive_score *= 1.05  # 5% bonus
        
        # Sort by adaptive score (descending)
        ranked_candidates = sorted(candidates, key=lambda c: c.adaptive_score, reverse=True)
        
        logger.info(f"🏆 Ranked {len(candidates)} candidates by adaptive criteria")
        if ranked_candidates:
            logger.info(f"   Top candidate: score={ranked_candidates[0].adaptive_score:.3f}, "
                       f"confidence={ranked_candidates[0].confidence:.3f}")
            if len(ranked_candidates) > 1:
                logger.info(f"   Score range: {ranked_candidates[-1].adaptive_score:.3f} - {ranked_candidates[0].adaptive_score:.3f}")
        
        return ranked_candidates
    
    
    # =========================================================================
    # PERFORMANCE ANALYSIS AND UTILITIES
    # =========================================================================
    
    def analyze_performance(self, positive_scores: List[float], negative_scores: List[float]) -> Dict:
        """Analyze detection performance metrics"""
        pos_mean, neg_mean = np.mean(positive_scores), np.mean(negative_scores)
        pos_std, neg_std = np.std(positive_scores), np.std(negative_scores)
        
        # Signal-to-noise ratio
        snr = abs(pos_mean - neg_mean) / (pos_std + neg_std + 1e-10)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(positive_scores) - 1) * pos_std**2 + 
                             (len(negative_scores) - 1) * neg_std**2) / 
                            (len(positive_scores) + len(negative_scores) - 2))
        cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
        
        # Find optimal threshold
        all_scores = np.concatenate([positive_scores, negative_scores])
        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
        
        best_accuracy = 0
        best_threshold = self.detection_threshold
        
        for threshold in thresholds:
            tp = np.sum(np.array(positive_scores) >= threshold)
            tn = np.sum(np.array(negative_scores) < threshold)
            accuracy = (tp + tn) / (len(positive_scores) + len(negative_scores))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return {
            'positive_mean': pos_mean,
            'negative_mean': neg_mean,
            'signal_to_noise_ratio': snr,
            'cohens_d': cohens_d,
            'optimal_threshold': best_threshold,
            'optimal_accuracy': best_accuracy,
            'current_threshold': self.detection_threshold,
            'separation_strength': 'Strong' if snr > 2.0 else 'Moderate' if snr > 1.0 else 'Weak'
        }
    
    def visualize_detection_results(self, phi0_responses: List[np.ndarray], 
                                  patch_names: List[str] = None, save_path: str = None) -> str:
        """Visualize detection results with enhanced plots"""
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            n_responses = len(phi0_responses)
            if n_responses == 0:
                logger.warning("No detection responses to visualize")
                return None
            
            # Create figure
            fig = plt.figure(figsize=(20, 12))
            cols = min(4, n_responses)
            rows = (n_responses + cols - 1) // cols
            
            for i, response in enumerate(phi0_responses):
                ax = plt.subplot(rows, cols, i + 1)
                
                # Display coherence map
                im = ax.imshow(response, cmap='hot', aspect='equal', vmin=0, vmax=np.max(response))
                
                # Add metrics to title
                title = patch_names[i] if patch_names and i < len(patch_names) else f'Response {i+1}'
                max_score = np.max(response)
                center_y, center_x = response.shape[0]//2, response.shape[1]//2
                center_score = response[center_y, center_x]
                
                is_detected = max_score > self.detection_threshold
                status = "🎯 DETECTED" if is_detected else "❌ Below threshold"
                
                ax.set_title(f'{title}\nMax: {max_score:.3f}, Center: {center_score:.3f}\n{status}', fontsize=10)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='φ⁰ Coherence Score')
                
                # Mark important points
                ax.plot(center_x, center_y, 'b+', markersize=8, markeredgewidth=2, label='Center')
                max_pos = np.unravel_index(np.argmax(response), response.shape)
                ax.plot(max_pos[1], max_pos[0], 'r*', markersize=10, label='Peak')
                
                # Add structure radius circle
                circle = plt.Circle((center_x, center_y), self.structure_radius_px, 
                                  fill=False, color='cyan', linewidth=2, linestyle='--',
                                  label=f'Structure ({self.structure_radius_m}m)')
                ax.add_patch(circle)
                ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                save_path = f'/tmp/phi0_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"🎯 Detection results visualization saved to: {save_path}")
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            return None

    def update_adaptive_thresholds_from_validation(self, positive_results: List[Dict], 
                                                 negative_results: List[Dict]) -> Dict:
        """
        Update adaptive thresholds based on validation performance.
        
        This method analyzes validation results to fine-tune the adaptive thresholds
        for better discrimination between true positives and false positives.
        
        Args:
            positive_results: Results from positive validation (should be detected)
            negative_results: Results from negative validation (should not be detected)
            
        Returns:
            Updated threshold configuration
        """
        logger.info("🔧 Updating adaptive thresholds based on validation performance...")
        
        # Extract scores from validation results
        pos_phi0_scores = [r.get('max_phi0_score', 0) for r in positive_results if 'max_phi0_score' in r]
        neg_phi0_scores = [r.get('max_phi0_score', 0) for r in negative_results if 'max_phi0_score' in r]
        
        pos_geo_scores = [r.get('geometric_score', 0) for r in positive_results if 'geometric_score' in r]
        neg_geo_scores = [r.get('geometric_score', 0) for r in negative_results if 'geometric_score' in r]
        
        if not pos_phi0_scores or not neg_phi0_scores:
            logger.warning("⚠️ Insufficient validation data for threshold updates")
            return self.get_adaptive_thresholds()
        
        # Current thresholds
        current_thresholds = self.get_adaptive_thresholds()
        updated_thresholds = current_thresholds.copy()
        
        # Analyze discrimination performance
        pos_phi0_mean = np.mean(pos_phi0_scores)
        neg_phi0_mean = np.mean(neg_phi0_scores)
        phi0_separation = pos_phi0_mean - neg_phi0_mean
        
        pos_geo_mean = np.mean(pos_geo_scores) if pos_geo_scores else 0
        neg_geo_mean = np.mean(neg_geo_scores) if neg_geo_scores else 0
        geo_separation = pos_geo_mean - neg_geo_mean
        
        logger.info(f"📊 Validation analysis:")
        logger.info(f"   φ⁰ separation: {phi0_separation:.3f} (pos: {pos_phi0_mean:.3f}, neg: {neg_phi0_mean:.3f})")
        logger.info(f"   Geometric separation: {geo_separation:.3f} (pos: {pos_geo_mean:.3f}, neg: {neg_geo_mean:.3f})")
        
        # Adaptive threshold adjustment based on separation quality
        if phi0_separation > 0.2:  # Good separation
            # Can afford to be more selective (increase thresholds)
            adjustment_factor = min(1.1, 1.0 + phi0_separation * 0.2)
            updated_thresholds['min_phi0_threshold'] *= adjustment_factor
            logger.info(f"   Increasing φ⁰ threshold by {(adjustment_factor-1)*100:.1f}% due to good separation")
        elif phi0_separation < 0.1:  # Poor separation
            # Need to be more lenient (decrease thresholds)
            adjustment_factor = max(0.9, 1.0 - (0.1 - phi0_separation) * 0.5)
            updated_thresholds['min_phi0_threshold'] *= adjustment_factor
            logger.info(f"   Decreasing φ⁰ threshold by {(1-adjustment_factor)*100:.1f}% due to poor separation")
        
        if geo_separation > 0.15:  # Good geometric separation
            adjustment_factor = min(1.05, 1.0 + geo_separation * 0.1)
            updated_thresholds['geometric_threshold'] *= adjustment_factor
            logger.info(f"   Increasing geometric threshold by {(adjustment_factor-1)*100:.1f}%")
        elif geo_separation < 0.05:  # Poor geometric separation
            adjustment_factor = max(0.95, 1.0 - (0.05 - geo_separation) * 0.3)
            updated_thresholds['geometric_threshold'] *= adjustment_factor
            logger.info(f"   Decreasing geometric threshold by {(1-adjustment_factor)*100:.1f}%")
        
        # Calculate validation metrics for quality assessment
        false_positive_rate = len(negative_results) / max(1, len(negative_results)) if negative_results else 0
        true_positive_rate = len([r for r in positive_results if r.get('detected', False)]) / max(1, len(positive_results))
        
        # Store validation-derived metadata
        updated_thresholds['validation_derived'] = True
        updated_thresholds['validation_metrics'] = {
            'phi0_separation': phi0_separation,
            'geometric_separation': geo_separation,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'positive_samples': len(positive_results),
            'negative_samples': len(negative_results)
        }
        
        # Clamp thresholds to reasonable bounds
        updated_thresholds['min_phi0_threshold'] = np.clip(updated_thresholds['min_phi0_threshold'], 0.15, 0.6)
        updated_thresholds['geometric_threshold'] = np.clip(updated_thresholds['geometric_threshold'], 0.2, 0.8)
        
        logger.info(f"✅ Updated thresholds:")
        logger.info(f"   φ⁰: {current_thresholds['min_phi0_threshold']:.3f} → {updated_thresholds['min_phi0_threshold']:.3f}")
        logger.info(f"   Geometric: {current_thresholds['geometric_threshold']:.3f} → {updated_thresholds['geometric_threshold']:.3f}")
        
        return updated_thresholds
    
    def visualize_adaptive_threshold_performance(self, positive_results: List[Dict], 
                                               negative_results: List[Dict], 
                                               save_path: str = None) -> None:
        """
        Create comprehensive visualization of adaptive threshold performance.
        
        Args:
            positive_results: Results from positive validation 
            negative_results: Results from negative validation
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from datetime import datetime
        except ImportError:
            logger.warning("⚠️ Matplotlib not available for threshold visualization")
            return
            
        # Extract metrics
        pos_phi0 = [r.get('max_phi0_score', 0) for r in positive_results if 'max_phi0_score' in r]
        neg_phi0 = [r.get('max_phi0_score', 0) for r in negative_results if 'max_phi0_score' in r]
        pos_geo = [r.get('geometric_score', 0) for r in positive_results if 'geometric_score' in r]
        neg_geo = [r.get('geometric_score', 0) for r in negative_results if 'geometric_score' in r]
        
        if not pos_phi0 or not neg_phi0:
            logger.warning("⚠️ Insufficient data for threshold visualization")
            return
            
        # Get current thresholds
        thresholds = self.get_adaptive_thresholds()
        training_quality = thresholds.get('training_quality', {})
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.8])
        
        # 1. φ⁰ Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        bins = np.linspace(0, max(max(pos_phi0, default=0), max(neg_phi0, default=0)), 20)
        ax1.hist(pos_phi0, bins=bins, alpha=0.6, label='Positive', color='green', density=True)
        ax1.hist(neg_phi0, bins=bins, alpha=0.6, label='Negative', color='red', density=True)
        ax1.axvline(thresholds['min_phi0_threshold'], color='black', linestyle='--', 
                   label=f"Threshold: {thresholds['min_phi0_threshold']:.3f}")
        ax1.set_xlabel('φ⁰ Score')
        ax1.set_ylabel('Density')
        ax1.set_title('φ⁰ Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Geometric Score Distribution  
        ax2 = fig.add_subplot(gs[0, 1])
        if pos_geo and neg_geo:
            geo_bins = np.linspace(0, max(max(pos_geo, default=0), max(neg_geo, default=0)), 20)
            ax2.hist(pos_geo, bins=geo_bins, alpha=0.6, label='Positive', color='green', density=True)
            ax2.hist(neg_geo, bins=geo_bins, alpha=0.6, label='Negative', color='red', density=True)
            ax2.axvline(thresholds['geometric_threshold'], color='black', linestyle='--',
                       label=f"Threshold: {thresholds['geometric_threshold']:.3f}")
        ax2.set_xlabel('Geometric Score')
        ax2.set_ylabel('Density') 
        ax2.set_title('Geometric Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter Plot: φ⁰ vs Geometric
        ax3 = fig.add_subplot(gs[0, 2])
        if pos_geo and neg_geo:
            ax3.scatter(pos_phi0[:len(pos_geo)], pos_geo, alpha=0.6, color='green', label='Positive', s=50)
            ax3.scatter(neg_phi0[:len(neg_geo)], neg_geo, alpha=0.6, color='red', label='Negative', s=50)
            
            # Threshold lines
            ax3.axvline(thresholds['min_phi0_threshold'], color='gray', linestyle='--', alpha=0.7)
            ax3.axhline(thresholds['geometric_threshold'], color='gray', linestyle='--', alpha=0.7)
            
            # Decision regions
            ax3.axvspan(thresholds['min_phi0_threshold'], ax3.get_xlim()[1], 
                       ymin=thresholds['geometric_threshold']/ax3.get_ylim()[1], 
                       alpha=0.1, color='green', label='Accept Region')
        
        ax3.set_xlabel('φ⁰ Score')
        ax3.set_ylabel('Geometric Score')
        ax3.set_title('Decision Space Visualization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Statistics Overview
        ax4 = fig.add_subplot(gs[1, 0])
        if self.training_stats['sample_count'] > 0:
            stats_names = ['Height\nProminence', 'Volume\nAbove Base', 'Elevation\nRange', 'Max Height']
            stats_medians = [
                self.training_stats['height_prominence']['median'],
                self.training_stats['volume_above_base']['median'], 
                self.training_stats['elevation_range']['median'],
                self.training_stats['structure_max_height']['median']
            ]
            stats_stds = [
                self.training_stats['height_prominence']['std'],
                self.training_stats['volume_above_base']['std'],
                self.training_stats['elevation_range']['std'], 
                self.training_stats['structure_max_height']['std']
            ]
            
            x_pos = np.arange(len(stats_names))
            ax4.bar(x_pos, stats_medians, yerr=stats_stds, capsize=5, alpha=0.7, color='skyblue')
            ax4.set_xlabel('Training Statistics')
            ax4.set_ylabel('Value')
            ax4.set_title(f'Training Data Statistics (n={self.training_stats["sample_count"]})')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(stats_names)
            ax4.grid(True, alpha=0.3)
        
        # 5. Threshold Comparison 
        ax5 = fig.add_subplot(gs[1, 1])
        threshold_names = ['φ⁰ Threshold', 'Geometric\nThreshold']
        current_vals = [thresholds['min_phi0_threshold'], thresholds['geometric_threshold']]
        
        # Default thresholds for comparison
        default_thresholds = {
            "windmill": [0.35, 0.50],
            "tower": [0.32, 0.45],
            "mound": [0.25, 0.35]
        }.get(self.structure_type, [0.30, 0.42])
        
        x_pos = np.arange(len(threshold_names))
        width = 0.35
        ax5.bar(x_pos - width/2, default_vals := default_thresholds, width, 
               label='Default', alpha=0.7, color='orange')
        ax5.bar(x_pos + width/2, current_vals, width, 
               label='Adaptive', alpha=0.7, color='green')
        
        ax5.set_xlabel('Threshold Type')
        ax5.set_ylabel('Threshold Value')
        ax5.set_title('Adaptive vs Default Thresholds')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(threshold_names)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Calculate metrics
        phi0_separation = np.mean(pos_phi0) - np.mean(neg_phi0)
        geo_separation = np.mean(pos_geo) - np.mean(neg_geo) if pos_geo and neg_geo else 0
        
        detected_positives = sum(1 for r in positive_results if r.get('detected', False))
        false_positives = sum(1 for r in negative_results if r.get('detected', False))
        
        tpr = detected_positives / len(positive_results) if positive_results else 0
        fpr = false_positives / len(negative_results) if negative_results else 0
        
        metrics_text = f"""ADAPTIVE THRESHOLD PERFORMANCE

Training Quality:
  Sample Count: {self.training_stats['sample_count']}
  Height CV: {training_quality.get('height_cv', 0):.3f}
  Consistency: {training_quality.get('consistency_factor', 0):.3f}

Discrimination Power:
  φ⁰ Separation: {phi0_separation:.3f}
  Geometric Separation: {geo_separation:.3f}

Validation Performance:
  True Positive Rate: {tpr:.1%}
  False Positive Rate: {fpr:.1%}
  Positive Samples: {len(positive_results)}
  Negative Samples: {len(negative_results)}

Threshold Adaptation:
  {"✅ Training-Derived" if thresholds.get('training_derived') else "❌ Default"}
  {"✅ Validation-Tuned" if thresholds.get('validation_derived') else "⏸️  No Validation"}
"""
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 7. Recommendation Summary
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Generate recommendations based on performance
        recommendations = []
        if phi0_separation < 0.1:
            recommendations.append("⚠️ Poor φ⁰ separation - consider more training data or feature tuning")
        if geo_separation < 0.05 and pos_geo and neg_geo:
            recommendations.append("⚠️ Poor geometric separation - review geometric features")
        if fpr > 0.3:
            recommendations.append("⚠️ High false positive rate - consider stricter thresholds")
        if tpr < 0.8:
            recommendations.append("⚠️ Low true positive rate - consider more lenient thresholds")
        if not recommendations:
            recommendations.append("✅ Threshold performance looks good!")
            
        rec_text = "RECOMMENDATIONS:\n" + "\n".join(recommendations)
        ax7.text(0.5, 0.5, rec_text, transform=ax7.transAxes, fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9))
        
        plt.suptitle(f'Adaptive Threshold Performance Analysis - {self.structure_type.title()} Detection', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"adaptive_threshold_analysis_{self.structure_type}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Adaptive threshold visualization saved to: {save_path}")
        plt.close()
    
    def detect_with_geometric_validation(self, feature_data: np.ndarray, 
                                       elevation_data: np.ndarray = None,
                                       enable_center_bias: bool = True) -> DetectionResult:
        """
        Detect patterns with comprehensive geometric validation and adaptive thresholds.
        
        This method combines pattern detection with geometric validation and adaptive
        size discrimination to provide a complete detection result with confidence scoring.
        
        Args:
            feature_data: 3D array of octonionic features (h, w, 8)
            elevation_data: Optional raw elevation data for histogram matching
            enable_center_bias: Whether to apply center bias weighting
            
        Returns:
            DetectionResult with detected flag, confidence, and detailed metrics
        """
        try:
            # Step 1: Apply φ⁰ pattern detection
            coherence_map = self.detect_patterns(feature_data, elevation_data, enable_center_bias)
            
            # Step 2: Extract key metrics
            max_score = float(np.max(coherence_map))
            h, w = coherence_map.shape
            center_y, center_x = h // 2, w // 2
            center_score = float(coherence_map[center_y, center_x])
            
            # Step 3: Get adaptive thresholds
            adaptive_thresholds = self.get_adaptive_thresholds()
            phi0_threshold = adaptive_thresholds['min_phi0_threshold']
            geometric_threshold = adaptive_thresholds['geometric_threshold']
            
            # Step 4: Geometric validation
            geometric_score = self._compute_geometric_validation_score(feature_data, elevation_data)
            
            # Step 5: Size discrimination if elevation data available
            size_passed = True
            size_metrics = {}
            if elevation_data is not None:
                size_metrics = self._analyze_patch_statistics(elevation_data)
                if size_metrics:
                    size_passed = (
                        size_metrics['height_prominence'] >= adaptive_thresholds['min_height_prominence'] and
                        size_metrics['volume_above_base'] >= adaptive_thresholds['min_volume'] and
                        size_metrics['elevation_range'] >= adaptive_thresholds['min_elevation_range']
                    )
            
            # Step 6: Combined detection decision
            phi0_passed = max_score >= phi0_threshold
            geometric_passed = geometric_score >= geometric_threshold
            
            # Detection logic: More stringent - require good φ⁰ AND at least one validation
            # Strong φ⁰ needs to be significantly above threshold to bypass validations
            strong_phi0 = max_score >= (phi0_threshold * 1.5)  # 50% above threshold = very strong
            good_phi0 = max_score >= phi0_threshold
            
            if strong_phi0:
                # Very strong φ⁰ score allows detection regardless of other metrics
                detected = True
            elif good_phi0:
                # Good φ⁰ needs BOTH geometric AND size validation (if available)
                if elevation_data is not None:
                    detected = geometric_passed and size_passed
                else:
                    detected = geometric_passed
            else:
                # Below φ⁰ threshold - no detection
                detected = False
            
            # Step 7: Confidence calculation
            confidence = self._calculate_combined_confidence(
                max_score, center_score, geometric_score, size_metrics, adaptive_thresholds
            )
            
            # Step 8: Determine reason
            if detected:
                reason = f"φ⁰={max_score:.3f} (≥{phi0_threshold:.3f})"
                if geometric_passed:
                    reason += f", geo={geometric_score:.3f} (≥{geometric_threshold:.3f})"
                if size_passed and size_metrics:
                    reason += f", size=OK"
            else:
                reasons = []
                if not phi0_passed:
                    reasons.append(f"φ⁰={max_score:.3f} < {phi0_threshold:.3f}")
                if not geometric_passed:
                    reasons.append(f"geometric={geometric_score:.3f} < {geometric_threshold:.3f}")


                if not size_passed:
                    reasons.append("size validation failed")
                reason = ", ".join(reasons)
            
            # Step 9: Build detailed results
            details = {
                'coherence_map': coherence_map,
                'max_phi0_score': max_score,
                'center_phi0_score': center_score,
                'geometric_score': geometric_score,
                'adaptive_thresholds': adaptive_thresholds,
                'size_metrics': size_metrics,
                'size_passed': size_passed,
                'phi0_passed': phi0_passed,
                'geometric_passed': geometric_passed,
                'training_derived': adaptive_thresholds.get('training_derived', False)
            }
            
            return DetectionResult(
                detected=detected,
                confidence=confidence,
                reason=reason,
                max_score=max_score,
                center_score=center_score,
                geometric_score=geometric_score,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                reason=f"Error: {str(e)}",
                max_score=0.0,
                center_score=0.0,
                geometric_score=0.0,
                details={'error': str(e)}
            )
    
    def _compute_geometric_validation_score(self, feature_data: np.ndarray, 
                                          elevation_data: np.ndarray = None) -> float:
        """
        Enhanced geometric validation that actually analyzes structural patterns.
        
        This method validates:
        1. Circular/radial pattern strength at the expected structure scale
        2. Peak centrality and isolation
        3. Elevation profile consistency with circular structures
        4. Feature coherence across the structure region
        
        Args:
            feature_data: 3D array of octonionic features
            elevation_data: Optional elevation data for structural analysis
            
        Returns:
            Geometric validation score [0, 1]
        """
        try:
            h, w = feature_data.shape[:2]
            center_y, center_x = h // 2, w // 2
            radius = self.structure_radius_px
            
            validation_scores = []
            
            # === 1. CIRCULAR SYMMETRY ANALYSIS ===
            if feature_data.shape[2] > 1:
                circular_score = self._analyze_circular_symmetry_pattern(
                    feature_data[:, :, 1], center_y, center_x, radius
                )
                validation_scores.append(circular_score)
            
            # === 2. RADIAL GRADIENT COHERENCE ===
            if feature_data.shape[2] > 2:
                radial_score = self._analyze_radial_gradient_pattern(
                    feature_data[:, :, 2], center_y, center_x, radius
                )
                validation_scores.append(radial_score)
            
            # === 3. ELEVATION-BASED STRUCTURAL VALIDATION ===
            if elevation_data is not None:
                elevation_score = self._analyze_elevation_structure(
                    elevation_data, center_y, center_x, radius
                )
                validation_scores.append(elevation_score)
            
            # === 4. PEAK CENTRALITY AND ISOLATION ===
            if feature_data.shape[2] > 6:
                isolation_score = self._analyze_peak_isolation(
                    feature_data[:, :, 6], center_y, center_x, radius
                )
                validation_scores.append(isolation_score)
            
            # === 5. MULTI-SCALE FEATURE COHERENCE ===
            coherence_score = self._analyze_feature_coherence(
                feature_data, center_y, center_x, radius
            )
            validation_scores.append(coherence_score)
            
            # Weighted combination of validation scores
            if validation_scores:
                # Weight elevation and coherence more heavily as they're more discriminative
                weights = []
                if feature_data.shape[2] > 1: weights.append(0.2)  # Circular symmetry
                if feature_data.shape[2] > 2: weights.append(0.2)  # Radial gradients
                if elevation_data is not None: weights.append(0.3)  # Elevation structure
                if feature_data.shape[2] > 6: weights.append(0.15)  # Peak isolation
                weights.append(0.35)  # Feature coherence (always included)
                
                # Normalize weights
                weights = np.array(weights[:len(validation_scores)])
                weights = weights / np.sum(weights)
                
                final_score = np.average(validation_scores, weights=weights)
                return float(np.clip(final_score, 0.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Geometric validation failed: {e}")
            return 0.0
    
    def _analyze_circular_symmetry_pattern(self, symmetry_feature: np.ndarray, 
                                         center_y: int, center_x: int, radius: int) -> float:
        """Analyze how well the symmetry feature forms a circular pattern"""
        h, w = symmetry_feature.shape
        
        # Create radial sampling pattern
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radial_samples = []
        
        for r in [radius//2, radius, radius*1.2]:  # Multiple radii
            ring_values = []
            for angle in angles:
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    ring_values.append(symmetry_feature[y, x])
            
            if len(ring_values) >= 8:  # Need sufficient samples
                # Measure circular consistency (low variance = good circle)
                ring_mean = np.mean(ring_values)
                ring_std = np.std(ring_values)
                consistency = 1.0 / (1.0 + 2.0 * ring_std / (abs(ring_mean) + 0.1))
                radial_samples.append(consistency)
        
        return np.mean(radial_samples) if radial_samples else 0.0
    
    def _analyze_radial_gradient_pattern(self, gradient_feature: np.ndarray, 
                                       center_y: int, center_x: int, radius: int) -> float:
        """Analyze how well gradients point radially outward from center"""
        h, w = gradient_feature.shape
        
        # Sample gradients in concentric rings
        radial_consistency = []
        
        for r in range(max(3, radius//3), min(radius*2, min(h, w)//2), max(1, radius//4)):
            ring_gradients = []
            angles = np.linspace(0, 2*np.pi, max(8, int(2*np.pi*r)), endpoint=False)
            
            for angle in angles:
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    ring_gradients.append(gradient_feature[y, x])
            
            if len(ring_gradients) >= 6:
                # For radial structures, gradients should be relatively consistent in rings
                ring_std = np.std(ring_gradients)
                ring_mean = abs(np.mean(ring_gradients))
                consistency = 1.0 / (1.0 + ring_std / (ring_mean + 0.05))
                radial_consistency.append(consistency)
        
        return np.mean(radial_consistency) if radial_consistency else 0.0
    
    def _analyze_elevation_structure(self, elevation: np.ndarray, 
                                   center_y: int, center_x: int, radius: int) -> float:
        """Analyze elevation profile for circular structure characteristics"""
        h, w = elevation.shape
        
        # Check if we have a clear central peak
        center_elevation = elevation[center_y, center_x]
        
        # Sample elevations in concentric rings
        ring_means = []
        ring_count = 0
        
        for r in range(max(2, radius//4), min(radius*2, min(h, w)//2), max(1, radius//3)):
            ring_elevations = []
            angles = np.linspace(0, 2*np.pi, max(8, int(2*np.pi*r)), endpoint=False)
            
            for angle in angles:
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < h and 0 <= x < w:
                    ring_elevations.append(elevation[y, x])
            
            if len(ring_elevations) >= 6:
                ring_means.append(np.mean(ring_elevations))
                ring_count += 1
        
        if ring_count < 2:
            return 0.0
        
        # For circular structures, expect elevation to decrease with distance from center
        elevation_trend_score = 0.0
        center_prominence_score = 0.0
        
        if ring_means:
            # Check if center is higher than surrounding rings
            outer_mean = np.mean(ring_means)
            height_prominence = center_elevation - outer_mean
            
            if height_prominence > 0.3:  # At least 30cm prominence
                center_prominence_score = min(1.0, height_prominence / 2.0)  # Normalize by 2m
            
            # Check for monotonic decrease (or at least center > edges)
            if len(ring_means) >= 2:
                inner_mean = ring_means[0] if ring_count > 1 else center_elevation
                outer_mean = ring_means[-1]
                
                if inner_mean > outer_mean:
                    elevation_trend_score = min(1.0, (inner_mean - outer_mean) / 1.5)  # Normalize by 1.5m
        
        # Combine prominence and trend scores
        return (center_prominence_score * 0.7 + elevation_trend_score * 0.3)
    
    def _analyze_peak_isolation(self, isolation_feature: np.ndarray, 
                              center_y: int, center_x: int, radius: int) -> float:
        """Analyze how isolated the central peak is"""
        h, w = isolation_feature.shape
        
        # Check center isolation value
        center_isolation = isolation_feature[center_y, center_x]
        
        # Check isolation in surrounding areas (should be lower for true structures)
        surround_y_start = max(0, center_y - radius*2)
        surround_y_end = min(h, center_y + radius*2 + 1)
        surround_x_start = max(0, center_x - radius*2)
        surround_x_end = min(w, center_x + radius*2 + 1)
        
        # Exclude center region
        surround_patch = isolation_feature[surround_y_start:surround_y_end, surround_x_start:surround_x_end].copy()
        
        # Mask out center area
        yy, xx = np.ogrid[:surround_patch.shape[0], :surround_patch.shape[1]]
        center_yy = center_y - surround_y_start
        center_xx = center_x - surround_x_start
        center_mask = ((yy - center_yy)**2 + (xx - center_xx)**2 <= radius**2)
        
        if np.any(~center_mask):
            surround_isolation = np.mean(surround_patch[~center_mask])
            isolation_contrast = center_isolation - surround_isolation
            
            # Good structures should have higher isolation at center
            return max(0.0, min(1.0, isolation_contrast / 0.5))  # Normalize by 0.5
        
        return center_isolation  # Fallback to raw center isolation
    
    def _analyze_feature_coherence(self, feature_data: np.ndarray, 
                                 center_y: int, center_x: int, radius: int) -> float:
        """Analyze overall coherence of features in the structure region"""
        h, w = feature_data.shape[:2]
        
        # Extract structure region
        struct_y_start = max(0, center_y - radius)
        struct_y_end = min(h, center_y + radius + 1)
        struct_x_start = max(0, center_x - radius)
        struct_x_end = min(w, center_x + radius + 1)
        
        structure_features = feature_data[struct_y_start:struct_y_end, struct_x_start:struct_x_end, :]
        
        # Extract surrounding region for comparison
        surround_y_start = max(0, center_y - radius*2)
        surround_y_end = min(h, center_y + radius*2 + 1)
        surround_x_start = max(0, center_x - radius*2)
        surround_x_end = min(w, center_x + radius*2 + 1)
        
        if (surround_y_end - surround_y_start < radius) or (surround_x_end - surround_x_start < radius):
            return 0.0
        
        surround_features = feature_data[surround_y_start:surround_y_end, surround_x_start:surround_x_end, :]
        
        # Create mask for surrounding area (exclude structure center)
        yy, xx = np.ogrid[:surround_features.shape[0], :surround_features.shape[1]]
        center_yy = center_y - surround_y_start
        center_xx = center_x - surround_x_start
        surround_mask = ((yy - center_yy)**2 + (xx - center_xx)**2 > radius**2)
        
        coherence_scores = []
        
        # Analyze each feature channel
        for f in range(min(4, feature_data.shape[2])):  # Use first 4 channels
            struct_values = structure_features[:, :, f].flatten()
            
            if np.any(surround_mask):
                surround_values = surround_features[:, :, f][surround_mask]
                
                if len(struct_values) > 0 and len(surround_values) > 0:
                    # Structure should have higher mean feature values for good structures
                    struct_mean = np.mean(struct_values)
                    surround_mean = np.mean(surround_values)
                    
                    # Structure should also have more coherent (lower variance) features
                    struct_std = np.std(struct_values)
                    surround_std = np.std(surround_values)
                    
                    # Combine contrast and coherence
                    contrast_score = max(0, struct_mean - surround_mean) / (abs(surround_mean) + 0.1)
                    coherence_score = 1.0 / (1.0 + struct_std / (abs(struct_mean) + 0.05))
                    
                    combined_score = (contrast_score * 0.6 + coherence_score * 0.4)
                    coherence_scores.append(min(1.0, combined_score))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_combined_confidence(self, max_score: float, center_score: float, 
                                     geometric_score: float, size_metrics: Dict, 
                                     adaptive_thresholds: Dict) -> float:
        """
        Calculate combined confidence score using multiple validation metrics.
        
        Args:
            max_score: Maximum φ⁰ coherence score
            center_score: Center φ⁰ coherence score  
            geometric_score: Geometric validation score
            size_metrics: Size discrimination metrics
            adaptive_thresholds: Current adaptive thresholds
            
        Returns:
            Combined confidence score [0, 1]
        """
        try:
            # Base confidence from φ⁰ scores
            phi0_confidence = min(1.0, max_score / adaptive_thresholds['min_phi0_threshold'])
            center_confidence = min(1.0, center_score / adaptive_thresholds['min_phi0_threshold'])
            
            # Geometric confidence
            geo_confidence = min(1.0, geometric_score / adaptive_thresholds['geometric_threshold'])
            
            # Size confidence
            size_confidence = 0.5  # Default if no size metrics
            if size_metrics:
                height_conf = min(1.0, size_metrics['height_prominence'] / adaptive_thresholds['min_height_prominence'])
                volume_conf = min(1.0, size_metrics['volume_above_base'] / adaptive_thresholds['min_volume'])
                range_conf = min(1.0, size_metrics['elevation_range'] / adaptive_thresholds['min_elevation_range'])
                size_confidence = (height_conf + volume_conf + range_conf) / 3.0
            
            # Weighted combination
            confidence = (
                phi0_confidence * 0.4 +       # φ⁰ max score (40%)
                center_confidence * 0.2 +     # φ⁰ center score (20%)
                geo_confidence * 0.2 +        # Geometric validation (20%)
                size_confidence * 0.2         # Size validation (20%)
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.0

    def _compute_structure_volume_metric(self, elevation: np.ndarray) -> np.ndarray:
        """
        Compute structure volume metric for size discrimination.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            2D array with volume-based size discrimination scores [0, 1]
        """
        h, w = elevation.shape
        radius = self.structure_radius_px
        volume_metric = np.zeros_like(elevation)
        
        # Compute volume metric for each point
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                # Define local structure region
                local_patch = elevation[y-radius:y+radius+1, x-radius:x+radius+1]
                center_elevation = elevation[y, x]
                
                # Calculate base elevation from surrounding area
                surround_size = radius + 3
                if (y >= surround_size and y < h - surround_size and 
                    x >= surround_size and x < w - surround_size):
                    
                    # Ring around the structure
                    y_start, y_end = y - surround_size, y + surround_size + 1
                    x_start, x_end = x - surround_size, x + surround_size + 1
                    surround_patch = elevation[y_start:y_end, x_start:x_end]
                    
                    # Create ring mask (exclude inner structure region)
                    yy, xx = np.ogrid[:surround_patch.shape[0], :surround_patch.shape[1]]
                    center_yy, center_xx = surround_size, surround_size
                    ring_mask = ((yy - center_yy)**2 + (xx - center_xx)**2 > radius**2)
                    
                    if np.any(ring_mask):
                        base_elevation = np.mean(surround_patch[ring_mask])
                    else:
                        base_elevation = np.mean(local_patch)
                else:
                    base_elevation = np.mean(local_patch)
                
                # Calculate volume above base for the structure region
                structure_mask = np.ones_like(local_patch, dtype=bool)
                volume_above_base = np.sum(np.maximum(0, local_patch[structure_mask] - base_elevation))
                
                # Normalize volume by expected structure size
                expected_volume = radius**2 * np.pi * 2.0  # Expected volume for 2m height structure
                volume_ratio = volume_above_base / (expected_volume + 1e-6)
                
                # Apply structure-type specific scaling
                if self.structure_type == "windmill":
                    # Windmills should have substantial volume
                    volume_metric[y, x] = min(1.0, volume_ratio / 0.5)  # Normalize to 50% of expected
                elif self.structure_type == "tower":
                    # Towers are more compact but tall
                    volume_metric[y, x] = min(1.0, volume_ratio / 0.3)  # Normalize to 30% of expected
                elif self.structure_type == "mound":
                    # Mounds can be larger and lower
                    volume_metric[y, x] = min(1.0, volume_ratio / 0.8)  # Normalize to 80% of expected
                else:
                    # Generic structures
                    volume_metric[y, x] = min(1.0, volume_ratio / 0.4)  # Normalize to 40% of expected
        
        return volume_metric
    
    def _compute_height_prominence_metric(self, elevation: np.ndarray) -> np.ndarray:
        """
        Compute height prominence metric for size discrimination.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            2D array with height-based size discrimination scores [0, 1]
        """
        h, w = elevation.shape
        radius = self.structure_radius_px
        height_metric = np.zeros_like(elevation)
        
        # Compute local height prominence for each point
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                center_elevation = elevation[y, x]
                
                # Define surrounding area for comparison
                surround_size = radius + 2
                if (y >= surround_size and y < h - surround_size and 
                    x >= surround_size and x < w - surround_size):
                    
                    # Extract surrounding region (excluding center structure)
                    y_start, y_end = y - surround_size, y + surround_size + 1
                    x_start, x_end = x - surround_size, x + surround_size + 1
                    surround_patch = elevation[y_start:y_end, x_start:x_end]
                    
                    # Create mask excluding the structure center
                    yy, xx = np.ogrid[:surround_patch.shape[0], :surround_patch.shape[1]]
                    center_yy, center_xx = surround_size, surround_size
                    surround_mask = ((yy - center_yy)**2 + (xx - center_xx)**2 > (radius//2)**2)
                    
                    if np.any(surround_mask):
                        surround_mean = np.mean(surround_patch[surround_mask])
                        surround_std = np.std(surround_patch[surround_mask])
                    else:
                        surround_mean = center_elevation
                        surround_std = 0.1
                else:
                    # Fallback to local neighborhood
                    local_patch = elevation[y-radius:y+radius+1, x-radius:x+radius+1]
                    surround_mean = np.mean(local_patch)
                    surround_std = np.std(local_patch)
                
                # Calculate height prominence
                height_prominence = center_elevation - surround_mean
                
                # Normalize by local variation and structure-type expectations
                relative_prominence = height_prominence / (surround_std + 0.1)
                
                # Apply structure-type specific scaling
                if self.structure_type == "windmill":
                    # Windmills should have clear height prominence (expect 2-4m)
                    normalized_prominence = height_prominence / 3.0  # Normalize by 3m
                elif self.structure_type == "tower":
                    # Towers have significant height (expect 3-6m)
                    normalized_prominence = height_prominence / 4.0  # Normalize by 4m
                elif self.structure_type == "mound":
                    # Mounds are more subtle (expect 1-2m)
                    normalized_prominence = height_prominence / 1.5  # Normalize by 1.5m
                else:
                    # Generic structures
                    normalized_prominence = height_prominence / 2.0  # Normalize by 2m
                
                # Combine absolute and relative prominence
                height_metric[y, x] = min(1.0, max(0.0, (normalized_prominence + relative_prominence) / 2.0))
        
        return height_metric

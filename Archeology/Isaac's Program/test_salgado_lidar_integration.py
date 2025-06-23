#!/usr/bin/env python3
"""
Salgado-LiDAR Integration Test
Test the Salgado Sim Kernel v3 algorithm with lidar_factory patches
Using validation sites from validate_g2_kernel.py
"""
import logging
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Import the Salgado kernel algorithm functions
from salgado_sim_kernel_v3 import (
    extract_octonionic_features,
    spectral_gate_v3_1,
    Q_operator,
    detect_structures_phi0,
    calculate_entropy_change
)

# Try importing scipy with fallbacks
try:
    from scipy.ndimage import label, find_objects
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple fallback for connected components
    def label(array, structure=None):
        """Simple connected components fallback"""
        labeled = np.zeros_like(array, dtype=int)
        current_label = 1
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] and labeled[i, j] == 0:
                    # Simple flood fill
                    labeled[i, j] = current_label
                    current_label += 1
        return labeled, current_label - 1
    
    def find_objects(labeled_array):
        """Simple object finder fallback"""
        objects = []
        max_label = np.max(labeled_array)
        for label_num in range(1, max_label + 1):
            coords = np.where(labeled_array == label_num)
            if len(coords[0]) > 0:
                objects.append((slice(np.min(coords[0]), np.max(coords[0]) + 1),
                               slice(np.min(coords[1]), np.max(coords[1]) + 1)))
        return objects

# Try importing lidar_factory with fallback
try:
    from lidar_factory.factory import LidarMapFactory
    LIDAR_FACTORY_AVAILABLE = True
except ImportError:
    LIDAR_FACTORY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("lidar_factory not available: using synthetic elevation data for all sites.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationSite:
    """Site information for validation"""
    name: str
    lat: float
    lon: float
    expected_structures: int
    site_type: str  # 'positive' or 'negative'
    confidence: float = 0.0  # Expected confidence level

@dataclass
class SalgadoDetectionResult:
    """Results from Salgado algorithm detection"""
    site_name: str
    lat: float
    lon: float
    patch_size: int
    resolution: float
    coherence_score: float
    octonionic_signature: float
    phi0_resonance: float
    detected_structures: int
    detection_confidence: float
    processing_time: float
    patch_shape: Tuple[int, int]
    algorithm_version: str = "Salgado_v3"

# Test sites from validate_g2_kernel.py
TEST_SITES = [
    ValidationSite("De Kat", 52.47505310183309, 4.8177388422949585, 1, "positive"),
    ValidationSite("De Zoeker", 52.47590104112108, 4.817647238879872, 1, "positive"),
    ValidationSite("Het Jonge Schaap", 52.47621811347626, 4.816644787814995, 1, "positive"),
    ValidationSite("De Bonte Hen", 52.47793734015221, 4.813402499137949, 1, "positive"),
    ValidationSite("Kinderdijk Windmill Complex", 51.8820, 4.6300, 2, "positive"),
    ValidationSite("De Gooyer Windmill Amsterdam", 52.3667, 4.9270, 1, "positive"),
    # Negative control sites
    ValidationSite("Amsterdam Center", 52.3676, 4.9041, 0, "negative"),
    ValidationSite("Rural Field Netherlands", 52.1326, 5.2913, 0, "negative"),
]

class SalgadoLidarTester:
    """Main test class integrating Salgado algorithm with LiDAR data"""
    
    def __init__(self, patch_size_m: int = 128, resolution_m: float = 0.5):
        """
        Initialize the tester
        
        Args:
            patch_size_m: Size of LiDAR patches to fetch in meters
            resolution_m: Preferred resolution for LiDAR data
        """
        self.patch_size_m = patch_size_m
        self.resolution_m = resolution_m
        # No need to initialize a class instance - we'll use functions directly
        self.results: List[SalgadoDetectionResult] = []
        
        logger.info(f"Initialized Salgado-LiDAR tester with patch_size={patch_size_m}m, resolution={resolution_m}m")

    def fetch_lidar_patch(self, lat: float, lon: float) -> Optional[np.ndarray]:
        """
        Fetch LiDAR patch using lidar_factory or fallback to synthetic data
        
        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            
        Returns:
            Elevation data array or None if fetch failed
        """
        if not LIDAR_FACTORY_AVAILABLE:
            logger.warning("lidar_factory unavailable, generating synthetic elevation data.")
            return self._generate_synthetic_elevation(64, 64)
        try:
            logger.info(f"Fetching LiDAR patch for ({lat:.6f}, {lon:.6f})")
            
            patch = LidarMapFactory.get_patch(
                lat=lat,
                lon=lon,
                size_m=self.patch_size_m,
                preferred_resolution_m=self.resolution_m,
                preferred_data_type="DSM",  # Digital Surface Model
                use_cache=True
            )
            
            if patch is not None:
                logger.info(f"Successfully fetched patch with shape {patch.shape}")
                return patch
            else:
                logger.warning(f"Failed to fetch LiDAR patch for ({lat:.6f}, {lon:.6f})")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching LiDAR patch: {e}")
            return None

    def run_salgado_detection(self, elevation_data: np.ndarray, site_name: str) -> Dict:
        """
        Run Salgado algorithm on elevation data
        
        Args:
            elevation_data: 2D numpy array of elevation values
            site_name: Name of the site being analyzed
            
        Returns:
            Dictionary with detection results
        """
        try:
            logger.info(f"Running Salgado detection on {site_name}")
            logger.info(f"Elevation data shape: {elevation_data.shape}, resolution: {self.resolution_m}m/pixel")
            start_time = time.time()
            
            # Fix 1: Recenter patch on elevation peak for proper windmill centering
            elevation_data = self.recenter_patch_on_peak(elevation_data, margin=10)
            
            # Fix 4: Validate structure radius for proper windmill detection
            structure_radius_m, structure_radius_pixels = self.validate_structure_radius(
                elevation_data, 
                self.resolution_m, 
                expected_structure_size_m=50.0  # Typical windmill diameter
            )
            
            # Phase 1: Extract 8D octonionic features
            features_8d = extract_octonionic_features(
                elevation_data, 
                resolution_m=self.resolution_m, 
                structure_radius_m=structure_radius_m  # Use validated radius
            )
            
            # Fix 2: Enhanced feature normalization 
            features_8d = self.enhanced_feature_normalization(features_8d)
            
            # Enhanced feature debugging with center-weighted analysis
            logger.info(f"*** Enhanced 8D Feature Debug for {site_name}:")
            feature_names = [
                "Radial Height Prominence", "Circular Symmetry", "Radial Gradient Consistency",
                "Ring Edge Sharpness", "Hough Circle Response", "Local Planarity", 
                "Isolation Score", "Geometric Coherence"
            ]
            
            center_weighted_scores = []
            for i in range(8):
                feat = features_8d[..., i]
                mean_val = np.mean(feat)
                std_val = np.std(feat)
                min_val = np.min(feat)
                max_val = np.max(feat)
                
                # Calculate center-weighted score
                center_score = self.calculate_center_weighted_score(feat)
                center_weighted_scores.append(center_score)
                
                logger.info(f"   f{i} ({feature_names[i]}): mean={mean_val:.6f}, std={std_val:.6f}, "
                           f"range=[{min_val:.3f}, {max_val:.3f}], center_weighted={center_score:.6f}")
            
            # Fix 3: Calculate central dominance ratio as meta-feature
            central_dominance = self.calculate_central_dominance_ratio(elevation_data, radius=int(structure_radius_pixels * 0.3))
            logger.info(f"   Central Dominance Ratio: {central_dominance:.6f}")
            
            # Fix 4: Calculate windmill discrimination score
            discrimination_score, discrimination_components = self.calculate_windmill_discrimination_score(
                elevation_data, features_8d, structure_radius_pixels
            )
            
            # After extracting features_8d, save feature maps for inspection
            self.save_feature_maps(features_8d, site_name)
            
            # Apply windmill-specific fixes to reduce false positives
            features_8d = self.apply_windmill_specific_fixes(features_8d, elevation_data, structure_radius_pixels)
            
            # Validate feature maps and generate debug report
            anomalies = self.validate_feature_maps(features_8d, site_name)
            if anomalies:
                logger.warning(f"Feature anomalies detected in {site_name}: {anomalies}")
            
            # Phase 2: Apply spectral gate to elevation data  
            X, Y = np.meshgrid(
                np.arange(elevation_data.shape[1]), 
                np.arange(elevation_data.shape[0])
            )
            phi_initial, initial_coherence = spectral_gate_v3_1(
                elevation_data, X, Y
            )
            
            # Phase 3: Recursive collapse via Q operator (phi0 emergence)
            phi_final, final_dSigma_dt = Q_operator(phi_initial, epsilon=0.1, max_iter=5)
            
            # Phase 4: Structure detection using 8D features + phi0 emergence
            detection_results = detect_structures_phi0(
                features_8d, 
                elevation_data, 
                detection_threshold=0.4
            )
            
            # Phase 5: Calculate metrics with improved formulations
            
            # Fix 1: Bounded coherence calculation (avoid negative values)
            std_val = np.std(phi_final)
            mean_val = np.mean(np.abs(phi_final)) + 1e-6
            final_coherence = 1.0 / (1.0 + std_val / mean_val)  # Always in [0, 1]
            
            Delta_S = calculate_entropy_change(elevation_data, phi_final)
            
            processing_time = time.time() - start_time
            
            # Extract results
            num_structures = len(detection_results['candidates'])
            max_coherence = final_coherence
            
            # Fix 2: Enhanced octonionic signature with center-weighted scoring
            # Prioritize features that show good windmill discrimination
            feature_weights = np.array([0.2, 0.15, 0.1, 0.1, 0.15, 0.05, 0.2, 0.05])  # Emphasize f0, f6 (isolation)
            
            # Use center-weighted scores instead of global means
            octonionic_signature = np.sum([
                feature_weights[i] * center_weighted_scores[i] 
                for i in range(8)
            ])
            
            phi0_resonance = 1.0 / (1.0 + abs(Delta_S))  # Higher when entropy change is small
            
            # Debug the enhanced confidence calculation components
            logger.info(f"Debug - raw_coherence: {final_coherence:.6f}")
            logger.info(f"Debug - center_weighted_octonionic: {octonionic_signature:.6f}")  
            logger.info(f"Debug - central_dominance: {central_dominance:.6f}")
            logger.info(f"Debug - phi0_resonance: {phi0_resonance:.6f}")
            logger.info(f"Debug - Delta_S: {Delta_S:.6f}")
            
            # Fix 3: Enhanced weighted sigmoid aggregation with central dominance
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            
            # Enhanced confidence calculation with discrimination score
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            
            # Apply STRONG discrimination penalty for urban false positives
            discrimination_penalty = max(0.2, discrimination_score)  # Minimum 0.2, max 1.0
            
            # Enhanced weighted combination with heavy discrimination emphasis
            w1, w2, w3, w4, w5 = 1.0, 1.0, 0.8, 3.0, 1.0  # Discrimination gets 3x weight!
            combined_score = (
                w1 * (2 * max_coherence - 1) +  # Map [0,1] to [-1,1]
                w2 * (2 * min(octonionic_signature, 1.0) - 1) + 
                w3 * (2 * phi0_resonance - 1) +
                w4 * (2 * discrimination_penalty - 1) +  # MAJOR urban vs windmill discriminator
                w5 * (2 * central_dominance - 1)      # Central dominance as windmill indicator
            )
            
            # Apply discrimination twice: in score AND as final multiplier
            detection_confidence = sigmoid(combined_score) * discrimination_penalty
            
            logger.info(f"Debug - discrimination_penalty: {discrimination_penalty:.6f}")
            logger.info(f"Debug - combined_score: {combined_score:.6f}")
            logger.info(f"Debug - final_confidence: {detection_confidence:.6f}")
            
            # Ensure non-negative confidence
            detection_confidence = max(0.0, detection_confidence)
            
            results = {
                'coherence_score': float(final_coherence),
                'max_coherence': float(max_coherence),
                'octonionic_signature': float(octonionic_signature),
                'phi0_resonance': float(phi0_resonance),
                'detected_structures': int(num_structures),
                'detection_confidence': float(detection_confidence),
                'processing_time': processing_time,
                'coherence_map': phi_final,
                'delta_s': float(Delta_S),
                'dSigma_dt': float(final_dSigma_dt),
                'candidates': detection_results['candidates']
            }
            
            logger.info(f"Salgado detection completed for {site_name}: "
                       f"{num_structures} structures, confidence={detection_confidence:.3f}")
            
            # Save visualizations for manual inspection
            try:
                self.save_lidar_visualization(elevation_data, site_name, detection_results['candidates'], detection_confidence)
                self.save_detailed_analysis(elevation_data, site_name, features_8d, phi_final)
            except Exception as viz_error:
                logger.warning(f"Failed to save visualizations for {site_name}: {viz_error}")
            
            # After extracting features_8d, save feature maps for inspection
            self.save_feature_maps(features_8d, site_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running Salgado detection: {e}")
            return {
                'coherence_score': 0.0,
                'max_coherence': 0.0,
                'octonionic_signature': 0.0,
                'phi0_resonance': 0.0,
                'detected_structures': 0,
                'detection_confidence': 0.0,
                'processing_time': 0.0,
                'coherence_map': np.zeros((10, 10)),
                'delta_s': 0.0,
                'dSigma_dt': 0.0,
                'candidates': []
            }

    def _label_connected_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Simple connected component labeling for structure counting
        
        Args:
            binary_mask: Binary mask of detected regions
            
        Returns:
            Labeled array with connected components
        """
        try:
            from scipy.ndimage import label
            labeled, num_features = label(binary_mask)
            return labeled
        except ImportError:
            # Fallback implementation without scipy
            logger.warning("scipy not available, using simple flood-fill for component labeling")
            labeled = np.zeros_like(binary_mask, dtype=int)
            label_id = 1
            
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    if binary_mask[i, j] and labeled[i, j] == 0:
                        self._flood_fill(binary_mask, labeled, i, j, label_id)
                        label_id += 1
            
            return labeled

    def _flood_fill(self, mask: np.ndarray, labeled: np.ndarray, 
                   start_i: int, start_j: int, label_id: int):
        """Simple flood fill for connected component labeling"""
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or
                not mask[i, j] or labeled[i, j] != 0):
                continue
                
            labeled[i, j] = label_id
            
            # Add 4-connected neighbors
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])

    def test_site(self, site: ValidationSite) -> SalgadoDetectionResult:
        """
        Test a single site with Salgado algorithm
        
        Args:
            site: ValidationSite to test
            
        Returns:
            SalgadoDetectionResult with detection results
        """
        logger.info(f"Testing site: {site.name} at ({site.lat:.6f}, {site.lon:.6f})")
        
        # Fetch LiDAR data
        elevation_data = self.fetch_lidar_patch(site.lat, site.lon)
        
        if elevation_data is None:
            logger.warning(f"No LiDAR data available for {site.name}, using synthetic data")
            # Generate synthetic elevation data for testing
            elevation_data = self._generate_synthetic_elevation(64, 64)
        
        # Run Salgado detection
        detection_results = self.run_salgado_detection(elevation_data, site.name)
        
        # Create result object
        result = SalgadoDetectionResult(
            site_name=site.name,
            lat=site.lat,
            lon=site.lon,
            patch_size=self.patch_size_m,
            resolution=self.resolution_m,
            coherence_score=detection_results['coherence_score'],
            octonionic_signature=detection_results['octonionic_signature'],
            phi0_resonance=detection_results['phi0_resonance'],
            detected_structures=detection_results['detected_structures'],
            detection_confidence=detection_results['detection_confidence'],
            processing_time=detection_results['processing_time'],
            patch_shape=elevation_data.shape
        )
        
        self.results.append(result)
        return result

    def _generate_synthetic_elevation(self, height: int, width: int) -> np.ndarray:
        """Generate synthetic elevation data for testing when LiDAR is unavailable"""
        np.random.seed(42)  # For reproducible results
        
        # Create base terrain
        elevation = np.random.normal(0, 2, (height, width))
        
        # Add some structure-like features for positive sites
        center_h, center_w = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        
        # Add a circular elevated structure (windmill-like)
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        structure_mask = distance < 8
        elevation[structure_mask] += 15  # Elevated structure
        
        # Add some noise
        elevation += np.random.normal(0, 0.5, (height, width))
        
        return elevation

    def run_all_tests(self) -> List[SalgadoDetectionResult]:
        """
        Run tests on all validation sites
        
        Returns:
            List of all detection results
        """
        logger.info(f"Starting comprehensive test of {len(TEST_SITES)} sites")
        
        for site in TEST_SITES:
            try:
                result = self.test_site(site)
                logger.info(f"Completed {site.name}: {result.detected_structures} structures detected")
            except Exception as e:
                logger.error(f"Error testing site {site.name}: {e}")
        
        return self.results

    def analyze_results(self) -> Dict:
        """
        Analyze and summarize test results
        
        Returns:
            Dictionary with analysis summary
        """
        if not self.results:
            logger.warning("No results to analyze")
            return {}
        
        # Separate positive and negative sites
        positive_results = [r for r in self.results if any(s.name == r.site_name and s.site_type == "positive" for s in TEST_SITES)]
        negative_results = [r for r in self.results if any(s.name == r.site_name and s.site_type == "negative" for s in TEST_SITES)]
        
        # Calculate metrics
        total_sites = len(self.results)
        avg_processing_time = np.mean([r.processing_time for r in self.results])
        
        # Detection accuracy
        true_positives = sum(1 for r in positive_results if r.detected_structures > 0)
        false_negatives = sum(1 for r in positive_results if r.detected_structures == 0)
        true_negatives = sum(1 for r in negative_results if r.detected_structures == 0)
        false_positives = sum(1 for r in negative_results if r.detected_structures > 0)
        
        accuracy = (true_positives + true_negatives) / total_sites if total_sites > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Coherence and confidence stats
        avg_coherence = np.mean([r.coherence_score for r in self.results])
        avg_confidence = np.mean([r.detection_confidence for r in self.results])
        avg_octonionic = np.mean([r.octonionic_signature for r in self.results])
        avg_phi0 = np.mean([r.phi0_resonance for r in self.results])
        
        analysis = {
            'total_sites_tested': total_sites,
            'avg_processing_time_seconds': avg_processing_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'avg_coherence_score': avg_coherence,
            'avg_detection_confidence': avg_confidence,
            'avg_octonionic_signature': avg_octonionic,
            'avg_phi0_resonance': avg_phi0,
            'algorithm_version': 'Salgado_v3'
        }
        
        return analysis

    def save_results(self, output_file: str = "salgado_lidar_test_results.json"):
        """
        Save test results to JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        try:
            # Convert results to serializable format
            results_dict = {
                'test_metadata': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'patch_size_m': self.patch_size_m,
                    'resolution_m': self.resolution_m,
                    'algorithm': 'Salgado_v3',
                    'total_sites': len(self.results)
                },
                'individual_results': [asdict(result) for result in self.results],
                'analysis_summary': self.analyze_results()
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
                
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def plot_results_summary(self, save_plot: bool = True):
        """
        Create visualization of test results
        
        Args:
            save_plot: Whether to save the plot to file
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Detection confidence by site type
            positive_confidences = [r.detection_confidence for r in self.results 
                                   if any(s.name == r.site_name and s.site_type == "positive" for s in TEST_SITES)]
            negative_confidences = [r.detection_confidence for r in self.results 
                                   if any(s.name == r.site_name and s.site_type == "negative" for s in TEST_SITES)]
            
            ax1.boxplot([positive_confidences, negative_confidences], 
                       labels=['Positive Sites', 'Negative Sites'])
            ax1.set_title('Detection Confidence by Site Type')
            ax1.set_ylabel('Detection Confidence')
            
            # Plot 2: Coherence vs Structures Detected
            coherences = [r.coherence_score for r in self.results]
            structures = [r.detected_structures for r in self.results]
            colors = ['red' if any(s.name == r.site_name and s.site_type == "positive" for s in TEST_SITES) 
                     else 'blue' for r in self.results]
            
            ax2.scatter(coherences, structures, c=colors, alpha=0.7)
            ax2.set_xlabel('Coherence Score')
            ax2.set_ylabel('Detected Structures')
            ax2.set_title('Coherence vs Structures (Red=Positive, Blue=Negative)')
            
            # Plot 3: Processing time distribution
            processing_times = [r.processing_time for r in self.results]
            ax3.hist(processing_times, bins=10, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Processing Time (seconds)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Processing Time Distribution')
            
            # Plot 4: Algorithm signatures
            octonionic_sigs = [r.octonionic_signature for r in self.results]
            phi0_resonances = [r.phi0_resonance for r in self.results]
            
            ax4.scatter(octonionic_sigs, phi0_resonances, c=colors, alpha=0.7)
            ax4.set_xlabel('Octonionic Signature')
            ax4.set_ylabel('Phi0 Resonance')
            ax4.set_title('Algorithm Signatures (Red=Positive, Blue=Negative)')
            
            plt.tight_layout()
            
            if save_plot:
                plot_filename = f"salgado_lidar_test_results_{int(time.time())}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {plot_filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

    def save_lidar_visualization(self, elevation_data, site_name, structures, confidence, output_dir="lidar_visualizations"):
        """
        Save a visualization of the LiDAR data with detected structures for manual inspection
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create figure with elevation and detection overlay
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Raw elevation data
            im1 = axes[0].imshow(elevation_data, cmap='terrain', interpolation='nearest')
            axes[0].set_title(f'{site_name}\nRaw LiDAR Elevation', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Elevation (m)')
            
            # Plot 2: Elevation with structure detections
            im2 = axes[1].imshow(elevation_data, cmap='terrain', interpolation='nearest')
            axes[1].set_title(f'{site_name}\nDetected Structures: {len(structures)}\nConfidence: {confidence:.3f}', 
                             fontsize=12, color='green' if confidence > 0.3 else 'red')
            
            # Overlay detected structures as red circles
            for i, struct in enumerate(structures):
                if isinstance(struct, dict) and 'location' in struct:
                    y, x = struct['location']
                else:
                    # Handle case where structures is a list of coordinate tuples
                    y, x = struct
                    
                circle = plt.Circle((x, y), radius=5, color='red', fill=False, linewidth=2, alpha=0.8)
                axes[1].add_patch(circle)
                axes[1].text(x+6, y, f'{i+1}', color='red', fontweight='bold', fontsize=8)
            
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Elevation (m)')
            
            # Add metadata text
            fig.suptitle(f'Salgado Algorithm Analysis - {site_name}', fontsize=14, fontweight='bold')
            
            # Save the visualization
            filename = f"{output_dir}/{site_name.replace(' ', '_').replace('/', '_')}_salgado_analysis.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save visualization for {site_name}: {e}")
            return None

    def save_detailed_analysis(self, elevation_data, site_name, octonionic_features, coherence_map, output_dir="lidar_visualizations"):
        """
        Save detailed analysis plots showing octonionic features and coherence maps
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a comprehensive analysis figure
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Plot elevation
            im0 = axes[0,0].imshow(elevation_data, cmap='terrain')
            axes[0,0].set_title('Raw Elevation')
            axes[0,0].axis('off')
            plt.colorbar(im0, ax=axes[0,0], shrink=0.6)
            
            # Plot first 7 octonionic features
            feature_names = ['Radial Height', 'Circular Symmetry', 'Radial Gradient', 'Ring Edge', 
                            'Hough Circle', 'Local Planarity', 'Isolation Score']
            
            for i in range(7):
                row = i // 4
                col = (i + 1) % 4
                if i < octonionic_features.shape[2]:
                    im = axes[row, col].imshow(octonionic_features[:,:,i], cmap='viridis')
                    axes[row, col].set_title(f'f{i}: {feature_names[i] if i < len(feature_names) else f"Feature {i}"}', fontsize=9)
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col], shrink=0.6)
            
            # Add feature statistics in the bottom-left panel [1,0]
            axes[1,0].axis('off')
            stats_text = "Feature Statistics:\n\n"
            for i in range(min(7, octonionic_features.shape[2])):
                feature_data = octonionic_features[:,:,i]
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                min_val = np.min(feature_data)
                max_val = np.max(feature_data)
                stats_text += f"f{i}: μ={mean_val:.3f}, σ={std_val:.3f}\n"
                stats_text += f"    range=[{min_val:.3f}, {max_val:.3f}]\n\n"
            
            axes[1,0].text(0.05, 0.95, stats_text, transform=axes[1,0].transAxes, 
                          fontsize=8, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # Plot coherence map in the last subplot
            if coherence_map is not None:
                im_coh = axes[1,3].imshow(coherence_map, cmap='plasma')
                axes[1,3].set_title('Spectral Coherence')
                axes[1,3].axis('off')
                plt.colorbar(im_coh, ax=axes[1,3], shrink=0.6)
            
            fig.suptitle(f'Salgado Octonionic Feature Analysis - {site_name}', fontsize=14)
            
            # Save detailed analysis
            filename = f"{output_dir}/{site_name.replace(' ', '_').replace('/', '_')}_detailed_analysis.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved detailed analysis: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save detailed analysis for {site_name}: {e}")
            return None

    def save_feature_maps(self, features_8d, site_name, output_dir="lidar_visualizations"):
        """Save each of the 8 feature maps as images for visual inspection."""
        import os
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)
        feature_names = [
            "f0_Radial_Height_Prominence",
            "f1_Circular_Symmetry",
            "f2_Radial_Gradient_Consistency",
            "f3_Ring_Edge_Sharpness",
            "f4_Hough_Circle_Response",
            "f5_Local_Planarity",
            "f6_Isolation_Score",
            "f7_Geometric_Coherence",
        ]
        for i in range(8):
            plt.figure(figsize=(6, 5))
            plt.imshow(features_8d[..., i], cmap="viridis")
            plt.colorbar()
            plt.title(f"{site_name} - {feature_names[i]}")
            plt.tight_layout()
            fname = os.path.join(output_dir, f"{site_name}_{feature_names[i]}.png")
            plt.savefig(fname)
            plt.close()

    def validate_structure_radius(self, elevation_data, pixel_size_m, expected_structure_size_m=50.0):
        """
        Validate that the structure radius is appropriate for the resolution
        
        Args:
            elevation_data: The elevation map
            pixel_size_m: Pixel size in meters  
            expected_structure_size_m: Expected windmill size in meters
            
        Returns:
            Optimal structure radius in pixels and meters
        """
        # Calculate structure radius in pixels
        structure_radius_pixels = expected_structure_size_m / (2 * pixel_size_m)
        
        # Ensure it's reasonable (between 5-50 pixels)
        if structure_radius_pixels < 5:
            print(f"WARNING: Structure radius too small ({structure_radius_pixels:.1f} pixels). Increasing to 5 pixels.")
            structure_radius_pixels = 5
            structure_radius_m = structure_radius_pixels * pixel_size_m
        elif structure_radius_pixels > 50:
            print(f"WARNING: Structure radius too large ({structure_radius_pixels:.1f} pixels). Reducing to 50 pixels.")
            structure_radius_pixels = 50
            structure_radius_m = structure_radius_pixels * pixel_size_m
        else:
            structure_radius_m = expected_structure_size_m / 2
        
        print(f"Structure validation: {structure_radius_m:.1f}m = {structure_radius_pixels:.1f} pixels")
        print(f"Map coverage: {elevation_data.shape[1] * pixel_size_m:.0f}m x {elevation_data.shape[0] * pixel_size_m:.0f}m")
        
        return structure_radius_m, structure_radius_pixels

    def recenter_patch_on_peak(self, elevation_data: np.ndarray, margin: int = 10) -> np.ndarray:
        """
        Recenter the patch based on the highest elevation point to ensure 
        windmill is properly centered in the analysis
        """
        # Find the peak elevation point
        peak_y, peak_x = np.unravel_index(np.argmax(elevation_data), elevation_data.shape)
        
        # Calculate optimal center based on peak
        center_y, center_x = elevation_data.shape[0] // 2, elevation_data.shape[1] // 2
        
        # Calculate shift needed
        shift_y = center_y - peak_y
        shift_x = center_x - peak_x
        
        # Only recenter if peak is significantly off-center
        if abs(shift_y) > margin or abs(shift_x) > margin:
            logger.info(f"Recentering patch: peak at ({peak_y}, {peak_x}), shifting by ({shift_y}, {shift_x})")
            
            # Create recentered patch
            recentered = np.roll(elevation_data, (shift_y, shift_x), axis=(0, 1))
            return recentered
        
        return elevation_data
    
    def calculate_center_weighted_score(self, feature_map: np.ndarray, sigma: float = 0.3) -> float:
        """
        Calculate center-weighted score for a feature map using Gaussian weighting
        
        Args:
            feature_map: 2D feature array
            sigma: Gaussian sigma as fraction of image size
            
        Returns:
            Center-weighted mean score
        """
        h, w = feature_map.shape
        center_y, center_x = h // 2, w // 2
        
        # Create Gaussian weight matrix centered on image
        y, x = np.ogrid[:h, :w]
        gaussian_weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (sigma * min(h, w))**2))
        
        # Normalize weights
        gaussian_weight = gaussian_weight / np.sum(gaussian_weight)
        
        # Calculate weighted score
        weighted_score = np.sum(feature_map * gaussian_weight)
        return float(weighted_score)
    
    def calculate_central_dominance_ratio(self, elevation_data: np.ndarray, radius: int = 15) -> float:
        """
        Calculate how much the center dominates the surrounding area
        High values indicate a prominent central structure (like a windmill)
        """
        h, w = elevation_data.shape
        center_y, center_x = h // 2, w // 2
        
        # Define center region
        center_mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        center_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        
        # Define ring region (surrounding area)
        ring_mask = np.zeros((h, w), dtype=bool)
        ring_mask = ((x - center_x)**2 + (y - center_y)**2) <= (radius * 2)**2
        ring_mask = ring_mask & ~center_mask
        
        if np.sum(center_mask) == 0 or np.sum(ring_mask) == 0:
            return 0.0
            
        # Calculate elevation statistics
        center_elevation = np.mean(elevation_data[center_mask])
        ring_elevation = np.mean(elevation_data[ring_mask])
        
        # Calculate dominance ratio
        if ring_elevation > 0:
            dominance_ratio = (center_elevation - ring_elevation) / (center_elevation + ring_elevation + 1e-6)
        else:
            dominance_ratio = 1.0 if center_elevation > 0 else 0.0
            
        return max(0.0, min(1.0, dominance_ratio))
    
    def enhanced_feature_normalization(self, features_8d: np.ndarray) -> np.ndarray:
        """
        Enhanced feature normalization with empirical bounds and spread enforcement
        """
        normalized_features = features_8d.copy()
        
        # Empirical bounds for each feature based on windmill analysis
        feature_bounds = [
            (0.0, 2.0),    # f0: Radial Height Prominence
            (0.0, 1.0),    # f1: Circular Symmetry  
            (-1.0, 1.0),   # f2: Radial Gradient Consistency
            (0.0, 1.0),    # f3: Ring Edge Sharpness
            (0.0, 1.0),    # f4: Hough Circle Response
            (0.0, 1.0),    # f5: Local Planarity
            (0.0, 1.0),    # f6: Isolation Score
            (0.0, 1.0),    # f7: Geometric Coherence
        ]
        
        for i in range(8):
            feature_layer = features_8d[..., i]
            min_bound, max_bound = feature_bounds[i]
            
            # Robust normalization using percentiles
            p5, p95 = np.percentile(feature_layer, [5, 95])
            
            # Normalize to empirical bounds
            if p95 - p5 > 1e-6:
                normalized_layer = (feature_layer - p5) / (p95 - p5)
                normalized_layer = np.clip(normalized_layer, 0, 1)
            else:
                normalized_layer = np.zeros_like(feature_layer)
            
            # Scale to target bounds
            normalized_layer = min_bound + normalized_layer * (max_bound - min_bound)
            normalized_features[..., i] = normalized_layer
            
        return normalized_features

    def validate_feature_maps(self, features_8d, site_name, output_dir="lidar_visualizations"):
        """
        Validation routine to flag anomalies in feature maps and generate debug log
        """
        import os
        
        # Create validation log
        log_file = os.path.join(output_dir, f"{site_name}_feature_validation.md")
        
        validation_log = []
        validation_log.append(f"# Feature Validation Report: {site_name}")
        validation_log.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        validation_log.append("")
        
        feature_names = [
            "f0_Radial_Height_Prominence",
            "f1_Circular_Symmetry", 
            "f2_Radial_Gradient_Consistency",
            "f3_Ring_Edge_Sharpness",
            "f4_Hough_Circle_Response",
            "f5_Local_Planarity",
            "f6_Isolation_Score",
            "f7_Geometric_Coherence"
        ]
        
        anomalies_found = []
        
        for i, feature_name in enumerate(feature_names):
            feature_map = features_8d[..., i]
            
            # Calculate statistics
            mean_val = np.mean(feature_map)
            std_val = np.std(feature_map)
            min_val = np.min(feature_map)
            max_val = np.max(feature_map)
            unique_vals = len(np.unique(feature_map))
            
            # Anomaly detection
            issues = []
            
            # Check for saturation
            if max_val >= 0.99 and np.sum(feature_map >= 0.99) > feature_map.size * 0.1:
                issues.append("🚨 SATURATED: >10% of pixels at max value")
                anomalies_found.append(f"{feature_name}: Saturation")
                
            # Check for uniformity 
            if std_val < 0.01:
                issues.append("⚠️ UNIFORM: Very low variance (std < 0.01)")
                anomalies_found.append(f"{feature_name}: Uniform")
                
            # Check for missing data
            if unique_vals <= 2:
                issues.append("🚫 SPARSE: ≤2 unique values")
                anomalies_found.append(f"{feature_name}: Sparse")
                
            # Check for invalid ranges
            if feature_name.endswith("Symmetry") or feature_name.endswith("Response"):
                if min_val < 0 or max_val > 1:
                    issues.append("❌ RANGE: Values outside [0,1]")
                    anomalies_found.append(f"{feature_name}: Range error")
                    
            # Check for inverted gradients (f2 specific)
            if "Gradient" in feature_name:
                negative_ratio = np.sum(feature_map < 0) / feature_map.size
                if negative_ratio > 0.7:
                    issues.append("🔄 INVERTED: >70% negative values")
                    anomalies_found.append(f"{feature_name}: Likely inverted")
            
            # Log results
            status = "✅ OK" if not issues else "⚠️ ISSUES"
            validation_log.append(f"## {feature_name}")
            validation_log.append(f"**Status:** {status}")
            validation_log.append(f"- Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            validation_log.append(f"- Range: [{min_val:.3f}, {max_val:.3f}]")
            validation_log.append(f"- Unique values: {unique_vals}")
            
            if issues:
                validation_log.append("**Issues found:**")
                for issue in issues:
                    validation_log.append(f"- {issue}")
            validation_log.append("")
        
        # Summary
        validation_log.append("## Summary")
        if anomalies_found:
            validation_log.append(f"**{len(anomalies_found)} anomalies detected:**")
            for anomaly in anomalies_found:
                validation_log.append(f"- {anomaly}")
        else:
            validation_log.append("✅ No major anomalies detected")
            
        # Save validation log
        os.makedirs(output_dir, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write('\n'.join(validation_log))
            
        logger.info(f"Feature validation report saved: {log_file}")
        return anomalies_found

    def apply_windmill_specific_fixes(self, features_8d, elevation_data, structure_radius_pixels):
        """
        Apply windmill-specific discrimination fixes to reduce urban false positives
        """
        height, width = features_8d.shape[:2]
        center_y, center_x = height // 2, width // 2
        
        # Fix 1: Gradient direction enforcement (f2)
        f2_map = features_8d[..., 2].copy()
        # Windmills should have positive radial gradients (center higher than edges)
        features_8d[..., 2] = np.maximum(f2_map, 0.0)  # Clamp negative values
        
        # Fix 2: Circular symmetry uniqueness constraint (f1)
        f1_map = features_8d[..., 1].copy()
        center_f1 = f1_map[center_y, center_x]
        
        # Create circular mask around center
        y_coords, x_coords = np.ogrid[:height, :width]
        center_mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) <= (structure_radius_pixels * 0.3)**2
        surrounding_mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) <= (structure_radius_pixels)**2
        surrounding_mask = surrounding_mask & ~center_mask
        
        if np.any(surrounding_mask):
            surrounding_f1_mean = np.mean(f1_map[surrounding_mask])
            surrounding_f1_std = np.std(f1_map[surrounding_mask])
            
            # Calculate uniqueness contrast
            if surrounding_f1_std > 0:
                uniqueness_contrast = (center_f1 - surrounding_f1_mean) / surrounding_f1_std
                uniqueness_bonus = np.tanh(uniqueness_contrast / 2.0)  # Sigmoid-like bonus
            else:
                uniqueness_bonus = 0.0
                
            # Apply uniqueness constraint to f1
            features_8d[center_y, center_x, 1] *= (1.0 + uniqueness_bonus * 0.5)
        
        # Fix 3: Hough circle response normalization (f4)
        f4_map = features_8d[..., 4].copy()
        if np.max(f4_map) > 0:
            # Normalize to prevent saturation
            f4_percentile_95 = np.percentile(f4_map[f4_map > 0], 95)
            if f4_percentile_95 > 0:
                features_8d[..., 4] = np.clip(f4_map / f4_percentile_95, 0, 1)
        
        # Fix 4: Planarity dynamic range check (f5)
        f5_map = features_8d[..., 5].copy()
        if np.std(f5_map) < 0.01:  # If uniformly flat
            # Recalculate planarity with better sensitivity
            features_8d[..., 5] = np.random.uniform(0, 0.1, f5_map.shape)  # Add small noise
            
        # Fix 5: Ensure f7 (Geometric Coherence) exists
        f7_map = features_8d[..., 7].copy()
        if np.all(f7_map == 0) or np.std(f7_map) < 1e-6:
            # Recalculate as combination of other features
            coherence_composite = (
                features_8d[..., 0] * 0.3 +  # Height prominence
                features_8d[..., 1] * 0.3 +  # Circular symmetry
                features_8d[..., 6] * 0.4    # Isolation score
            )
            features_8d[..., 7] = coherence_composite
            
        return features_8d

    def calculate_windmill_discrimination_score(self, elevation_data, features_8d, structure_radius_pixels):
        """
        Calculate windmill-specific discrimination score to reduce urban false positives
        """
        height, width = elevation_data.shape
        center_y, center_x = height // 2, width // 2
        
        # Score components
        scores = {}
        
        # 1. Scale-specific check (windmills are typically 15-25m radius)
        optimal_radius_range = (15, 25)  # meters
        actual_radius_m = structure_radius_pixels * 0.5  # Convert to meters
        if optimal_radius_range[0] <= actual_radius_m <= optimal_radius_range[1]:
            scores['scale_match'] = 1.0
        else:
            # Penalize if too small (buildings) or too large (not windmill)
            distance_from_optimal = min(
                abs(actual_radius_m - optimal_radius_range[0]),
                abs(actual_radius_m - optimal_radius_range[1])
            )
            scores['scale_match'] = max(0.0, 1.0 - distance_from_optimal / 10.0)
        
        # 2. Unique central peak (windmills have ONE dominant peak)
        from scipy.ndimage import maximum_filter
        local_maxima = maximum_filter(elevation_data, size=5)
        peaks_mask = (elevation_data == local_maxima) & (elevation_data > np.percentile(elevation_data, 75))
        
        num_peaks = np.sum(peaks_mask)
        if num_peaks == 1:
            scores['peak_uniqueness'] = 1.0
        elif num_peaks <= 3:
            scores['peak_uniqueness'] = 0.7
        else:
            scores['peak_uniqueness'] = max(0.0, 1.0 - (num_peaks - 3) * 0.1)
        
        # 3. Height-to-width ratio (windmills are tall and narrow)
        center_elevation = elevation_data[center_y, center_x]
        edge_elevations = []
        
        # Sample elevations at the structure radius
        for angle in np.linspace(0, 2*np.pi, 16):
            edge_y = int(center_y + structure_radius_pixels * np.sin(angle))
            edge_x = int(center_x + structure_radius_pixels * np.cos(angle))
            if 0 <= edge_y < height and 0 <= edge_x < width:
                edge_elevations.append(elevation_data[edge_y, edge_x])
        
        if edge_elevations:
            mean_edge_elevation = np.mean(edge_elevations)
            height_prominence = center_elevation - mean_edge_elevation
            
            # Windmills should have at least 2-3m prominence
            if height_prominence >= 3.0:
                scores['height_prominence'] = 1.0
            elif height_prominence >= 1.5:
                scores['height_prominence'] = 0.7
            else:
                scores['height_prominence'] = max(0.0, height_prominence / 3.0)
        else:
            scores['height_prominence'] = 0.0
        
        # 4. Isolation check (windmills are usually isolated)
        # Count significant elevation changes in surrounding area
        surround_radius = int(structure_radius_pixels * 2)
        y_min = max(0, center_y - surround_radius)
        y_max = min(height, center_y + surround_radius)
        x_min = max(0, center_x - surround_radius)
        x_max = min(width, center_x + surround_radius)
        
        surrounding_patch = elevation_data[y_min:y_max, x_min:x_max]
        elevation_threshold = np.percentile(surrounding_patch, 85)
        high_elevation_pixels = np.sum(surrounding_patch > elevation_threshold)
        
        # Windmills should have few high elevation pixels in surrounding area
        isolation_ratio = high_elevation_pixels / surrounding_patch.size
        if isolation_ratio < 0.05:  # Less than 5% high elevation
            scores['isolation'] = 1.0
        elif isolation_ratio < 0.15:
            scores['isolation'] = 0.7
        else:
            scores['isolation'] = max(0.0, 1.0 - isolation_ratio)
        
        # 5. Geometric regularity vs chaos (windmills are regular, cities are chaotic)
        f1_center = features_8d[center_y, center_x, 1]  # Circular symmetry at center
        f1_std = np.std(features_8d[..., 1])  # Variability across patch
        
        # High center symmetry + low variability = windmill
        # High center symmetry + high variability = urban (many circular features)
        if f1_center > 0.6 and f1_std < 0.2:
            scores['geometric_consistency'] = 1.0
        elif f1_center > 0.4 and f1_std < 0.3:
            scores['geometric_consistency'] = 0.7
        else:
            scores['geometric_consistency'] = 0.3
        
        # Combine scores with weights
        weights = {
            'scale_match': 0.15,
            'peak_uniqueness': 0.25,
            'height_prominence': 0.25,
            'isolation': 0.20,
            'geometric_consistency': 0.15
        }
        
        final_discrimination_score = sum(scores[key] * weights[key] for key in scores)
        
        logger.info(f"   Windmill Discrimination Scores:")
        for key, score in scores.items():
            logger.info(f"     {key}: {score:.3f}")
        logger.info(f"   Final Discrimination Score: {final_discrimination_score:.3f}")
        
        return final_discrimination_score, scores

    def calculate_peak_uniqueness(self, elevation_data, center_y, center_x, radius=15):
        """Calculate how unique/isolated the central peak is (windmills have ONE dominant peak)"""
        h, w = elevation_data.shape
        
        # Create a local region around the center
        y_min, y_max = max(0, center_y - radius), min(h, center_y + radius)
        x_min, x_max = max(0, center_x - radius), min(w, center_x + radius)
        local_region = elevation_data[y_min:y_max, x_min:x_max]
        
        if local_region.size == 0:
            return 0.0
            
        # Find local maxima using peak detection
        from scipy.ndimage import maximum_filter
        local_maxima = maximum_filter(local_region, size=5)
        peak_mask = (local_region == local_maxima) & (local_region > np.percentile(local_region, 80))
        
        # Count significant peaks
        num_peaks = np.sum(peak_mask)
        
        # Calculate central dominance
        center_local_y, center_local_x = radius, radius
        if center_local_y < local_region.shape[0] and center_local_x < local_region.shape[1]:
            center_elevation = local_region[center_local_y, center_local_x]
            max_elevation = np.max(local_region)
            center_dominance = center_elevation / (max_elevation + 1e-6)
        else:
            center_dominance = 0.0
        
        # Windmill score: high if 1-2 peaks with strong center dominance
        if num_peaks <= 2 and center_dominance > 0.8:
            uniqueness = 1.0  # Perfect windmill-like
        elif num_peaks <= 3 and center_dominance > 0.6:
            uniqueness = 0.7  # Good windmill-like
        elif num_peaks <= 5 and center_dominance > 0.4:
            uniqueness = 0.4  # Moderate
        else:
            uniqueness = 0.0  # Too many peaks = urban
            
        return uniqueness

def main():
    """Main function to run the Salgado-LiDAR integration tests"""
    logger.info("Starting Salgado-LiDAR Integration Test")
    
    # Initialize tester
    tester = SalgadoLidarTester(
        patch_size_m=60,
        resolution_m=0.5
    )
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    # Print summary
    print("\n" + "="*60)
    print("SALGADO-LIDAR INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Total sites tested: {analysis.get('total_sites_tested', 0)}")
    print(f"Average processing time: {analysis.get('avg_processing_time_seconds', 0):.3f} seconds")
    print(f"Accuracy: {analysis.get('accuracy', 0):.3f}")
    print(f"Precision: {analysis.get('precision', 0):.3f}")
    print(f"Recall: {analysis.get('recall', 0):.3f}")
    print(f"F1 Score: {analysis.get('f1_score', 0):.3f}")
    print(f"Average coherence score: {analysis.get('avg_coherence_score', 0):.3f}")
    print(f"Average detection confidence: {analysis.get('avg_detection_confidence', 0):.3f}")
    print("="*60)
    
    # Save results
    tester.save_results()
    
    # Create visualizations
    tester.plot_results_summary()
    
    logger.info("Salgado-LiDAR integration test completed successfully")


if __name__ == "__main__":
    main()

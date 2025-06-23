#!/usr/bin/env python3
"""
G₂ Kernel Validation System with Real AHN4 Data
Enhanced validation using the new G₂ kernel detector from kernel/ directory
Uses real AHN4 elevation data from Google Earth Engine for Dutch windmill detection
"""
import logging
import time # Added import
import json # Added import
import numpy as np # Added import
import matplotlib.pyplot as plt # Added import
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import the new G₂ kernel detector
from kernel.core_detector import G2StructureDetector

# Import LidarMapFactory
from lidar_factory.factory import LidarMapFactory # Changed import

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

# Real Dutch windmill locations from ahn4_data_fetcher (before retirement)
# Each entry: (name, lat, lon, expected_structures, site_type)
_DUTCH_WINDMILL_LOCATIONS_DATA = [
    # Training Sites (first 6) - Real locations from ahn4_data_fetcher
    ("De Kat", 52.47505310183309, 4.8177388422949585, 1, "positive"),
    ("De Zoeker", 52.47590104112108, 4.817647238879872, 1, "positive"),
    ("Het Jonge Schaap", 52.47621811347626, 4.816644787814995, 1, "positive"),
    ("De Bonte Hen", 52.47793734015221, 4.813402499137949, 1, "positive"),
    ("De Gekroonde Poelenburg", 52.474166977199445, 4.817628676751737, 1, "positive"),
    ("De Huisman", 52.47323132365517, 4.816668420518732, 1, "positive"),
    
    # Validation Sites (remaining) - Mix of positive and negative
    ("Kinderdijk Windmill Complex", 51.8820, 4.6300, 2, "positive"),
    ("De Gooyer Windmill Amsterdam", 52.3667, 4.9270, 1, "positive"),
    ("Molen de Adriaan Haarlem", 52.3823, 4.6308, 1, "positive"),
    ("Historic Windmill Leiden", 52.1589, 4.4937, 1, "positive"),
    
    # Negative control sites - areas without windmills
    ("Some Trees Area", 52.628085, 4.762604, 0, "negative"),
    ("Dutch Farmland", 52.2593, 5.2714, 0, "negative"),
    ("Complex Zaanse Schans (not windmill)", 52.4776, 4.8097, 0, "negative"),
    ("Rural Field Near Utrecht", 52.0907, 5.1214, 0, "negative"),
    ("Open Agricultural Area Gelderland", 52.0585, 5.8710, 0, "negative"),
    ("Forest Area Veluwe National Park", 52.1500, 5.9000, 0, "negative"),
]


class G2ValidationSystem:
    """Enhanced validation system using G₂ kernel detector"""
    
    def __init__(self, output_dir: str = "g2_validation_results"):
        """Initialize validation system with LidarMapFactory and Dutch windmill profile"""
        # Load the Dutch windmill profile from templates first
        from kernel.detector_profile import DetectorProfileManager
        self.profile_manager = DetectorProfileManager(templates_dir="kernel/templates")
        self.dutch_profile = self.profile_manager.load_template("dutch_windmill.json")
        
        # Initialize the G₂ detector with the Dutch windmill profile
        self.detector = G2StructureDetector(profile=self.dutch_profile)
        
        # LidarMapFactory is used via static methods, no instance needed here.
        # self.ahn4_fetcher = AHN4DataFetcher() # Removed
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.lidar_dir = self.output_dir / "lidar_images"
        self.results_dir = self.output_dir / "detection_results"
        self.viz_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.lidar_dir, self.results_dir, self.viz_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.validation_results = []
        self.site_paths = {}  # Store paths to saved LiDAR images
        
        logger.info(f"G₂ Validation System initialized with output directory: {self.output_dir}")
        logger.info(f"Using Dutch windmill profile: {self.dutch_profile.name}")
        logger.info(f"Profile version: {self.dutch_profile.version}")
        logger.info(f"Structure type: {self.dutch_profile.structure_type}")
        logger.info(f"Enabled features: {list(self.dutch_profile.get_enabled_features().keys())}")

    def get_training_sites(self) -> List[ValidationSite]:
        """Get training sites with known Dutch windmill structures from internal list"""
        # dutch_locations = self.ahn4_fetcher.get_dutch_windmill_locations() # Removed
        training_sites_data = _DUTCH_WINDMILL_LOCATIONS_DATA[:6]
        training_sites = []
        
        for name, lat, lon, expected, site_type_str in training_sites_data:
            training_sites.append(ValidationSite(
                name=name,
                lat=lat,
                lon=lon,
                expected_structures=expected,
                site_type=site_type_str
            ))
        
        return training_sites

    def get_validation_sites(self) -> List[ValidationSite]:
        """Get validation sites for testing from internal list"""
        # dutch_locations = self.ahn4_fetcher.get_dutch_windmill_locations() # Removed
        validation_sites_data = _DUTCH_WINDMILL_LOCATIONS_DATA[6:]
        validation_sites = []
        
        for name, lat, lon, expected, site_type_str in validation_sites_data:
            validation_sites.append(ValidationSite(
                name=name,
                lat=lat,
                lon=lon,
                expected_structures=expected,
                site_type=site_type_str
            ))
        
        return validation_sites

    def get_ahn4_lidar_data(self, site: ValidationSite, size: Tuple[int, int] = (80, 80)) -> Optional[np.ndarray]:
        """Get LiDAR data for Dutch windmill validation using LidarMapFactory with fallback"""
        # Request 40x40m patch at 0.5m precision = 80x80 pixels (square)
        patch_edge_size_m = 40  # 40m x 40m patch for windmill detection
        target_resolution_m = 0.5  # 0.5m per pixel precision

        logger.info(f"Fetching LiDAR data for {site.name} at {site.lat:.4f}, {site.lon:.4f} using LidarMapFactory")
        logger.info(f"Requesting {patch_edge_size_m}m x {patch_edge_size_m}m patch at {target_resolution_m}m/pixel (expected: 80x80 pixels)")
        
        # Use LidarMapFactory to get the patch
        # Let the factory choose the best available dataset (AHN4 preferred, with fallbacks)
        # We specify preferred_data_type as DSM.
        lidar_data = LidarMapFactory.get_patch(
            lat=site.lat,
            lon=site.lon,
            size_m=patch_edge_size_m,
            preferred_resolution_m=target_resolution_m,
            preferred_data_type="DSM"   # Digital Surface Model
        )
        
        if lidar_data is not None and lidar_data.size > 0:
            logger.info(f"Successfully fetched LiDAR data for {site.name}. Shape: {lidar_data.shape}")
            # Expected shape for 40m at 0.5m/px is (80, 80) - should be square
            expected_shape = (80, 80)
            if lidar_data.shape != expected_shape:
                logger.warning(f"Expected square patch {expected_shape}, got {lidar_data.shape}. This may indicate factory/GEE projection issues.")
            return lidar_data
        else:
            logger.error(f"Failed to fetch LiDAR data for {site.name} using LidarMapFactory.")
            return None

    def save_lidar_image(self, lidar_data: np.ndarray, site: ValidationSite) -> str:
        """Save LiDAR data as image and return path"""
        file_path = self.lidar_dir / f"{site.name}_lidar.npy"
        np.save(file_path, lidar_data)
        
        # Also save as PNG for visual inspection
        plt.figure(figsize=(10, 10))
        plt.imshow(lidar_data, cmap='terrain', origin='lower')
        plt.colorbar(label='Elevation (m)')
        plt.title(f'LiDAR Data: {site.name} ({site.site_type}) - Windmill Detection')
        plt.tight_layout()
        
        png_path = self.lidar_dir / f"{site.name}_lidar.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.site_paths[site.name] = str(file_path)
        logger.info(f"Saved LiDAR data for {site.name} to {file_path}")
        
        return str(file_path)

    def run_detection(self, lidar_data: np.ndarray, site: ValidationSite) -> Dict:
        """Run G₂ kernel detection on LiDAR data with Dutch windmill profile"""
        logger.info(f"Running G₂ Dutch windmill detection on {site.name}")
        logger.info(f"Using profile: {self.dutch_profile.name} v{self.dutch_profile.version}")
        
        try:
            start_time = time.time()
            
            # Create elevation patch for G₂ detector
            from kernel.core_detector import ElevationPatch
            elevation_patch = ElevationPatch(
                elevation_data=lidar_data,
                lat=site.lat,
                lon=site.lon,
                source=f"ahn4_{site.name}",
                resolution_m=self.dutch_profile.geometry.resolution_m,
                patch_size_m=min(self.dutch_profile.geometry.patch_size_m)
            )
            
            # Run detection using the G₂ kernel with Dutch windmill profile
            detection_result = self.detector.detect_structure(elevation_patch)
            
            processing_time = time.time() - start_time
            
            # Log detailed G₂ detection information
            logger.info(f"G₂ Detection Result for {site.name}:")
            logger.info(f"  Detected: {detection_result.detected}")
            logger.info(f"  Confidence: {detection_result.confidence:.4f}")
            logger.info(f"  Final Score: {detection_result.final_score:.4f}")
            logger.info(f"  Base Score: {detection_result.base_score:.4f}")
            
            # Log feature results
            if hasattr(detection_result, 'feature_results') and detection_result.feature_results:
                logger.info("  Feature Scores:")
                for feature_name, feature_result in detection_result.feature_results.items():
                    if hasattr(feature_result, 'score'):
                        logger.info(f"    {feature_name}: {feature_result.score:.4f}")
            
            # Convert G₂ result to our validation format
            results = {
                'detections': [{
                    'confidence': detection_result.confidence,
                    'x': lidar_data.shape[1] // 2,  # Center point for this validation
                    'y': lidar_data.shape[0] // 2,
                    'detected': detection_result.detected,
                    'final_score': detection_result.final_score,
                    'base_score': detection_result.base_score,
                    'feature_scores': {name: getattr(result, 'score', 0.0) 
                                     for name, result in (detection_result.feature_results or {}).items()}
                }] if detection_result.detected else [],
                'processing_time': processing_time,
                'profile_used': self.dutch_profile.name,
                'g2_result': {
                    'detected': detection_result.detected,
                    'confidence': detection_result.confidence,
                    'final_score': detection_result.final_score,
                    'base_score': detection_result.base_score,
                    'feature_scores': {name: getattr(result, 'score', 0.0) 
                                     for name, result in (detection_result.feature_results or {}).items()},
                    'aggregation_method': getattr(detection_result.aggregation_result, 'aggregation_method', 'unknown') if hasattr(detection_result, 'aggregation_result') else 'unknown'
                }
            }
            
            # Create validation result
            validation_result = {
                'site_name': site.name,
                'site_type': site.site_type,
                'expected_structures': site.expected_structures,
                'detected_structures': len(results.get('detections', [])),
                'confidence_scores': [d.get('confidence', 0) for d in results.get('detections', [])],
                'max_confidence': max([d.get('confidence', 0) for d in results.get('detections', [])], default=0),
                'mean_confidence': np.mean([d.get('confidence', 0) for d in results.get('detections', [])]) if results.get('detections') else 0,
                'detection_locations': [(d.get('x', 0), d.get('y', 0)) for d in results.get('detections', [])],
                'processing_time': processing_time,
                'profile_used': self.dutch_profile.name,
                'raw_results': results
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"G₂ Detection failed for {site.name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'site_name': site.name,
                'site_type': site.site_type,
                'expected_structures': site.expected_structures,
                'detected_structures': 0,
                'confidence_scores': [],
                'max_confidence': 0,
                'mean_confidence': 0,
                'detection_locations': [],
                'processing_time': 0,
                'profile_used': self.dutch_profile.name,
                'error': str(e)
            }

    def visualize_detection_results(self, lidar_data: np.ndarray, detection_result: Dict, site: ValidationSite):
        """Create comprehensive visualization of detection results with feature scores"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Original LiDAR data
        im1 = axes[0, 0].imshow(lidar_data, cmap='terrain', origin='lower')
        axes[0, 0].set_title(f'Original LiDAR: {site.name}')
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Y coordinate')
        plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
        
        # Detection overlay
        axes[0, 1].imshow(lidar_data, cmap='terrain', origin='lower', alpha=0.7)
        
        # Overlay detected structures
        for i, (x, y) in enumerate(detection_result['detection_locations']):
            confidence = detection_result['confidence_scores'][i] if i < len(detection_result['confidence_scores']) else 0
            color = 'red' if confidence > 0.5 else 'orange'
            circle = plt.Circle((x, y), 15, fill=False, color=color, linewidth=2)
            axes[0, 1].add_patch(circle)
            axes[0, 1].text(x+20, y+20, f'{confidence:.2f}', color=color, fontweight='bold')
        
        axes[0, 1].set_title(f'Detections: {detection_result["detected_structures"]} found')
        axes[0, 1].set_xlabel('X coordinate')
        axes[0, 1].set_ylabel('Y coordinate')
        
        # Feature scores bar chart
        if 'raw_results' in detection_result and 'g2_result' in detection_result['raw_results']:
            g2_result = detection_result['raw_results']['g2_result']
            if 'feature_scores' in g2_result and g2_result['feature_scores']:
                feature_names = list(g2_result['feature_scores'].keys())
                feature_scores = list(g2_result['feature_scores'].values())
                
                # Create bar chart
                bars = axes[0, 2].bar(range(len(feature_names)), feature_scores, 
                                    color=['red' if score < 0.5 else 'green' for score in feature_scores],
                                    alpha=0.7, edgecolor='black')
                
                # Add value labels on bars
                for bar, score in zip(bars, feature_scores):
                    height = bar.get_height()
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
                
                axes[0, 2].set_xticks(range(len(feature_names)))
                axes[0, 2].set_xticklabels(feature_names, rotation=45, ha='right')
                axes[0, 2].set_ylabel('Feature Score')
                axes[0, 2].set_title('G₂ Feature Scores')
                axes[0, 2].set_ylim(0, 1.1)
                axes[0, 2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
                axes[0, 2].legend()
            else:
                axes[0, 2].text(0.5, 0.5, 'No Feature Scores Available', 
                              ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Feature Scores')
        else:
            axes[0, 2].text(0.5, 0.5, 'No G₂ Results Available', 
                          ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Feature Scores')
        
        # Confidence histogram
        if detection_result['confidence_scores']:
            axes[1, 0].hist(detection_result['confidence_scores'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(site.confidence, color='red', linestyle='--', label=f'Expected: {site.confidence}')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Detection Confidence Distribution')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No detections', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('No Detection Confidence Data')
        
        # Summary statistics
        stats_text = f"""Site: {site.name}
Type: {site.site_type}
Expected Structures: {site.expected_structures}
Detected Structures: {detection_result['detected_structures']}
Max Confidence: {detection_result['max_confidence']:.3f}
Mean Confidence: {detection_result['mean_confidence']:.3f}
Processing Time: {detection_result['processing_time']:.3f}s"""
        
        # Add G₂ specific stats if available
        if 'raw_results' in detection_result and 'g2_result' in detection_result['raw_results']:
            g2_result = detection_result['raw_results']['g2_result']
            stats_text += f"""

G₂ Detection Results:
Detected: {g2_result.get('detected', 'N/A')}
Confidence: {g2_result.get('confidence', 0):.3f}
Final Score: {g2_result.get('final_score', 0):.3f}
Base Score: {g2_result.get('base_score', 0):.3f}"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontfamily='monospace', fontsize=11)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Detection Summary')
        
        # Feature details table
        if 'raw_results' in detection_result and 'g2_result' in detection_result['raw_results']:
            g2_result = detection_result['raw_results']['g2_result']
            if 'feature_scores' in g2_result and g2_result['feature_scores']:
                # Create a simple table of feature scores
                feature_data = []
                for name, score in g2_result['feature_scores'].items():
                    feature_data.append([name, f"{score:.4f}"])
                
                # Create table
                table = axes[1, 2].table(cellText=feature_data, 
                                       colLabels=['Feature', 'Score'],
                                       cellLoc='center',
                                       loc='center',
                                       colWidths=[0.6, 0.4])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2)
                
                # Color code the scores
                for i, (name, score) in enumerate(g2_result['feature_scores'].items()):
                    score_val = float(score)
                    color = 'lightgreen' if score_val >= 0.7 else 'lightcoral' if score_val < 0.3 else 'lightyellow'
                    table[(i+1, 1)].set_facecolor(color)
                
                axes[1, 2].axis('off')
                axes[1, 2].set_title('Feature Score Details')
            else:
                axes[1, 2].text(0.5, 0.5, 'No Feature Details Available', 
                              ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Feature Details')
        else:
            axes[1, 2].text(0.5, 0.5, 'No G₂ Feature Details Available', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Details')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.viz_dir / f"{site.name}_detection_results.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization for {site.name} to {viz_path}")

    def calculate_validation_metrics(self) -> Dict:
        """Calculate overall validation metrics"""
        if not self.validation_results:
            return {}
        
        # Separate positive and negative sites
        positive_sites = [r for r in self.validation_results if r['site_type'] == 'positive']
        negative_sites = [r for r in self.validation_results if r['site_type'] == 'negative']
        
        # Calculate metrics for positive sites
        true_positives = sum(1 for r in positive_sites if r['detected_structures'] > 0)
        false_negatives = sum(1 for r in positive_sites if r['detected_structures'] == 0)
        
        # Calculate metrics for negative sites
        true_negatives = sum(1 for r in negative_sites if r['detected_structures'] == 0)
        false_positives = sum(1 for r in negative_sites if r['detected_structures'] > 0)
        
        # Overall metrics
        total_sites = len(self.validation_results)
        accuracy = (true_positives + true_negatives) / total_sites if total_sites > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'total_sites': total_sites,
            'positive_sites': len(positive_sites),
            'negative_sites': len(negative_sites),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_processing_time': np.mean([r['processing_time'] for r in self.validation_results]),
            'avg_confidence_positive': np.mean([r['mean_confidence'] for r in positive_sites]) if positive_sites else 0,
            'avg_confidence_negative': np.mean([r['mean_confidence'] for r in negative_sites]) if negative_sites else 0
        }

    def run_full_validation(self):
        """Run complete validation pipeline"""
        logger.info("Starting G₂ Kernel Validation Pipeline")
        
        # Get all sites
        training_sites = self.get_training_sites()
        validation_sites = self.get_validation_sites()
        all_sites = training_sites + validation_sites
        
        logger.info(f"Processing {len(all_sites)} sites ({len(training_sites)} training, {len(validation_sites)} validation)")
        
        # Process each site
        for i, site in enumerate(all_sites):
            logger.info(f"Processing site {i+1}/{len(all_sites)}: {site.name}")
            
            # Get real AHN4 LiDAR data
            lidar_data = self.get_ahn4_lidar_data(site)
            
            # Skip site if no real data available
            if lidar_data is None:
                logger.warning(f"Skipping {site.name} - no real AHN4 data available")
                continue
            
            # Save LiDAR image
            lidar_path = self.save_lidar_image(lidar_data, site)
            
            # Run detection
            detection_result = self.run_detection(lidar_data, site)
            
            # Store results
            self.validation_results.append(detection_result)
            
            # Create visualization
            self.visualize_detection_results(lidar_data, detection_result, site)
            
            # Save individual result
            result_path = self.results_dir / f"{site.name}_results.json"
            with open(result_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                def make_json_serializable_local(obj):
                    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
                    if isinstance(obj, dict):
                        return {k: make_json_serializable_local(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_serializable_local(item) for item in obj]
                    elif isinstance(obj, tuple):
                        return [make_json_serializable_local(item) for item in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):  # Other numpy scalars
                        return obj.item()
                    else:
                        return obj
                
                serializable_result = make_json_serializable_local(detection_result)
                json.dump(serializable_result, f, indent=2)
        
        # Calculate overall metrics
        metrics = self.calculate_validation_metrics()
        
        # Check if we have any results
        if not metrics:
            logger.error("No validation results available - all sites failed to fetch LiDAR data")
            logger.error("Please ensure Google Earth Engine is properly initialized or other data sources are available")
            return {
                'validation_metrics': {},
                'site_results': [],
                'site_paths': {},
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'error': 'No LiDAR data could be fetched for any validation sites'
            }
        
        # Save comprehensive results
        comprehensive_results = {
            'validation_metrics': metrics,
            'site_results': self.validation_results,
            'site_paths': self.site_paths,
            'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }
        
        results_path = self.output_dir / f"g2_validation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        def make_json_serializable(obj):
            """Recursively convert numpy types and other non-serializable objects to JSON-serializable types"""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # Other numpy scalars
                return obj.item()
            else:
                return obj

        with open(results_path, 'w') as f:
            serializable_results = make_json_serializable(comprehensive_results)
            json.dump(serializable_results, f, indent=2)
        
        # Create summary visualization only if we have results
        self.create_summary_visualization(metrics)
        
        logger.info(f"Validation complete! Results saved to {results_path}")
        if metrics:
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        else:
            logger.warning("No metrics to display - validation failed to process any sites")
        
        return comprehensive_results

    def create_summary_visualization(self, metrics: Dict):
        """Create summary visualization of validation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Metrics bar chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        bars = axes[0, 0].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Validation Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrix
        cm_data = [
            [metrics['true_positives'], metrics['false_negatives']],
            [metrics['false_positives'], metrics['true_negatives']]
        ]
        
        im = axes[0, 1].imshow(cm_data, cmap='Blues', origin='lower')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_xticklabels(['Predicted Negative', 'Predicted Positive'])
        axes[0, 1].set_yticklabels(['Actual Negative', 'Actual Positive'])
        axes[0, 1].set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[0, 1].text(j, i, cm_data[i][j], ha="center", va="center", color="black", fontweight='bold')
        
        # Site type distribution
        site_types = ['Positive Sites', 'Negative Sites']
        site_counts = [metrics['positive_sites'], metrics['negative_sites']]
        
        axes[0, 2].pie(site_counts, labels=site_types, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0, 2].set_title('Site Distribution')
        
        # Processing time histogram
        processing_times = [r['processing_time'] for r in self.validation_results]
        axes[1, 0].hist(processing_times, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(metrics['avg_processing_time'], color='red', linestyle='--', 
                          label=f'Mean: {metrics["avg_processing_time"]:.3f}s')
        axes[1, 0].set_xlabel('Processing Time (s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Processing Time Distribution')
        axes[1, 0].legend()
        
        # Confidence comparison
        confidence_data = [
            [metrics['avg_confidence_positive'], metrics['avg_confidence_negative']]
        ]
        
        x_pos = np.arange(2)
        bars = axes[1, 1].bar(['Positive Sites', 'Negative Sites'], 
                             [metrics['avg_confidence_positive'], metrics['avg_confidence_negative']], 
                             color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Average Confidence')
        axes[1, 1].set_title('Average Confidence by Site Type')
        
        # Add value labels
        for bar, value in zip(bars, [metrics['avg_confidence_positive'], metrics['avg_confidence_negative']]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Summary statistics text
        summary_text = f"""Validation Summary
        
Total Sites: {metrics['total_sites']}
Positive Sites: {metrics['positive_sites']}
Negative Sites: {metrics['negative_sites']}

Performance Metrics:
Accuracy: {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1 Score: {metrics['f1_score']:.3f}

Average Processing Time: {metrics['avg_processing_time']:.3f}s
"""
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_path = self.viz_dir / f"validation_summary_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary visualization saved to {summary_path}")

def main():
    """Main validation execution"""
    print("G₂ Kernel Validation System")
    print("=" * 50)
    
    # Initialize validation system
    validator = G2ValidationSystem("g2_validation_results")
    
    # Run full validation
    results = validator.run_full_validation()
    
    print("\nValidation Complete!")
    print(f"Results saved to: {validator.output_dir}")
    print(f"LiDAR images: {validator.lidar_dir}")
    print(f"Detection results: {validator.results_dir}")
    print(f"Visualizations: {validator.viz_dir}")

if __name__ == "__main__":
    main()

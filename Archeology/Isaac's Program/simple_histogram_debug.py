#!/usr/bin/env python3
"""
Simple debug script to understand elevation histogram matching failures.
This directly loads data and compares windmill vs negative patch histograms.
Updated to use the new clean phi0_core with apex-centered detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from phi0_core import PhiZeroStructureDetector, ElevationPatch
import logging
import ee
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_earth_engine():
    """Initialize Earth Engine with service account authentication"""
    try:
        service_account_path = "/media/im3/plus/lab4/RE/re_archaeology/sage-striker-294302-b89a8b7e205b.json"
        if os.path.exists(service_account_path):
            credentials = ee.ServiceAccountCredentials(
                'elevation-pattern-detection@sage-striker-294302.iam.gserviceaccount.com',
                service_account_path
            )
            ee.Initialize(credentials)
            logger.info("✅ Earth Engine initialized with service account")
        else:
            ee.Initialize()
            logger.info("✅ Earth Engine initialized with default credentials")
    except Exception as e:
        logger.error(f"❌ Earth Engine initialization failed: {e}")
        raise

def load_elevation_patch(lat, lon, name, buffer_radius_m=20, resolution_m=0.5):
    """Load elevation patch from Earth Engine AHN4 data"""
    try:
        center = ee.Geometry.Point([lon, lat])
        polygon = center.buffer(buffer_radius_m).bounds()
        ahn4_dsm = ee.ImageCollection("AHN/AHN4").select('dsm').median()
        ahn4_dsm = ahn4_dsm.reproject(crs='EPSG:28992', scale=resolution_m)
        rect = ahn4_dsm.sampleRectangle(region=polygon, defaultValue=-9999, properties=[])
        elev_block = rect.get('dsm').getInfo()
        elevation_array = np.array(elev_block, dtype=np.float32)
        elevation_array = np.where(elevation_array == -9999, np.nan, elevation_array)
        
        if np.isnan(elevation_array).all():
            raise Exception(f"No valid elevation data for {name}")
        
        if np.isnan(elevation_array).any():
            mean_val = np.nanmean(elevation_array)
            elevation_array = np.where(np.isnan(elevation_array), mean_val, elevation_array)
        
        patch = ElevationPatch(
            elevation_data=elevation_array,
            coordinates=(lat, lon),
            patch_size_m=buffer_radius_m * 2,
            resolution_m=resolution_m,
            metadata={'source': 'AHN4', 'name': name}
        )
        logger.info(f"✅ Loaded {name}: {elevation_array.shape}, range: {np.min(elevation_array):.2f}-{np.max(elevation_array):.2f}m")
        return patch
    except Exception as e:
        logger.error(f"❌ Failed to load {name}: {e}")
        return None

def compute_elevation_histogram_score(local_elevation, kernel_elevation):
    """Compute elevation histogram similarity score using the same logic as the detection algorithm"""
    
    # Check for meaningful variation
    local_range = np.max(local_elevation) - np.min(local_elevation)
    kernel_range = np.max(kernel_elevation) - np.min(kernel_elevation)
    
    print(f"    Local range: {local_range:.2f}m, Kernel range: {kernel_range:.2f}m")
    
    if local_range < 0.5 or kernel_range < 0.5:
        print("    Insufficient variation")
        return 0.0, None, None
    
    # Apply base elevation removal
    local_relative = local_elevation - np.min(local_elevation)
    kernel_relative = kernel_elevation - np.min(kernel_elevation)
    
    # Normalize to [0,1]
    local_max_rel = np.max(local_relative)
    kernel_max_rel = np.max(kernel_relative)
    
    if local_max_rel <= 0.1 or kernel_max_rel <= 0.1:
        print("    Insufficient relative variation")
        return 0.0, None, None
    
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
    
    # Compute cosine similarity
    local_norm = np.linalg.norm(local_hist)
    kernel_norm = np.linalg.norm(kernel_hist)
    
    if local_norm > 1e-8 and kernel_norm > 1e-8:
        score = np.dot(local_hist, kernel_hist) / (local_norm * kernel_norm)
        score = max(0.0, min(1.0, score))
    else:
        score = 0.0
    
    return score, local_hist, kernel_hist

def compute_elevation_histogram_score_v2(local_elevation, kernel_elevation, detector):
    """Compute elevation histogram similarity using the new detector's method"""
    try:
        # Use the detector's internal method
        score = detector._compute_elevation_histogram_score(local_elevation, kernel_elevation)
        
        # Also compute histograms for visualization
        local_range = np.max(local_elevation) - np.min(local_elevation)
        kernel_range = np.max(kernel_elevation) - np.min(kernel_elevation)
        
        print(f"    Local range: {local_range:.2f}m, Kernel range: {kernel_range:.2f}m")
        
        if local_range < 0.5 or kernel_range < 0.5:
            print("    Insufficient variation")
            return 0.0, None, None
        
        # Apply same normalization as detector
        local_relative = local_elevation - np.min(local_elevation)
        kernel_relative = kernel_elevation - np.min(kernel_elevation)
        
        local_max_rel = np.max(local_relative)
        kernel_max_rel = np.max(kernel_relative)
        
        if local_max_rel < 0.1 or kernel_max_rel < 0.1:
            print("    Insufficient relative variation")
            return 0.0, None, None
        
        local_normalized = local_relative / local_max_rel
        kernel_normalized = kernel_relative / kernel_max_rel
        
        # Create histograms for visualization
        num_bins = 16
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        local_hist, _ = np.histogram(local_normalized.flatten(), bins=bin_edges, density=True)
        kernel_hist, _ = np.histogram(kernel_normalized.flatten(), bins=bin_edges, density=True)
        
        # Normalize to probability distributions
        local_hist = local_hist / (np.sum(local_hist) + 1e-8)
        kernel_hist = kernel_hist / (np.sum(kernel_hist) + 1e-8)
        
        return score, local_hist, kernel_hist
        
    except Exception as e:
        print(f"    Error in histogram computation: {e}")
        return 0.0, None, None

def debug_kernel_construction(detector, training_patches):
    """Debug the kernel construction process to understand histogram differences"""
    print("\n=== DEBUGGING KERNEL CONSTRUCTION ===")
    
    all_elevations = []
    individual_histograms = []
    
    for i, patch in enumerate(training_patches):
        if hasattr(patch, 'elevation_data') and patch.elevation_data is not None:
            elevation = patch.elevation_data
            h, w = elevation.shape
            kernel_size = detector.kernel_size
            
            if h >= kernel_size and w >= kernel_size:
                # Find apex center (same as in learn_pattern_kernel)
                center_y, center_x = detector._find_optimal_kernel_center(elevation)
                half_kernel = kernel_size // 2
                
                if (center_y >= half_kernel and center_y < h - half_kernel and
                    center_x >= half_kernel and center_x < w - half_kernel):
                    
                    start_y = center_y - half_kernel
                    start_x = center_x - half_kernel
                    elevation_kernel = elevation[start_y:start_y+kernel_size, start_x:start_x+kernel_size]
                    
                    if np.std(elevation_kernel) > 0.01:
                        all_elevations.append(elevation_kernel)
                        
                        # Compute histogram for this individual training patch
                        elev_range = np.max(elevation_kernel) - np.min(elevation_kernel)
                        if elev_range >= 0.5:
                            elev_relative = elevation_kernel - np.min(elevation_kernel)
                            elev_max_rel = np.max(elev_relative)
                            if elev_max_rel >= 0.1:
                                elev_normalized = elev_relative / elev_max_rel
                                
                                num_bins = 16
                                bin_edges = np.linspace(0, 1, num_bins + 1)
                                hist, _ = np.histogram(elev_normalized.flatten(), bins=bin_edges, density=True)
                                hist = hist / (np.sum(hist) + 1e-8)
                                individual_histograms.append({
                                    'name': patch.metadata['name'] if patch.metadata else f'Patch_{i}',
                                    'histogram': hist,
                                    'elevation': elevation_kernel,
                                    'range': elev_range
                                })
                                
                                print(f"Training patch {i} ({patch.metadata.get('name', 'Unknown')}): range={elev_range:.2f}m")
    
    if all_elevations:
        # Step 1: Raw average
        raw_average = np.mean(all_elevations, axis=0)
        print(f"Step 1 - Raw average: shape={raw_average.shape}, range={np.max(raw_average)-np.min(raw_average):.2f}m")
        
        # Step 2: Apply G2 symmetrization 
        symmetrized = detector._apply_g2_symmetrization(raw_average)
        print(f"Step 2 - After G2 symmetrization: range={np.max(symmetrized)-np.min(symmetrized):.2f}m")
        
        # Compare histograms
        print(f"\nHistogram comparison:")
        print(f"Number of individual training histograms: {len(individual_histograms)}")
        
        # Compute final kernel histogram
        kernel_range = np.max(symmetrized) - np.min(symmetrized)
        if kernel_range >= 0.5:
            kernel_relative = symmetrized - np.min(symmetrized)
            kernel_max_rel = np.max(kernel_relative)
            if kernel_max_rel >= 0.1:
                kernel_normalized = kernel_relative / kernel_max_rel
                
                num_bins = 16
                bin_edges = np.linspace(0, 1, num_bins + 1)
                kernel_hist, _ = np.histogram(kernel_normalized.flatten(), bins=bin_edges, density=True)
                kernel_hist = kernel_hist / (np.sum(kernel_hist) + 1e-8)
                
                print(f"Final kernel histogram computed successfully")
                
                # DETAILED DEBUGGING: Step by step analysis
                print(f"\n=== DETAILED STEP-BY-STEP ANALYSIS ===")
                
                # Show raw individual elevations vs final kernel
                print(f"Raw elevation analysis:")
                for idx, indiv in enumerate(individual_histograms):
                    raw_elev = all_elevations[idx]
                    print(f"  Training {idx} ({indiv['name']}):")
                    print(f"    Raw elevation range: {np.min(raw_elev):.2f} to {np.max(raw_elev):.2f}m")
                    print(f"    Raw elevation std: {np.std(raw_elev):.2f}m")
                    
                    # Compare raw vs symmetrized
                    raw_relative = raw_elev - np.min(raw_elev)
                    raw_normalized = raw_relative / np.max(raw_relative)
                    raw_hist, _ = np.histogram(raw_normalized.flatten(), bins=bin_edges, density=True)
                    raw_hist = raw_hist / (np.sum(raw_hist) + 1e-8)
                    
                    raw_vs_final = np.dot(raw_hist, kernel_hist) / (
                        np.linalg.norm(raw_hist) * np.linalg.norm(kernel_hist) + 1e-8)
                    print(f"    Raw vs final kernel similarity: {raw_vs_final:.4f}")
                    
                    # Check if processing changes histogram significantly
                    processed_vs_raw = np.dot(indiv['histogram'], raw_hist) / (
                        np.linalg.norm(indiv['histogram']) * np.linalg.norm(raw_hist) + 1e-8)
                    print(f"    Processed vs raw similarity: {processed_vs_raw:.4f}")
                
                print(f"\nFinal kernel elevation analysis:")
                print(f"  Range: {np.min(symmetrized):.2f} to {np.max(symmetrized):.2f}m")
                print(f"  Std: {np.std(symmetrized):.2f}m")
                
                # Check if the issue is in normalization or histogram computation
                print(f"\nHistogram computation validation:")
                kernel_relative_check = symmetrized - np.min(symmetrized)
                kernel_normalized_check = kernel_relative_check / np.max(kernel_relative_check)
                print(f"  Kernel normalized range: {np.min(kernel_normalized_check):.3f} to {np.max(kernel_normalized_check):.3f}")
                print(f"  Kernel normalized std: {np.std(kernel_normalized_check):.3f}")
                
                # Compare with individual training histograms
                if individual_histograms:
                    # Average training histogram (calculate first)
                    avg_training_hist = np.mean([h['histogram'] for h in individual_histograms], axis=0)
                    
                    # Show actual histogram values
                    print(f"\nActual histogram comparison (first few bins):")
                    print(f"  Final kernel hist: {kernel_hist[:5]}")
                    print(f"  Avg training hist: {avg_training_hist[:5]}")
                    print(f"  Training 0 hist: {individual_histograms[0]['histogram'][:5]}")
                    
                    print(f"\nComparing final kernel vs individual training patches:")
                    for indiv in individual_histograms:
                        similarity = np.dot(kernel_hist, indiv['histogram']) / (
                            np.linalg.norm(kernel_hist) * np.linalg.norm(indiv['histogram']) + 1e-8)
                        print(f"  {indiv['name']}: similarity = {similarity:.4f}")
                    
                    avg_similarity = np.dot(kernel_hist, avg_training_hist) / (
                        np.linalg.norm(kernel_hist) * np.linalg.norm(avg_training_hist) + 1e-8)
                    print(f"  Average training: similarity = {avg_similarity:.4f}")
                    
                    return {
                        'individual_histograms': individual_histograms,
                        'kernel_histogram': kernel_hist,
                        'avg_training_histogram': avg_training_hist,
                        'raw_average': raw_average,
                        'symmetrized': symmetrized
                    }
                
                # CRITICAL TEST: Extract same region from same training patch and compare
                print(f"\n=== CRITICAL CONSISTENCY TEST ===")
                if individual_histograms and training_patches:
                    # Take the first training patch and extract the same region again
                    test_patch = training_patches[0]
                    test_elevation = test_patch.elevation_data
                    test_center_y, test_center_x = detector._find_optimal_kernel_center(test_elevation)
                    
                    # Extract the same region as was used in training
                    half_kernel = detector.kernel_size // 2
                    if (test_center_y >= half_kernel and test_center_y < test_elevation.shape[0] - half_kernel and
                        test_center_x >= half_kernel and test_center_x < test_elevation.shape[1] - half_kernel):
                        
                        start_y = test_center_y - half_kernel
                        start_x = test_center_x - half_kernel
                        extracted_region = test_elevation[start_y:start_y+detector.kernel_size, 
                                                        start_x:start_x+detector.kernel_size]
                        
                        # Compute histogram using the same method as detector
                        test_score = detector._compute_elevation_histogram_score(extracted_region, symmetrized)
                        
                        # Also compute histogram manually for comparison
                        test_range = np.max(extracted_region) - np.min(extracted_region)
                        if test_range >= 0.5:
                            test_relative = extracted_region - np.min(extracted_region)
                            test_max_rel = np.max(test_relative)
                            if test_max_rel >= 0.1:
                                test_normalized = test_relative / test_max_rel
                                test_hist, _ = np.histogram(test_normalized.flatten(), bins=bin_edges, density=True)
                                test_hist = test_hist / (np.sum(test_hist) + 1e-8)
                                
                                manual_similarity = np.dot(test_hist, kernel_hist) / (
                                    np.linalg.norm(test_hist) * np.linalg.norm(kernel_hist) + 1e-8)
                                
                                original_similarity = individual_histograms[0]['histogram']
                                orig_vs_reextracted = np.dot(original_similarity, test_hist) / (
                                    np.linalg.norm(original_similarity) * np.linalg.norm(test_hist) + 1e-8)
                                
                                print(f"Consistency test for {test_patch.metadata.get('name', 'Unknown')}:")
                                print(f"  Detector score: {test_score:.4f}")
                                print(f"  Manual similarity: {manual_similarity:.4f}")
                                print(f"  Original vs re-extracted: {orig_vs_reextracted:.4f}")
                                print(f"  Expected (from training): {individual_histograms[0]['histogram']} vs kernel similarity")
                                
                                if abs(test_score - manual_similarity) > 0.01:
                                    print(f"  ⚠️  DISCREPANCY: Detector and manual methods differ!")
                                if orig_vs_reextracted < 0.95:
                                    print(f"  ⚠️  INCONSISTENCY: Re-extraction differs from original!")
    
    return None

def debug_elevation_histograms():
    """Debug elevation histogram matching with apex-centered extraction for fair comparison."""
    
    print("=== DEBUGGING ELEVATION HISTOGRAM MATCHING (Apex-Centered Comparison) ===")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Initialize new detection system with apex-centered kernel learning
    # Use smaller kernel size to allow apex centering with 82x82 patches
    detector = PhiZeroStructureDetector(
        resolution_m=0.5, 
        kernel_size=21,  # Smaller kernel allows apex centering in 82x82 patches
        structure_type="windmill"
    )
    
    # Test windmill and negative locations
    test_locations = [
        # Windmills (should score high)
        {"name": "De Kat", "lat": 52.47505310183309, "lon": 4.8177388422949585, "is_windmill": True},
        {"name": "De Zoeker", "lat": 52.47590104112108, "lon": 4.817647238879872, "is_windmill": True},  # moved 5m north
        {"name": "Het Klaverblad", "lat": 52.4775485810242, "lon": 4.813724798553969, "is_windmill": True},
        {"name": "Test Site - A False Positive", "lat": 52.483710, "lon": 4.810826,  "is_windmill": False},
        {"name": "Test Site - B False Positive", "lat": 51.869760, "lon": 4.632234,  "is_windmill": False},
        # Negative patches (should score low)
        {"name": "De Kat East", "lat": 52.47505310183309, "lon": 4.8182, "is_windmill": False},  # 50m east
        {"name": "Het Klaverblad East", "lat": 52.4775485810242, "lon": 4.8142, "is_windmill": False},  # 50m east
    ]
    
    # Load training data to create kernel
    print("\n1. Loading training windmills...")
    training_patches = []
    for loc in test_locations:
        if loc["is_windmill"]:
            patch = load_elevation_patch(loc["lat"], loc["lon"], loc["name"], buffer_radius_m=40)
            if patch:
                training_patches.append(patch)
    
    if not training_patches:
        print("❌ Failed to load training data")
        return
    
    # Construct kernel using apex-centered approach
    print("   Using apex-centered kernel extraction...")
    detector.learn_pattern_kernel(training_patches, use_apex_center=True)
    print(f"✅ Kernel constructed from {len(training_patches)} windmills with apex centering")
    
    # Debug kernel construction process
    debug_info = debug_kernel_construction(detector, training_patches)
    
    # Get the elevation kernel
    if not hasattr(detector, 'elevation_kernel') or detector.elevation_kernel is None:
        print("❌ No elevation kernel available")
        return
    
    kernel_elevation = detector.elevation_kernel
    print(f"   Kernel elevation shape: {kernel_elevation.shape}")
    print(f"   Kernel elevation range: {np.min(kernel_elevation):.2f}-{np.max(kernel_elevation):.2f}m")
    
    # Test all locations with apex-centered extraction for fair comparison
    results = []
    
    print("\n2. Testing locations with apex-centered extraction...")
    for loc in test_locations:
        print(f"\nTesting: {loc['name']} ({'windmill' if loc['is_windmill'] else 'negative'})")
        
        # Load patch
        patch = load_elevation_patch(loc["lat"], loc["lon"], loc["name"], buffer_radius_m=40)
        if not patch:
            continue
        
        elevation = patch.elevation_data
        h, w = elevation.shape
        kernel_size = detector.kernel_size
        
        # Show apex information for ALL patches (windmills and negatives)
        apex_y, apex_x = detector._find_optimal_kernel_center(elevation)
        geometric_center_y, geometric_center_x = h//2, w//2
        apex_elevation = elevation[apex_y, apex_x]
        center_elevation_val = elevation[geometric_center_y, geometric_center_x]
        
        print(f"  Patch shape: {elevation.shape}")
        print(f"  Apex location: ({apex_y}, {apex_x}) = {apex_elevation:.2f}m")
        print(f"  Geometric center: ({geometric_center_y}, {geometric_center_x}) = {center_elevation_val:.2f}m")
        print(f"  Apex vs center difference: {apex_elevation - center_elevation_val:.2f}m")
        
        # Extract kernel region using APEX centering for ALL patches (fair comparison)
        center_elevation = None
        extraction_method = "unknown"
        
        if h >= kernel_size and w >= kernel_size:
            # Try apex-centered extraction first
            half_kernel = kernel_size // 2
            
            # Check if we can extract a full kernel around the apex
            if (apex_y >= half_kernel and apex_y < h - half_kernel and
                apex_x >= half_kernel and apex_x < w - half_kernel):
                start_y = apex_y - half_kernel
                start_x = apex_x - half_kernel
                center_elevation = elevation[start_y:start_y+kernel_size, start_x:start_x+kernel_size]
                extraction_method = "apex-centered"
                print(f"  ✅ Using apex-centered extraction at ({apex_y}, {apex_x})")
            else:
                # Fallback to geometric center if apex is too close to edge
                start_y = (h - kernel_size) // 2
                start_x = (w - kernel_size) // 2
                center_elevation = elevation[start_y:start_y+kernel_size, start_x:start_x+kernel_size]
                extraction_method = "geometric-centered (apex too close to edge)"
                print(f"  ⚠️  Apex too close to edge, using geometric center")
        else:
            # Patch is smaller than kernel - use full patch
            center_elevation = elevation
            extraction_method = "full-patch (too small)"
            print(f"  ⚠️  Patch too small for kernel, using full patch")
        
        print(f"  Extraction method: {extraction_method}")
        print(f"  Extracted region shape: {center_elevation.shape}")
        print(f"  Extracted elevation range: {np.min(center_elevation):.2f}-{np.max(center_elevation):.2f}m")
        
        # Compute histogram score using the same method as the detector
        score, local_hist, kernel_hist = compute_elevation_histogram_score_v2(center_elevation, kernel_elevation, detector)
        
        print(f"  📊 Elevation histogram score: {score:.4f}")
        
        # Store results
        if local_hist is not None and kernel_hist is not None:
            results.append({
                'name': loc['name'],
                'is_windmill': loc['is_windmill'],
                'score': score,
                'local_hist': local_hist,
                'kernel_hist': kernel_hist,
                'elevation': center_elevation,
                'elevation_range': np.max(center_elevation) - np.min(center_elevation)
            })
    
    # Analyze results
    if results:
        print(f"\n3. Results Summary:")
        print("   Name                    | Type     | Score   | Range")
        print("   " + "-"*55)
        
        windmill_scores = []
        negative_scores = []
        
        for result in results:
            type_str = "Windmill" if result['is_windmill'] else "Negative"
            print(f"   {result['name']:22} | {type_str:8} | {result['score']:.4f} | {result['elevation_range']:.2f}m")
            
            if result['is_windmill']:
                windmill_scores.append(result['score'])
            else:
                negative_scores.append(result['score'])
        
        # Problem analysis
        if windmill_scores and negative_scores:
            print(f"\n4. Problem Analysis:")
            print(f"   Average windmill score: {np.mean(windmill_scores):.4f}")
            print(f"   Average negative score: {np.mean(negative_scores):.4f}")
            print(f"   Score difference: {np.mean(windmill_scores) - np.mean(negative_scores):.4f}")
            
            if np.mean(negative_scores) >= np.mean(windmill_scores) * 0.9:
                print("   *** PROBLEM: Negatives score nearly as high as windmills! ***")
                print("   This explains the high false positive rate.")
            else:
                print("   ✅ Good separation between windmills and negatives")
        
        # Create visualization
        if len(results) >= 2:
            fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 8))
            if len(results) == 1:
                axes = axes.reshape(2, 1)
            
            for i, result in enumerate(results):
                # Plot elevation pattern
                axes[0, i].imshow(result['elevation'], cmap='terrain')
                title_color = 'green' if result['is_windmill'] else 'red'
                axes[0, i].set_title(f"{result['name']}\nScore: {result['score']:.4f}", 
                                    color=title_color, fontsize=10)
                axes[0, i].axis('off')
                
                # Plot histograms
                bins = np.arange(len(result['local_hist']))
                width = 0.4
                axes[1, i].bar(bins - width/2, result['local_hist'], width, label='Local', alpha=0.7)
                axes[1, i].bar(bins + width/2, result['kernel_hist'], width, label='Kernel', alpha=0.7)
                axes[1, i].set_title('Elevation Histograms')
                axes[1, i].legend()
                axes[1, i].set_xlabel('Normalized Elevation Bin')
                axes[1, i].set_ylabel('Density')
            
            plt.tight_layout()
            plt.savefig('/media/im3/plus/lab4/RE/re_archaeology/simple_histogram_debug_clean.png', dpi=150, bbox_inches='tight')
            print(f"\n5. Visualization saved to simple_histogram_debug_clean.png")
    
    else:
        print("❌ No valid results to analyze")

if __name__ == "__main__":
    debug_elevation_histograms()

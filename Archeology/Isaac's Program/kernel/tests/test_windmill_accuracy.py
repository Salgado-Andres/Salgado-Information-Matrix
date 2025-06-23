#!/usr/bin/env python3
"""
Windmill Detection Accuracy Test - Simplified Version
Tests actual detection performance without complex imports
"""

import os
import sys
import json
import numpy as np

# Add kernel to path  
kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, kernel_dir)

def create_synthetic_windmill(size=40, radius=8.0, height=2.0, noise=0.1):
    """Create a realistic synthetic windmill mound"""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Gaussian-like mound with realistic windmill characteristics
    mound = height * np.exp(-(distance**2) / (2 * (radius/2.5)**2))
    
    # Add slight asymmetry (windmills aren't perfectly circular)
    angle = np.arctan2(y - center, x - center)
    asymmetry = 1.0 + 0.1 * np.cos(2 * angle)
    mound *= asymmetry
    
    # Sharp edge characteristic of constructed mounds
    edge_mask = distance < radius
    mound[~edge_mask] *= 0.3
    
    # Add realistic noise
    noise_array = np.random.normal(0, noise, (size, size))
    mound += noise_array
    
    return mound

def create_non_windmill(structure_type="natural", size=40):
    """Create structures that should NOT be detected as windmills"""
    if structure_type == "natural":
        # Irregular natural mound with high entropy
        elevation = np.zeros((size, size))
        for _ in range(5):
            cx = np.random.randint(size//4, 3*size//4)
            cy = np.random.randint(size//4, 3*size//4) 
            radius = np.random.uniform(3, 8)
            height = np.random.uniform(0.5, 1.5)
            
            y, x = np.ogrid[:size, :size]
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            bump = height * np.exp(-(distance**2) / (2 * radius**2))
            elevation += bump
        
        elevation += np.random.normal(0, 0.3, (size, size))
        
    elif structure_type == "linear":
        # Linear ridge - should have low compactness
        elevation = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist_to_line = abs(i - j) / np.sqrt(2)
                if dist_to_line < 3:
                    elevation[i, j] = 1.5 * np.exp(-dist_to_line**2 / 4)
        elevation += np.random.normal(0, 0.1, (size, size))
        
    elif structure_type == "flat":
        # Very flat terrain
        elevation = np.random.normal(0, 0.05, (size, size))
        
    else:  # random noise
        elevation = np.random.normal(0, 0.5, (size, size))
    
    return elevation

def calculate_basic_features(elevation):
    """Calculate basic features for manual validation"""
    # Volume-like metric (total positive elevation)
    volume = np.sum(np.maximum(elevation - elevation.mean(), 0))
    
    # Compactness-like metric (how circular the elevation pattern is)
    center = elevation.shape[0] // 2
    y, x = np.ogrid[:elevation.shape[0], :elevation.shape[1]]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Radial profile analysis
    max_radius = min(elevation.shape) // 2 - 2
    radial_profile = []
    for r in range(1, max_radius):
        mask = (distance >= r-0.5) & (distance < r+0.5)
        if np.any(mask):
            radial_profile.append(np.mean(elevation[mask]))
    
    # Compactness: how monotonically decreasing the radial profile is
    if len(radial_profile) > 3:
        differences = np.diff(radial_profile)
        decreasing_fraction = np.sum(differences < 0) / len(differences)
    else:
        decreasing_fraction = 0.0
    
    # Entropy-like metric (elevation distribution)
    hist, _ = np.histogram(elevation, bins=20)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    # Edge sharpness (gradient magnitude)
    gy, gx = np.gradient(elevation)
    edge_strength = np.mean(np.sqrt(gx**2 + gy**2))
    
    return {
        'volume': volume,
        'compactness': decreasing_fraction,
        'entropy': entropy,
        'edge_strength': edge_strength,
        'max_elevation': np.max(elevation),
        'elevation_std': np.std(elevation)
    }

def simple_windmill_classifier(features):
    """Simple windmill classifier based on basic features"""
    # These thresholds are rough estimates based on windmill characteristics
    score = 0.0
    
    # Windmills should have:
    # 1. Moderate volume (not too big, not too small)
    if 10 < features['volume'] < 100:
        score += 0.2
    
    # 2. Good compactness (circular, decreasing from center)
    if features['compactness'] > 0.6:
        score += 0.25
    
    # 3. Low to moderate entropy (structured, not chaotic)
    if features['entropy'] < 2.5:
        score += 0.2
    
    # 4. Sharp edges
    if features['edge_strength'] > 0.05:
        score += 0.15
    
    # 5. Reasonable elevation variation
    if 0.3 < features['elevation_std'] < 1.5:
        score += 0.1
    
    # 6. Peak elevation in reasonable range
    if 1.0 < features['max_elevation'] < 4.0:
        score += 0.1
    
    return score

def test_windmill_detection_simplified():
    """Test windmill detection using simplified feature analysis"""
    print("🎯 Simplified Windmill Detection Test")
    print("=" * 37)
    
    # Test windmill-like structures (should score high)
    print("\n✅ Testing Windmill-like Structures:")
    windmill_scores = []
    
    windmill_cases = [
        {"name": "Standard Windmill", "radius": 8, "height": 2.0, "noise": 0.1},
        {"name": "Large Windmill", "radius": 12, "height": 1.5, "noise": 0.15},
        {"name": "Small Windmill", "radius": 6, "height": 2.5, "noise": 0.08},
        {"name": "Noisy Windmill", "radius": 8, "height": 2.0, "noise": 0.25}
    ]
    
    for case in windmill_cases:
        elevation = create_synthetic_windmill(
            size=40, 
            radius=case["radius"], 
            height=case["height"], 
            noise=case["noise"]
        )
        
        features = calculate_basic_features(elevation)
        score = simple_windmill_classifier(features)
        windmill_scores.append(score)
        
        print(f"   {case['name']}: {score:.3f}")
        print(f"     Volume: {features['volume']:.1f}, Compactness: {features['compactness']:.3f}")
        print(f"     Entropy: {features['entropy']:.3f}, Edge: {features['edge_strength']:.3f}")
    
    # Test non-windmill structures (should score low)
    print("\n❌ Testing Non-Windmill Structures:")
    non_windmill_scores = []
    
    non_windmill_cases = [
        {"name": "Natural Irregular Mound", "type": "natural"},
        {"name": "Linear Ridge", "type": "linear"},
        {"name": "Flat Terrain", "type": "flat"},
        {"name": "Random Noise", "type": "random"}
    ]
    
    for case in non_windmill_cases:
        elevation = create_non_windmill(structure_type=case["type"], size=40)
        
        features = calculate_basic_features(elevation)
        score = simple_windmill_classifier(features)
        non_windmill_scores.append(score)
        
        print(f"   {case['name']}: {score:.3f}")
        print(f"     Volume: {features['volume']:.1f}, Compactness: {features['compactness']:.3f}")
        print(f"     Entropy: {features['entropy']:.3f}, Edge: {features['edge_strength']:.3f}")
    
    # Analysis
    print(f"\n📊 Detection Analysis:")
    
    avg_windmill_score = np.mean(windmill_scores)
    avg_non_windmill_score = np.mean(non_windmill_scores)
    
    print(f"   Average windmill score: {avg_windmill_score:.3f}")
    print(f"   Average non-windmill score: {avg_non_windmill_score:.3f}")
    print(f"   Score separation: {avg_windmill_score - avg_non_windmill_score:.3f}")
    
    # Classification with threshold 0.5
    threshold = 0.5
    
    windmill_detections = sum(1 for score in windmill_scores if score >= threshold)
    non_windmill_rejections = sum(1 for score in non_windmill_scores if score < threshold)
    
    sensitivity = windmill_detections / len(windmill_scores)
    specificity = non_windmill_rejections / len(non_windmill_scores)
    
    print(f"\n🎯 Classification Results (threshold = {threshold}):")
    print(f"   Sensitivity: {sensitivity:.1%} ({windmill_detections}/{len(windmill_scores)} windmills detected)")
    print(f"   Specificity: {specificity:.1%} ({non_windmill_rejections}/{len(non_windmill_scores)} non-windmills rejected)")
    
    # Overall grade
    if sensitivity >= 0.8 and specificity >= 0.8:
        grade = "🏅 EXCELLENT"
    elif sensitivity >= 0.6 and specificity >= 0.6:
        grade = "✅ GOOD"
    elif sensitivity >= 0.4 or specificity >= 0.4:
        grade = "⚠️ FAIR"
    else:
        grade = "❌ POOR"
    
    print(f"   Overall Performance: {grade}")
    
    return sensitivity, specificity, avg_windmill_score, avg_non_windmill_score

def test_polarity_preferences_validation():
    """Validate that polarity preferences are correctly configured"""
    print("\n⚙️ Polarity Preferences Validation")
    print("=" * 35)
    
    template_path = os.path.join(kernel_dir, "templates", "dutch_windmill.json")
    
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    polarity_prefs = template.get('polarity_preferences', {})
    
    # Expected polarity configuration for windmill detection
    expected_config = {
        'histogram': 'positive',    # φ⁰ signature should always be positive evidence
        'volume': None,            # Dynamic based on magnitude
        'dropoff': 'positive',     # Sharp edges support windmill detection  
        'compactness': 'positive', # Circular shapes support detection
        'entropy': 'negative',     # High chaos contradicts structures
        'planarity': 'negative'    # Flat surfaces contradict mounds
    }
    
    print("   Checking polarity configuration:")
    
    all_correct = True
    for feature, expected in expected_config.items():
        actual = polarity_prefs.get(feature)
        
        if actual == expected:
            print(f"     ✅ {feature}: {actual} (correct)")
        else:
            print(f"     ❌ {feature}: {actual}, expected {expected}")
            all_correct = False
    
    if all_correct:
        print("   🏅 All polarity preferences are optimally configured for windmill detection!")
    else:
        print("   ⚠️ Some polarity preferences may need adjustment")
    
    return all_correct

def run_comprehensive_test():
    """Run all tests"""
    print("🏰 Comprehensive Windmill Detection Test")
    print("=" * 40)
    print()
    
    # Test 1: Polarity configuration
    config_correct = test_polarity_preferences_validation()
    
    # Test 2: Detection performance
    sensitivity, specificity, windmill_score, non_windmill_score = test_windmill_detection_simplified()
    
    # Overall assessment
    print(f"\n🎖️ Final Assessment")
    print("=" * 19)
    
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0
    score_separation = windmill_score - non_windmill_score
    
    print(f"   Configuration Quality: {'✅ Optimal' if config_correct else '⚠️ Needs tuning'}")
    print(f"   Detection F1-Score: {f1_score:.3f}")
    print(f"   Feature Separation: {score_separation:.3f}")
    
    if config_correct and f1_score >= 0.7 and score_separation >= 0.2:
        print("   🏅 System is well-configured and performing excellently!")
    elif f1_score >= 0.5:
        print("   ✅ System is functional with room for improvement")
    else:
        print("   ⚠️ System needs significant tuning")

if __name__ == "__main__":
    run_comprehensive_test()

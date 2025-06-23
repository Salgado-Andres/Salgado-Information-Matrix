#!/usr/bin/env python3
"""
G₂ Kernel - Dutch Windmill Detection Test
Tests the windmill profile with polarity preferences and simulated windmill structures.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add kernel to path
kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, kernel_dir)

# Import after path setup
import detector_profile
from detector_profile import DetectorProfileManager, StructureType
import core_detector
from core_detector import G2StructureDetector
import aggregator
from aggregator import StreamingDetectionAggregator

def create_synthetic_windmill_mound(size: int = 40, radius: float = 8.0, height: float = 2.0, noise_level: float = 0.1) -> np.ndarray:
    """
    Create a synthetic circular windmill mound for testing
    
    Args:
        size: Grid size (size x size)
        radius: Mound radius in pixels
        height: Peak height above baseline
        noise_level: Random noise amplitude
    
    Returns:
        2D elevation array with windmill-like circular mound
    """
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Calculate distance from center
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Create circular mound with smooth falloff
    # Use a Gaussian-like profile for realistic windmill mound
    mound = height * np.exp(-(distance**2) / (2 * (radius/2.5)**2))
    
    # Add some asymmetry and edge characteristics typical of windmill mounds
    # Slight elliptical distortion
    angle = np.arctan2(y - center, x - center)
    asymmetry = 1.0 + 0.1 * np.cos(2 * angle)  # Slight 2-fold asymmetry
    mound *= asymmetry
    
    # Add edge sharpening (windmill mounds often have defined edges)
    edge_mask = distance < radius
    mound[~edge_mask] *= 0.3  # Sharp dropoff outside radius
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, (size, size))
    mound += noise
    
    return mound

def create_non_windmill_structure(size: int = 40, structure_type: str = "natural") -> np.ndarray:
    """
    Create structures that should NOT be detected as windmills
    
    Args:
        size: Grid size
        structure_type: "natural", "linear", "random", or "flat"
    """
    if structure_type == "natural":
        # Natural irregular mound (high entropy, no clear circular pattern)
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        # Create multiple overlapping bumps with random positioning
        elevation = np.zeros((size, size))
        for _ in range(5):
            bump_x = np.random.randint(size//4, 3*size//4)
            bump_y = np.random.randint(size//4, 3*size//4)
            bump_size = np.random.uniform(3, 8)
            bump_height = np.random.uniform(0.5, 1.5)
            
            distance = np.sqrt((x - bump_x)**2 + (y - bump_y)**2)
            bump = bump_height * np.exp(-(distance**2) / (2 * bump_size**2))
            elevation += bump
            
        # Add significant noise for high entropy
        elevation += np.random.normal(0, 0.3, (size, size))
        
    elif structure_type == "linear":
        # Linear feature (road, ridge) - should have low compactness
        elevation = np.zeros((size, size))
        # Create a diagonal ridge
        for i in range(size):
            for j in range(size):
                dist_to_line = abs(i - j) / np.sqrt(2)
                if dist_to_line < 3:
                    elevation[i, j] = 1.5 * np.exp(-dist_to_line**2 / 4)
        
        elevation += np.random.normal(0, 0.1, (size, size))
        
    elif structure_type == "flat":
        # Very flat area with minimal elevation change
        elevation = np.random.normal(0, 0.05, (size, size))
        
    else:  # random
        # Completely random noise
        elevation = np.random.normal(0, 0.5, (size, size))
    
    return elevation

def test_windmill_profile_loading():
    """Test that the Dutch windmill profile loads correctly with polarity preferences"""
    print("🧪 Testing Windmill Profile Loading")
    print("=" * 40)
    
    # Load Dutch windmill template with correct path
    kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(kernel_dir, "templates")
    profiles_dir = os.path.join(kernel_dir, "profiles")
    
    manager = DetectorProfileManager(
        profiles_dir=profiles_dir,
        templates_dir=templates_dir
    )
    
    try:
        profile = manager.load_template("dutch_windmill.json")
        
        print(f"✅ Profile loaded: {profile.name}")
        print(f"   Structure type: {profile.structure_type}")
        print(f"   Features enabled: {len(profile.get_enabled_features())}/6")
        
        # Check polarity preferences
        polarity_prefs = {}
        for feature_name, config in profile.features.items():
            if hasattr(config, 'polarity_preference') and config.polarity_preference:
                polarity_prefs[feature_name] = config.polarity_preference
        
        print(f"   Polarity preferences set: {len(polarity_prefs)}")
        for feature, polarity in polarity_prefs.items():
            print(f"     {feature}: {polarity}")
        
        return profile
        
    except Exception as e:
        print(f"❌ Failed to load windmill profile: {e}")
        return None

def test_windmill_detection_positive():
    """Test detection on synthetic windmill structures (should detect)"""
    print("\n🎯 Testing Positive Windmill Detection")
    print("=" * 42)
    
    # Load windmill profile with correct paths
    kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(kernel_dir, "templates")
    profiles_dir = os.path.join(kernel_dir, "profiles")
    
    manager = DetectorProfileManager(
        profiles_dir=profiles_dir,
        templates_dir=templates_dir
    )
    profile = manager.load_template("dutch_windmill.json")
    
    # Create detector with windmill profile
    detector = G2StructureDetector(profile=profile)
    
    # Test multiple windmill variants
    test_cases = [
        {"name": "Standard Windmill Mound", "radius": 8.0, "height": 2.0, "noise": 0.1},
        {"name": "Large Windmill Mound", "radius": 12.0, "height": 1.5, "noise": 0.15},
        {"name": "Small Windmill Mound", "radius": 6.0, "height": 2.5, "noise": 0.08},
        {"name": "Noisy Windmill Mound", "radius": 8.0, "height": 2.0, "noise": 0.25},
    ]
    
    detections = 0
    total_confidence = 0.0
    
    for case in test_cases:
        # Create synthetic windmill
        elevation = create_synthetic_windmill_mound(
            size=40, 
            radius=case["radius"], 
            height=case["height"], 
            noise_level=case["noise"]
        )
        
        # Run detection
        result = detector.detect_structure(elevation)
        
        total_confidence += result.confidence
        is_detected = result.is_positive
        
        if is_detected:
            detections += 1
        
        print(f"   {case['name']}: {'✅ DETECTED' if is_detected else '❌ missed'} "
              f"(confidence: {result.confidence:.3f}, score: {result.final_score:.3f})")
        
        # Show polarity breakdown if available
        if hasattr(result, 'positive_evidence_count'):
            print(f"      Positive evidence: {result.positive_evidence_count}, "
                  f"Negative evidence: {result.negative_evidence_count}")
    
    avg_confidence = total_confidence / len(test_cases)
    detection_rate = detections / len(test_cases)
    
    print(f"\n📊 Windmill Detection Results:")
    print(f"   Detection rate: {detection_rate:.1%} ({detections}/{len(test_cases)})")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    return detection_rate, avg_confidence

def test_windmill_detection_negative():
    """Test detection on non-windmill structures (should NOT detect)"""
    print("\n🚫 Testing Negative Windmill Detection (Non-Windmills)")
    print("=" * 54)
    
    # Load windmill profile with correct paths
    kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(kernel_dir, "templates")
    profiles_dir = os.path.join(kernel_dir, "profiles")
    
    manager = DetectorProfileManager(
        profiles_dir=profiles_dir,
        templates_dir=templates_dir
    )
    profile = manager.load_template("dutch_windmill.json")
    
    # Create detector with windmill profile
    detector = G2StructureDetector(profile=profile)
    
    # Test non-windmill structures
    test_cases = [
        {"name": "Natural Irregular Mound", "type": "natural"},
        {"name": "Linear Ridge Feature", "type": "linear"},
        {"name": "Flat Terrain", "type": "flat"},
        {"name": "Random Noise", "type": "random"},
    ]
    
    false_positives = 0
    total_confidence = 0.0
    
    for case in test_cases:
        # Create non-windmill structure
        elevation = create_non_windmill_structure(size=40, structure_type=case["type"])
        
        # Run detection
        result = detector.detect_structure(elevation)
        
        total_confidence += result.confidence
        is_detected = result.is_positive
        
        if is_detected:
            false_positives += 1
        
        print(f"   {case['name']}: {'❌ FALSE POS' if is_detected else '✅ correctly rejected'} "
              f"(confidence: {result.confidence:.3f}, score: {result.final_score:.3f})")
        
        # Show polarity breakdown if available
        if hasattr(result, 'positive_evidence_count'):
            print(f"      Positive evidence: {result.positive_evidence_count}, "
                  f"Negative evidence: {result.negative_evidence_count}")
    
    avg_confidence = total_confidence / len(test_cases)
    false_positive_rate = false_positives / len(test_cases)
    specificity = 1.0 - false_positive_rate
    
    print(f"\n📊 Non-Windmill Rejection Results:")
    print(f"   Specificity: {specificity:.1%} (correctly rejected {len(test_cases)-false_positives}/{len(test_cases)})")
    print(f"   False positive rate: {false_positive_rate:.1%}")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    return specificity, false_positive_rate

def test_polarity_preferences_impact():
    """Test the impact of polarity preferences on detection accuracy"""
    print("\n⚙️ Testing Polarity Preferences Impact")
    print("=" * 40)
    
    # Load windmill profile with correct paths
    kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(kernel_dir, "templates")
    profiles_dir = os.path.join(kernel_dir, "profiles")
    
    manager = DetectorProfileManager(
        profiles_dir=profiles_dir,
        templates_dir=templates_dir
    )
    profile = manager.load_template("dutch_windmill.json")
    
    # Create a test windmill
    windmill_elevation = create_synthetic_windmill_mound(size=40, radius=8.0, height=2.0, noise=0.1)
    
    # Test with polarity preferences (current configuration)
    detector_with_prefs = G2StructureDetector(profile=profile)
    result_with_prefs = detector_with_prefs.detect_structure(windmill_elevation)
    
    print(f"   With polarity preferences:")
    print(f"     Score: {result_with_prefs.final_score:.3f}, Confidence: {result_with_prefs.confidence:.3f}")
    print(f"     Detection: {'✅ POSITIVE' if result_with_prefs.is_positive else '❌ negative'}")
    
    # Create a profile without polarity preferences (remove them)
    profile_no_prefs = manager.load_template("dutch_windmill.json")
    for feature_config in profile_no_prefs.features.values():
        feature_config.polarity_preference = None
    
    detector_no_prefs = G2StructureDetector(profile=profile_no_prefs)
    result_no_prefs = detector_no_prefs.detect_structure(windmill_elevation)
    
    print(f"   Without polarity preferences (dynamic only):")
    print(f"     Score: {result_no_prefs.final_score:.3f}, Confidence: {result_no_prefs.confidence:.3f}")
    print(f"     Detection: {'✅ POSITIVE' if result_no_prefs.is_positive else '❌ negative'}")
    
    # Compare results
    score_diff = result_with_prefs.final_score - result_no_prefs.final_score
    conf_diff = result_with_prefs.confidence - result_no_prefs.confidence
    
    print(f"\n📈 Impact of Polarity Preferences:")
    print(f"   Score difference: {score_diff:+.3f}")
    print(f"   Confidence difference: {conf_diff:+.3f}")
    
    if score_diff > 0:
        print("   ✅ Polarity preferences improved detection accuracy")
    elif score_diff < 0:
        print("   ⚠️ Polarity preferences decreased detection accuracy")
    else:
        print("   ➖ No significant impact from polarity preferences")
    
    return score_diff, conf_diff

def run_comprehensive_windmill_test():
    """Run all windmill detection tests"""
    print("🏰 G₂ Kernel - Dutch Windmill Detection Test Suite")
    print("=" * 50)
    print()
    
    # Test 1: Profile loading
    profile = test_windmill_profile_loading()
    if not profile:
        print("❌ Cannot continue - profile loading failed")
        return
    
    # Test 2: Positive detection (windmills)
    detection_rate, avg_confidence_pos = test_windmill_detection_positive()
    
    # Test 3: Negative detection (non-windmills)
    specificity, false_pos_rate = test_windmill_detection_negative()
    
    # Test 4: Polarity preferences impact
    score_diff, conf_diff = test_polarity_preferences_impact()
    
    # Overall assessment
    print("\n🎯 Overall Assessment")
    print("=" * 21)
    
    # Calculate F1-like score (harmonic mean of sensitivity and specificity)
    sensitivity = detection_rate  # True positive rate
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0
    
    print(f"   Sensitivity (windmill detection): {sensitivity:.1%}")
    print(f"   Specificity (non-windmill rejection): {specificity:.1%}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   Average positive confidence: {avg_confidence_pos:.3f}")
    print(f"   Polarity preference impact: {score_diff:+.3f}")
    
    # Grade the performance
    if f1_score >= 0.9 and detection_rate >= 0.8:
        grade = "🏅 EXCELLENT"
    elif f1_score >= 0.7 and detection_rate >= 0.6:
        grade = "✅ GOOD"
    elif f1_score >= 0.5:
        grade = "⚠️ FAIR"
    else:
        grade = "❌ POOR"
    
    print(f"\n📊 Performance Grade: {grade}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if detection_rate < 0.8:
        print("   - Consider lowering detection threshold for better sensitivity")
    if false_pos_rate > 0.2:
        print("   - Consider raising detection threshold to reduce false positives")
    if abs(score_diff) < 0.05:
        print("   - Polarity preferences may need tuning for greater impact")
    if f1_score >= 0.8:
        print("   - Windmill profile is well-tuned for this structure type! 🎯")

if __name__ == "__main__":
    run_comprehensive_windmill_test()

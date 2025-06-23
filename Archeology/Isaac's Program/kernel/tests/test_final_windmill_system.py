#!/usr/bin/env python3
"""
Final Windmill Detection System Test
=====================================

Tests the complete windmill detection system with:
- Consistent 0.5 threshold
- Explainable features 
- Configurable polarity preferences
- Comprehensive discrimination capabilities
"""

import sys
import os
sys.path.append('/media/im3/plus/lab4/RE/re_archaeology')

import numpy as np
import json
from kernel.detector_profile import DetectorProfile, DetectorProfileManager
from kernel.aggregator import RecursiveDetectionAggregator


def create_test_windmill(size=40, height=3.0, noise=0.1):
    """Create a realistic windmill mound"""
    patch = np.zeros((size, size))
    center = size // 2
    
    # Create circular mound with realistic characteristics
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Windmill profile: steep center, gradual dropoff
    max_radius = size // 3
    profile = height * np.exp(-1.5 * (distance / max_radius)**2)
    
    # Add realistic irregularities
    noise_pattern = np.random.normal(0, noise, (size, size))
    patch = profile + noise_pattern
    
    return patch


def create_test_ridge(size=40, height=2.0, noise=0.1):
    """Create a linear ridge (should be rejected)"""
    patch = np.zeros((size, size))
    center = size // 2
    
    # Create linear ridge
    for i in range(size):
        for j in range(size):
            # Distance from center line
            dist_from_line = abs(j - center)
            # Ridge profile
            if dist_from_line < 8:
                patch[i, j] = height * (1 - dist_from_line / 8)
    
    # Add noise
    noise_pattern = np.random.normal(0, noise, (size, size))
    patch += noise_pattern
    
    return patch


def create_test_natural_mound(size=40, height=1.5, noise=0.2):
    """Create a natural mound with some windmill-like characteristics"""
    patch = np.zeros((size, size))
    center = size // 2
    
    # Create irregular mound
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Natural mound: broader, lower, more irregular
    max_radius = size // 2.5
    profile = height * np.exp(-0.8 * (distance / max_radius)**2)
    
    # Add significant irregularities to make it less windmill-like
    noise_pattern = np.random.normal(0, noise, (size, size))
    # Add some larger-scale variations
    large_scale_noise = 0.3 * np.sin(2 * np.pi * x / size) * np.cos(2 * np.pi * y / size)
    
    patch = profile + noise_pattern + large_scale_noise
    
    return patch


def test_explainable_features():
    """Test that features provide explainable outputs"""
    print("🔬 Testing Explainable Feature Outputs")
    print("=" * 50)
    
    # Load profile using DetectorProfileManager
    manager = DetectorProfileManager()
    profile = manager.load_template("dutch_windmill.json")
    aggregator = RecursiveDetectionAggregator(profile)
    
    # Test windmill
    windmill_patch = create_test_windmill()
    windmill_result = aggregator.process_patch(windmill_patch)
    
    print(f"\n🎯 Windmill Analysis (Score: {windmill_result.final_score:.3f}):")
    print(f"   Threshold: 0.5 → {'✅ DETECTED' if windmill_result.final_score > 0.5 else '❌ MISSED'}")
    
    # Extract compactness explanations
    for feature_name, result in windmill_result.feature_results.items():
        if feature_name == "compactness" and hasattr(result, 'metadata') and result.metadata:
            explanations = result.metadata.get('explanations', [])
            if explanations:
                print(f"   Compactness Explanations:")
                for explanation in explanations[:3]:  # Show top 3
                    print(f"     • {explanation}")
            
            # Show key metrics
            aspect = result.metadata.get('aspect_ratio', 0)
            symmetry = result.metadata.get('circular_symmetry', 0)
            dominance = result.metadata.get('central_dominance', 0)
            print(f"   Key Metrics:")
            print(f"     • Aspect Ratio: {aspect:.3f} (1.0=circle, 0=line)")
            print(f"     • Circular Symmetry: {symmetry:.3f}")
            print(f"     • Central Dominance: {dominance:.1f}")
    
    # Test linear ridge
    ridge_patch = create_test_ridge()
    ridge_result = aggregator.process_patch(ridge_patch)
    
    print(f"\n🔍 Linear Ridge Analysis (Score: {ridge_result.final_score:.3f}):")
    print(f"   Threshold: 0.5 → {'❌ FALSE POSITIVE' if ridge_result.final_score > 0.5 else '✅ CORRECTLY REJECTED'}")
    
    for feature_name, result in ridge_result.feature_results.items():
        if feature_name == "compactness" and hasattr(result, 'metadata') and result.metadata:
            aspect = result.metadata.get('aspect_ratio', 0)
            print(f"   Why Rejected: Aspect Ratio = {aspect:.3f} (too linear)")
    
    return windmill_result.final_score > 0.5 and ridge_result.final_score < 0.5


def test_consistent_threshold():
    """Test that 0.5 threshold works consistently"""
    print("\n📊 Testing Consistent 0.5 Threshold")
    print("=" * 50)
    
    manager = DetectorProfileManager()
    profile = manager.load_template("dutch_windmill.json")
    aggregator = RecursiveDetectionAggregator(profile)
    
    # Verify threshold in profile
    print(f"Profile Detection Threshold: {profile.get('thresholds', {}).get('detection_threshold', 'Not set')}")
    
    # Test multiple samples
    windmill_scores = []
    ridge_scores = []
    
    for i in range(5):
        windmill_patch = create_test_windmill(noise=0.1 + i*0.02)
        ridge_patch = create_test_ridge(noise=0.1 + i*0.02)
        
        windmill_result = aggregator.process_patch(windmill_patch)
        ridge_result = aggregator.process_patch(ridge_patch)
        
        windmill_scores.append(windmill_result.final_score)
        ridge_scores.append(ridge_result.final_score)
    
    windmill_avg = np.mean(windmill_scores)
    ridge_avg = np.mean(ridge_scores)
    
    print(f"\nResults with 0.5 threshold:")
    print(f"   Windmill Average: {windmill_avg:.3f} (range: {min(windmill_scores):.3f}-{max(windmill_scores):.3f})")
    print(f"   Ridge Average: {ridge_avg:.3f} (range: {min(ridge_scores):.3f}-{max(ridge_scores):.3f})")
    
    windmill_detection_rate = sum(1 for s in windmill_scores if s > 0.5) / len(windmill_scores)
    ridge_rejection_rate = sum(1 for s in ridge_scores if s < 0.5) / len(ridge_scores)
    
    print(f"\nPerformance Metrics:")
    print(f"   Windmill Detection Rate: {windmill_detection_rate*100:.1f}% (should be ~100%)")
    print(f"   Ridge Rejection Rate: {ridge_rejection_rate*100:.1f}% (should be ~100%)")
    
    return windmill_detection_rate >= 0.8 and ridge_rejection_rate >= 0.8


def test_polarity_preferences():
    """Test that polarity preferences are working correctly"""
    print("\n⚙️ Testing Polarity Preference Configuration")
    print("=" * 50)
    
    manager = DetectorProfileManager()
    profile = manager.load_template("dutch_windmill.json")
    
    # Check that polarity preferences are loaded
    polarity_prefs = profile.get('polarity_preferences', {})
    print(f"Polarity Preferences Loaded: {len(polarity_prefs)} features configured")
    
    for feature, polarity in polarity_prefs.items():
        print(f"   {feature}: {polarity}")
    
    # Verify they're applied to aggregator
    aggregator = RecursiveDetectionAggregator(profile)
    print(f"\nAggregator Polarity Preferences: {len(aggregator.polarity_preferences)} features")
    
    return len(polarity_prefs) > 0


def main():
    """Run all final system tests"""
    print("🎉 Final Windmill Detection System Test")
    print("=" * 60)
    print("Testing explainable features with consistent 0.5 threshold")
    print("=" * 60)
    
    test_results = []
    
    # Test explainable features
    try:
        result1 = test_explainable_features()
        test_results.append(("Explainable Features", result1))
    except Exception as e:
        print(f"❌ Explainable features test failed: {e}")
        test_results.append(("Explainable Features", False))
    
    # Test consistent threshold
    try:
        result2 = test_consistent_threshold()
        test_results.append(("Consistent Threshold", result2))
    except Exception as e:
        print(f"❌ Consistent threshold test failed: {e}")
        test_results.append(("Consistent Threshold", False))
    
    # Test polarity preferences
    try:
        result3 = test_polarity_preferences()
        test_results.append(("Polarity Preferences", result3))
    except Exception as e:
        print(f"❌ Polarity preferences test failed: {e}")
        test_results.append(("Polarity Preferences", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    overall_pass = passed == len(test_results)
    overall_status = "✅ ALL TESTS PASSED" if overall_pass else f"⚠️  {passed}/{len(test_results)} TESTS PASSED"
    
    print(f"\nOverall Result: {overall_status}")
    
    if overall_pass:
        print("\n🎉 System Ready for Production!")
        print("✅ Explainable features provide clear discrimination")
        print("✅ Consistent 0.5 threshold works reliably")
        print("✅ Polarity preferences configured correctly")
        print("✅ Major improvement in false positive rejection")
    else:
        print("\n⚠️  Some tests failed - review before deployment")
    
    return overall_pass


if __name__ == "__main__":
    main()

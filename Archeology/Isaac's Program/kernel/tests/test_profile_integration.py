#!/usr/bin/env python3
"""
Test DetectorProfile Integration with G2StructureDetector

This test suite verifies that the G2StructureDetector properly integrates
with the DetectorProfile system, supporting all three initialization modes:
1. Direct profile object usage
2. Profile loading by name 
3. Backward compatibility with legacy parameters

Tests cover configuration propagation, feature module setup, and end-to-end detection.
"""

import sys
import os
import numpy as np
import logging
import warnings

# Add the kernel directory to path for running from main project directory
sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from kernel.core_detector import G2StructureDetector, ElevationPatch
from kernel.detector_profile import (
    DetectorProfile, 
    DetectorProfileManager,
    StructureType,
    GeometricParameters,
    FeatureConfiguration,
    DetectionThresholds
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_elevation_data(size: int = 40, add_structure: bool = True) -> np.ndarray:
    """Create synthetic elevation data for testing"""
    # Generate random terrain base
    elevation_data = np.random.randn(size, size) * 0.5 + 10.0
    
    if add_structure:
        # Add a synthetic windmill-like circular structure
        center = size // 2
        radius = 8
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        elevation_data[mask] += 2.0  # Raise elevation in circular area
    
    return elevation_data


def test_profile_object_mode():
    """Test 1: Using existing profile object"""
    print("\n📋 Test 1: Profile Object Mode")
    print("-" * 40)
    
    try:
        # Create a custom profile
        profile = DetectorProfile(
            name="Test Custom Profile",
            description="Custom test profile for integration testing",
            structure_type=StructureType.WINDMILL,
            geometry=GeometricParameters(
                resolution_m=0.3,
                structure_radius_m=10.0,
                patch_size_m=(30.0, 30.0)
            ),
            thresholds=DetectionThresholds(
                detection_threshold=0.6,
                early_decision_threshold=0.9,
                min_modules_for_decision=3
            ),
            max_workers=4
        )
        
        # Initialize detector with profile
        detector = G2StructureDetector(profile=profile)
        
        # Verify configuration propagation
        assert detector.profile.name == "Test Custom Profile"
        assert detector.resolution_m == 0.3
        assert detector.structure_radius_m == 10.0
        assert detector.max_workers == 4
        assert len(detector.feature_modules) == 6  # All default features enabled
        
        print(f"✅ Success! Profile: '{detector.profile.name}'")
        print(f"   Resolution: {detector.resolution_m}m")
        print(f"   Structure radius: {detector.structure_radius_m}m")
        print(f"   Max workers: {detector.max_workers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_profile_name_mode():
    """Test 2: Loading profile by name (fallback to default)"""
    print("\n📁 Test 2: Profile Name Mode")
    print("-" * 40)
    
    try:
        # Try to load a non-existent profile (should fallback to default)
        detector = G2StructureDetector(profile_name="non_existent_profile")
        
        # Verify default profile was created
        assert detector.profile.name == "Amazon Windmill Default"
        assert detector.resolution_m == 0.5
        assert detector.structure_radius_m == 8.0
        assert len(detector.feature_modules) == 6
        
        print(f"✅ Success! Fallback profile: '{detector.profile.name}'")
        print(f"   Resolution: {detector.resolution_m}m")
        print(f"   Structure radius: {detector.structure_radius_m}m")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_legacy_compatibility_mode():
    """Test 3: Backward compatibility with legacy parameters"""
    print("\n⚠️  Test 3: Legacy Compatibility Mode")
    print("-" * 40)
    
    try:
        # Test with legacy parameters (should show deprecation warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            detector = G2StructureDetector(
                resolution_m=0.25,
                structure_radius_m=15.0,
                structure_type="settlement",
                max_workers=8,
                enable_refinement=False
            )
            
            # Verify deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
        
        # Verify legacy parameters were applied
        assert detector.profile.name == "Legacy Parameter Profile"
        assert detector.resolution_m == 0.25
        assert detector.structure_radius_m == 15.0
        assert detector.structure_type == "settlement"
        assert detector.max_workers == 8
        assert detector.enable_refinement == False
        
        print(f"✅ Success! Legacy profile: '{detector.profile.name}'")
        print(f"   Resolution: {detector.resolution_m}m")
        print(f"   Structure radius: {detector.structure_radius_m}m")
        print(f"   Structure type: {detector.structure_type}")
        print(f"   Deprecation warning: ✅ Shown correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_default_profile_mode():
    """Test 4: Default profile when no parameters provided"""
    print("\n🎯 Test 4: Default Profile Mode")
    print("-" * 40)
    
    try:
        # Create detector with no parameters
        detector = G2StructureDetector()
        
        # Verify default Amazon Windmill profile
        assert detector.profile.name == "Amazon Windmill Default"
        assert detector.resolution_m == 0.5
        assert detector.structure_radius_m == 8.0
        assert detector.structure_type == "windmill"
        assert len(detector.feature_modules) == 6
        
        print(f"✅ Success! Default profile: '{detector.profile.name}'")
        print(f"   Resolution: {detector.resolution_m}m")
        print(f"   Structure radius: {detector.structure_radius_m}m")
        print(f"   Structure type: {detector.structure_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_feature_configuration_propagation():
    """Test 5: Feature module configuration from profile"""
    print("\n⚙️  Test 5: Feature Configuration Propagation")
    print("-" * 40)
    
    try:
        # Create profile with custom feature configuration using correct registry names
        profile = DetectorProfile(
            name="Custom Feature Profile",
            features={
                "ElevationHistogram": FeatureConfiguration(enabled=True, weight=2.0),
                "Volume": FeatureConfiguration(enabled=True, weight=1.5),
                "DropoffSharpness": FeatureConfiguration(enabled=False, weight=0.0),  # Disabled
                "Compactness": FeatureConfiguration(enabled=True, weight=1.2),
                "ElevationEntropy": FeatureConfiguration(enabled=False, weight=0.0),  # Disabled
                "Planarity": FeatureConfiguration(enabled=True, weight=0.8)
            }
        )
        
        detector = G2StructureDetector(profile=profile)
        
        # Verify profile configuration is correct
        enabled_features = detector.profile.get_enabled_features()
        assert len(enabled_features) == 4  # Only 4 features enabled in profile
        assert "ElevationHistogram" in enabled_features
        assert "Volume" in enabled_features
        assert "DropoffSharpness" not in enabled_features  # Should be disabled in profile
        assert "ElevationEntropy" not in enabled_features  # Should be disabled in profile
        
        # Note: feature_modules still contains all modules (registry returns all)
        # but only enabled ones get weights during initialization
        assert len(detector.feature_modules) == 6  # All modules loaded by registry
        
        # Verify custom weights are applied to enabled features  
        assert detector.feature_modules["ElevationHistogram"].weight == 2.0
        assert detector.feature_modules["Volume"].weight == 1.5
        assert detector.feature_modules["Compactness"].weight == 1.2
        assert detector.feature_modules["Planarity"].weight == 0.8
        
        print(f"✅ Success! Feature configuration applied correctly")
        print(f"   Profile enabled features: {len(enabled_features)}/6")
        print(f"   Profile disabled features: DropoffSharpness, ElevationEntropy")
        print(f"   Registry loaded modules: {len(detector.feature_modules)}")
        print(f"   Custom weights: ElevationHistogram=2.0, Volume=1.5")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_end_to_end_detection():
    """Test 6: End-to-end detection with profile integration"""
    print("\n🔬 Test 6: End-to-End Detection")
    print("-" * 40)
    
    try:
        # Create detector with custom profile
        profile = DetectorProfile(
            name="E2E Test Profile",
            thresholds=DetectionThresholds(
                detection_threshold=0.4,  # Lower threshold for easier testing
                early_decision_threshold=0.8
            )
        )
        
        detector = G2StructureDetector(profile=profile)
        
        # Create test elevation data with structure
        elevation_data = create_test_elevation_data(size=40, add_structure=True)
        patch = ElevationPatch(
            elevation_data=elevation_data,
            lat=-2.5,
            lon=-60.0,
            resolution_m=0.5,
            source="synthetic_test_data"
        )
        
        # Run detection
        result = detector.detect_structure(patch)
        
        # Verify result structure
        assert hasattr(result, 'detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'final_score')
        assert hasattr(result, 'feature_results')
        assert len(result.feature_results) == 6  # All features should have run
        
        print(f"✅ Success! Detection completed")
        print(f"   Detected: {result.detected}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Final score: {result.final_score:.3f}")
        print(f"   Features processed: {len(result.feature_results)}")
        print(f"   Profile used: {detector.profile.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_test_suite():
    """Run the complete DetectorProfile integration test suite"""
    print("🧪 DetectorProfile Integration Test Suite")
    print("=" * 60)
    print("Testing integration between DetectorProfile and G2StructureDetector")
    
    tests = [
        test_profile_object_mode,
        test_profile_name_mode,
        test_legacy_compatibility_mode,
        test_default_profile_mode,
        test_feature_configuration_propagation,
        test_end_to_end_detection
    ]
    
    results = []
    for test_func in tests:
        success = test_func()
        results.append(success)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Profile system is working correctly.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_integration_test_suite()
    sys.exit(0 if success else 1)

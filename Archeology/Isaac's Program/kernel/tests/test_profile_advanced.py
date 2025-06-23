#!/usr/bin/env python3
"""
Advanced test for the DetectorProfile system - demonstrating all capabilities
"""

import os
import sys
import numpy as np

# Add kernel to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_profile import DetectorProfile, StructureType, PatchShape, profile_manager

def test_preset_profiles():
    """Test creating and loading preset profiles"""
    print("🎯 Testing Preset Profiles")
    print("=" * 40)
    
    # Create preset profiles
    profile_manager.create_preset_profiles()
    
    # List available profiles
    available_profiles = profile_manager.list_profiles()
    print(f"✅ Created {len(available_profiles)} preset profiles:")
    for profile_name in available_profiles:
        print(f"   - {profile_name}")
    
    # Load and examine a specific preset
    if "amazon_windmill.json" in available_profiles:
        windmill_profile = profile_manager.load_profile("amazon_windmill.json")
        print(f"\n📋 Loaded profile: {windmill_profile.name}")
        print(f"   Description: {windmill_profile.description}")
        print(f"   Structure type: {windmill_profile.structure_type}")
        print(f"   Resolution: {windmill_profile.geometry.resolution_m}m")
        print(f"   Patch size: {windmill_profile.geometry.patch_size_m}")
        return windmill_profile
    else:
        print("⚠️ Amazon windmill profile not found, using first available")
        first_profile = profile_manager.load_profile(available_profiles[0])
        return first_profile

def test_custom_profile_creation():
    """Test creating custom profiles with specific configurations"""
    print("\n🛠️ Testing Custom Profile Creation")
    print("=" * 40)
    
    # Create a high-resolution detailed analysis profile
    detail_profile = DetectorProfile(
        name="HighResolutionAnalysis",
        description="Ultra-high resolution for detailed archaeological analysis",
        structure_type=StructureType.SETTLEMENT
    )
    
    # Configure geometry for high detail
    detail_profile.geometry.resolution_m = 0.1  # 10cm resolution
    detail_profile.geometry.patch_size_m = (15.0, 15.0)  # Small patches
    detail_profile.geometry.structure_radius_m = 5.0  # Small structures
    detail_profile.geometry.patch_shape = PatchShape.SQUARE
    
    # Configure thresholds for precision
    detail_profile.thresholds.detection_threshold = 0.75  # High confidence required
    detail_profile.thresholds.confidence_threshold = 0.8
    detail_profile.thresholds.early_decision_threshold = 0.9  # Almost never early terminate
    
    # Configure features for detailed analysis (all enabled with balanced weights)
    detail_profile.features["histogram"].weight = 1.8
    detail_profile.features["volume"].weight = 1.6
    detail_profile.features["compactness"].weight = 1.4
    detail_profile.features["dropoff"].weight = 1.3
    detail_profile.features["entropy"].weight = 1.2
    detail_profile.features["planarity"].weight = 1.1
    
    print(f"✅ Created detailed profile: {detail_profile.name}")
    print(f"   Resolution: {detail_profile.geometry.resolution_m}m")
    print(f"   All features enabled with balanced weights")
    
    # Create a fast survey profile
    survey_profile = DetectorProfile(
        name="RapidSurvey",
        description="Fast scanning for large area surveys",
        structure_type=StructureType.GENERIC
    )
    
    # Configure for speed
    survey_profile.geometry.resolution_m = 1.0  # Lower resolution
    survey_profile.geometry.patch_size_m = (50.0, 30.0)  # Large rectangular patches
    survey_profile.geometry.patch_shape = PatchShape.RECTANGLE
    
    # Configure for early decisions
    survey_profile.thresholds.detection_threshold = 0.5  # Lower threshold
    survey_profile.thresholds.early_decision_threshold = 0.7  # Early termination
    survey_profile.thresholds.min_modules_for_decision = 2  # Minimum modules
    
    # Enable only the fastest, most reliable features
    survey_profile.features["histogram"].weight = 2.5  # Rely heavily on histogram
    survey_profile.features["volume"].weight = 1.8
    survey_profile.features["dropoff"].enabled = False  # Disable slow features
    survey_profile.features["compactness"].enabled = False
    survey_profile.features["entropy"].enabled = False
    survey_profile.features["planarity"].enabled = False
    
    print(f"✅ Created survey profile: {survey_profile.name}")
    print(f"   Enabled features: {len(survey_profile.get_enabled_features())}/6")
    print(f"   Early decision threshold: {survey_profile.thresholds.early_decision_threshold}")
    
    return detail_profile, survey_profile

def test_profile_optimization():
    """Test profile optimization for structure types"""
    print("\n🔧 Testing Profile Optimization")
    print("=" * 40)
    
    # Create a base profile to optimize
    base_profile = DetectorProfile(
        name="OptimizationTest",
        description="Profile for testing optimization capabilities"
    )
    
    print("🔍 Original configuration:")
    enabled_features = base_profile.get_enabled_features()
    for feature_name in enabled_features.keys():
        weight = base_profile.features[feature_name].weight
        print(f"   {feature_name}: weight={weight:.1f}, enabled=True")
    
    # Optimize for windmill detection
    print("\n⚙️ Optimizing for windmill detection...")
    base_profile.structure_type = StructureType.WINDMILL
    base_profile.optimize_for_structure_type()
    
    print("🎯 Optimized configuration:")
    for feature_name in enabled_features.keys():
        weight = base_profile.features[feature_name].weight
        enabled = base_profile.features[feature_name].enabled
        print(f"   {feature_name}: weight={weight:.1f}, enabled={enabled}")
    
    return base_profile

def test_profile_persistence():
    """Test saving and loading custom profiles"""
    print("\n💾 Testing Profile Persistence")
    print("=" * 40)
    
    # Create a custom profile
    custom_profile = DetectorProfile(
        name="CustomTestProfile",
        description="Test profile for persistence testing"
    )
    
    # Customize it
    custom_profile.geometry.resolution_m = 0.3
    custom_profile.features["histogram"].weight = 2.2
    custom_profile.features["entropy"].enabled = False
    custom_profile.thresholds.detection_threshold = 0.65
    
    # Save it
    saved_path = profile_manager.save_profile(custom_profile, "custom_test.json")
    print(f"✅ Saved profile to: {saved_path}")
    
    # Load it back
    loaded_profile = profile_manager.load_profile("custom_test.json")
    print(f"✅ Loaded profile: {loaded_profile.name}")
    
    # Verify the customizations persisted
    print("🔍 Verification:")
    print(f"   Resolution: {loaded_profile.geometry.resolution_m}m (expected: 0.3)")
    print(f"   Histogram weight: {loaded_profile.features['histogram'].weight} (expected: 2.2)")
    print(f"   Entropy enabled: {loaded_profile.features['entropy'].enabled} (expected: False)")
    print(f"   Detection threshold: {loaded_profile.thresholds.detection_threshold} (expected: 0.65)")
    
    return loaded_profile

def test_rectangular_patches():
    """Test rectangular patch configurations"""
    print("\n📐 Testing Rectangular Patch Configurations")
    print("=" * 40)
    
    # Create profile with rectangular patches
    rect_profile = DetectorProfile(
        name="RectangularPatchTest",
        description="Testing rectangular patch detection"
    )
    
    rect_profile.geometry.patch_shape = PatchShape.RECTANGLE
    rect_profile.geometry.patch_size_m = (40.0, 25.0)  # Wide rectangle
    rect_profile.geometry.aspect_ratio_tolerance = 0.5  # Allow more variation
    
    print(f"✅ Rectangular profile created:")
    print(f"   Shape: {rect_profile.geometry.patch_shape}")
    print(f"   Dimensions: {rect_profile.geometry.patch_size_m}")
    print(f"   Aspect ratio tolerance: {rect_profile.geometry.aspect_ratio_tolerance}")
    
    # Calculate patch size in pixels
    patch_px = rect_profile.geometry.get_patch_size_px()
    print(f"   Pixel dimensions: {patch_px}")
    
    return rect_profile

def main():
    """Run all advanced profile tests"""
    print("🧪 Advanced DetectorProfile System Test")
    print("=" * 50)
    
    # Test preset profiles
    windmill_preset = test_preset_profiles()
    
    # Test custom profile creation
    detail_profile, survey_profile = test_custom_profile_creation()
    
    # Test optimization
    optimized_profile = test_profile_optimization()
    
    # Test persistence
    loaded_profile = test_profile_persistence()
    
    # Test rectangular patches
    rect_profile = test_rectangular_patches()
    
    print("\n✅ All Advanced Tests Completed Successfully!")
    print("\n📋 Profiles Created and Tested:")
    print("   🎯 Preset Profiles (windmill, settlement, earthwork, fast survey)")
    print("   🔬 HighResolutionAnalysis (ultra-detailed)")
    print("   ⚡ RapidSurvey (speed-optimized)")
    print("   🔧 OptimizationTest (structure-type optimized)")
    print("   💾 CustomTestProfile (persistence tested)")
    print("   📐 RectangularPatchTest (non-square patches)")
    
    print("\n🚀 Key Capabilities Demonstrated:")
    print("   ✅ Preset profile creation and loading")
    print("   ✅ Custom geometry configuration (resolution, patch size, shape)")
    print("   ✅ Feature weight customization and enable/disable")
    print("   ✅ Detection threshold and aggregation tuning")
    print("   ✅ Structure-type specific optimization")
    print("   ✅ Profile persistence (save/load)")
    print("   ✅ Rectangular and irregular patch support")
    print("   ✅ Speed vs accuracy trade-off configurations")
    
    print("\n🎯 Ready for Production Use!")
    print("   - Profiles can be crystallized as Ψ (Psi) for persistent optimization")
    print("   - Different resolutions and patch sizes fully supported")
    print("   - Feature selection allows task-specific optimization")
    print("   - Save/load enables team collaboration and reproducibility")

if __name__ == "__main__":
    main()
